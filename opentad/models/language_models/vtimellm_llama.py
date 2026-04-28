import json
import os
import traceback
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import auto, Enum

import torch
import torch.nn as nn
import transformers
from torch.nn.modules.module import T
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM, StoppingCriteria, \
    LogitsProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from opentad.models.builder import LANGUAGE_MODELS
from opentad.models.language_models.utils import find_all_linear_names
from opentad.utils.checkpoint import load_lora

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<video>"


class VTimeLLMMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, images,
            time_markers: Optional[list[torch.LongTensor]] = None
    ):
        # print(position_ids, attention_mask)
        # if past_key_values:
        #     print(past_key_values[-1][-1].shape)
        # print(input_ids.shape, position_ids.shape, attention_mask.shape, past_key_values.shape, images)
        if images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
                if self.get_model().config.model_type == 'chatglm':
                    target_shape = past_key_values[-1][-1].shape[0] + 1
                else:
                    target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.get_model().mm_projector(concat_images)
            image_features = self.get_model().mm_dropout(image_features)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            # image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.get_model().mm_projector(images)
            image_features = self.get_model().mm_dropout(image_features)
        # print([image.shape for image in image_features])

        # concat the time markers into image features
        if time_markers is not None and len(time_markers) > 0:
            T = image_features[0].shape[0]
            B = len(image_features)
            image_features_with_time_markers = []

            if type(image_features) is tuple:
                for b in range(B):
                    feat = []
                    T = image_features[b].shape[0]
                    for t in range(T):
                        t_marker = time_markers[t]
                        t_marker = t_marker.squeeze(0).to(image_features[0].device)
                        img_feat = image_features[b][t].unsqueeze(0)
                        feat.append(torch.cat([t_marker, img_feat], dim=0))
                    image_features_with_time_markers.append(torch.cat(feat, dim=0))
                image_features = tuple(image_features_with_time_markers)
            else:
                for t in range(T):
                    t_marker = time_markers[t]
                    t_marker = t_marker.repeat(B, 1, 1).to(image_features.device)
                    img_feat = image_features[:, t, :].unsqueeze(1)
                    image_features_with_time_markers.append(torch.cat([t_marker, img_feat], dim=1))
                image_features = torch.cat(image_features_with_time_markers, dim=1)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            # print(cur_image_idx-1)
            # print(image_features[cur_image_idx-1,0,0:10])
            # print(cur_new_input_embeds[-1,0:10])
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if self.get_model().config.model_type == 'chatglm':
            fake_input_ids = torch.full((new_input_embeds.shape[0], new_input_embeds.shape[1]), -10000,
                                        dtype=new_input_embeds.dtype, device=new_input_embeds.device)
            attention_mask = attention_mask.to(torch.int8)
            new_input_embeds = new_input_embeds.transpose(0, 1).contiguous()
        else:
            fake_input_ids = None
        # print(position_ids, attention_mask)
        return fake_input_ids, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


class VTimeLLMConfig(LlamaConfig):
    model_type = "VTimeLLM"


class VTimeLLMLlamaModel(LlamaModel):
    config_class = VTimeLLMConfig

    def __init__(self, config: LlamaConfig):
        super(VTimeLLMLlamaModel, self).__init__(config)

    def initialize_vision_modules(self, pretrain_mm_mlp_adapter, mm_mlp_adapter_channels, mm_mlp_adapter_dropout=0.1):
        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(mm_mlp_adapter_channels, self.config.hidden_size)
            self.mm_dropout = nn.Dropout(mm_mlp_adapter_dropout)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu', weights_only=True)

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            print("load mlp:", pretrain_mm_mlp_adapter)


class VTimeLLMLlamaForCausalLM(LlamaForCausalLM, VTimeLLMMetaForCausalLM):
    config_class = VTimeLLMConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = VTimeLLMLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.time_markers = None

    def get_model(self):
        return self.model

    def set_time_markers(self, time_markers: list[torch.LongTensor]):
        embeddings = self.model.get_input_embeddings()
        self.time_markers = []
        for time_marker in time_markers:
            new_time_marker = embeddings(time_marker)
            self.time_markers.append(new_time_marker)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                self.time_markers
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("VTimeLLM", VTimeLLMConfig)
AutoModelForCausalLM.register(VTimeLLMConfig, VTimeLLMLlamaForCausalLM)


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)


@LANGUAGE_MODELS.register_module()
class VTimeLLM(nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 merge_lora_path: list = [],  # if it is the stage 3, you should provide the path to the stage 2 model
                 mm_mlp_adapter_channels=768,
                 mm_mlp_adapter_dropout=0.1,
                 mm_mlp_adapter_path=None,  # Both stage2 and stage3 should provide this
                 tune_mm_mlp_adapter: bool = True,  # In the paper, the mm_mlp_adapter is tuned in stage 1
                 tune_llm: bool = False,
                 output_tuning_data_path: str = None,  # leave None if no need for tuning data
                 model_max_length: int = 2048,
                 bits: int = 16,
                 compute_dtype: str = 'bf16',
                 double_quant: bool = True,  # Compress the quantization statistics through double quantization.
                 quant_type: str = 'nf4',  # Quantization data type to use. Should be one of `fp4` or `nf4`.
                 lora_enable: bool = True,
                 lora_r: int = 64,
                 lora_alpha: int = 128,
                 lora_dropout: float = 0.05,
                 lora_bias: str = "none",
                 gradient_checkpointing: bool = True,
                 time_marker_window_size: int = 0,  # The maximum frame window size for the TimeMarker
                 ):
        super(VTimeLLM, self).__init__()
        self.merge_lora_path = merge_lora_path
        self.mm_mlp_adapter_channels = mm_mlp_adapter_channels
        self.mm_mlp_adapter_dropout = mm_mlp_adapter_dropout
        self.mm_mlp_adapter_path = mm_mlp_adapter_path
        self.tune_mm_mlp_adapter = tune_mm_mlp_adapter
        self.tune_llm = tune_llm
        self.output_tuning_data_path = output_tuning_data_path
        self.model_max_length = model_max_length
        self.bits = bits
        self.compute_dtype = compute_dtype
        self.double_quant = double_quant
        self.quant_type = quant_type
        self.lora_enable = lora_enable
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.gradient_checkpointing = gradient_checkpointing
        self.time_marker_window_size = time_marker_window_size

        if output_tuning_data_path is not None and not os.path.exists(output_tuning_data_path):
            os.makedirs(output_tuning_data_path)

        compute_dtype = (torch.float16 if compute_dtype == 'fp16' else (
            torch.bfloat16 if compute_dtype == 'bf16' else torch.float32))
        bnb_model_from_pretrained_args = {}
        if bits in [4, 8]:
            from transformers import BitsAndBytesConfig
            bnb_model_from_pretrained_args.update(dict(
                device_map={"": 'cuda:0'},
                load_in_4bit=bits == 4,
                load_in_8bit=bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=bits == 4,
                    load_in_8bit=bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=double_quant,
                    bnb_4bit_quant_type=quant_type  # {'fp4', 'nf4'}
                )
            ))

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.unk_token
        self.tokenizer = tokenizer

        self.model = VTimeLLMLlamaForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            **bnb_model_from_pretrained_args
        )
        self.model.config.use_cache = False
        self.model.config._attn_implementation = "flash_attention_2"

        if bits in [4, 8]:
            from peft import prepare_model_for_kbit_training
            self.model.config.torch_dtype = (
                torch.float32 if compute_dtype == torch.float16 else compute_dtype)
            self.model = prepare_model_for_kbit_training(self.model,
                                                         use_gradient_checkpointing=gradient_checkpointing)

        if gradient_checkpointing:
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        self.model.get_model().initialize_vision_modules(mm_mlp_adapter_path, mm_mlp_adapter_channels,
                                                         mm_mlp_adapter_dropout)
        if merge_lora_path is not None and len(merge_lora_path) > 0:
            for lora_path in merge_lora_path:
                self.model = load_lora(self.model, lora_path, non_lora_name='non_lora_trainables.bin')
                self.model = self.model.merge_and_unload()

        if not tune_llm:
            self.model.requires_grad_(False)
        for p in self.model.get_model().mm_projector.parameters():
            p.requires_grad = tune_mm_mlp_adapter

        if lora_enable:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=find_all_linear_names(self.model),
                lora_dropout=lora_dropout,
                bias=lora_bias,
                task_type="CAUSAL_LM",
            )
            if hasattr(self.model, "peft_config"):
                del self.model.peft_config

            if bits == 16 and compute_dtype != torch.float32:
                self.model.to(compute_dtype)

            self.model = get_peft_model(self.model, lora_config)

        # freeze the model
        self.model.config.tune_mm_mlp_adapter = tune_mm_mlp_adapter
        self.model.config.freeze_mm_mlp_adapter = not tune_mm_mlp_adapter

        if bits in [4, 8]:
            self.model.get_model().mm_projector.to(dtype=compute_dtype, device='cuda:0')
            from peft.tuners.lora import LoraLayer
            for name, module in self.model.named_modules():
                if isinstance(module, LoraLayer):
                    if compute_dtype == torch.bfloat16:
                        module = module.to(torch.bfloat16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if compute_dtype == torch.bfloat16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)

        self.conv_vicuna_v1 = Conversation(
            system="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the user's questions.",
            roles=("USER", "ASSISTANT"),
            version="v1",
            messages=(),
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )
        self.stop_str = self.conv_vicuna_v1.sep if self.conv_vicuna_v1.sep_style != SeparatorStyle.TWO else self.conv_vicuna_v1.sep2
        self.keywords = [self.stop_str]

        self.allowed_token_ids = self.logits_processor = None
        self.logits_processor_T = 0
        self.prompt_len = -1

    def tokenizer_image_token(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])
        elif tokenizer.name == "GLMTokenizer":
            offset = 2
            input_ids = prompt_chunks[0][:2]

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def forward_train(self, x, masks, gt_segments, gt_labels, metas, curr_epoch, prompt_base, llm_proposal_order,
                      **kwargs):
        if self.time_marker_window_size != 0 and self.model.time_markers is None:
            self.model.set_time_markers(
                [self.tokenizer.encode(f"{i}", add_special_tokens=False, return_tensors='pt').cuda() for i in
                 range(0, self.time_marker_window_size)])

        id_action_map = {i: v for i, v in enumerate(metas[0]['class_map'])}
        (B, T, D) = x.shape

        # prepare the ground truth of VLLMs
        gt_strings = []
        # combine them into a sentence
        for gt_seg, gt_lab in zip(gt_segments, gt_labels):
            if llm_proposal_order == 'certainty':
                # sort the ground truth segments by duration, the longer duration indicates the more confident action
                duration = gt_seg[:, 1] - gt_seg[:, 0]
                sorted_indices = torch.argsort(duration, descending=True)
                gt_seg = gt_seg[sorted_indices]
                gt_lab = gt_lab[sorted_indices]
            elif llm_proposal_order == 'timestamp':
                # sort the ground truth segments by the start timestamp
                sorted_indices = torch.argsort(gt_seg[:, 0])
                gt_seg = gt_seg[sorted_indices]
                gt_lab = gt_lab[sorted_indices]

            gt_seg = gt_seg.cpu().tolist()
            gt_lab = gt_lab.cpu().tolist()
            gt_str = []
            for i in range(len(gt_seg)):
                gt_str.append(f"{id_action_map[gt_lab[i]]}: {int(round(gt_seg[i][0]))}-{int(round(gt_seg[i][1]))}")
            gt_str = ", ".join(gt_str) + "." if len(gt_str) > 0 else ""
            gt_strings.append(gt_str)

        # save the tuning data
        if self.output_tuning_data_path and curr_epoch == 0:
            output_file = os.path.join(self.output_tuning_data_path, 'tuning_data.json')
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    try:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            existing_data = []
                    except json.JSONDecodeError:
                        existing_data = []
            else:
                existing_data = []
            for i in range(B):
                video_name = metas[i]['video_name']
                start_frame = int(metas[i]['window_start_frame'])
                existing_data.append({
                    'video_name': video_name,
                    'start_frame': start_frame,
                    'sample_stride': metas[i]['snippet_stride'],
                    'conversations': [
                        {
                            "from": "human",
                            "value": prompt_base
                        },
                        {
                            "from": "gpt",
                            "value": gt_strings[i]
                        }
                    ],
                })
            with open(output_file, 'w') as f:
                json.dump(existing_data, f, indent=4)
            return {}
        # elif self.output_tuning_data_path:
        #     output_file = os.path.join(self.output_tuning_data_path, 'tuning_data.json')
        #     raise AssertionError(f"Tuning data is successfully saved in the {output_file}.")
        # else:
        #     return {}
        #     raise AssertionError("Please use the tools/train_vlm to run the VLMs.")
        input_ids = []
        pad_token_id = self.tokenizer.pad_token_id

        for b in range(B):
            conversation = self.conv_vicuna_v1.copy()
            conversation.append_message(conversation.roles[0], prompt_base)
            if self.prompt_len == -1:
                conversation_temp = conversation.copy()
                conversation_temp.append_message(conversation_temp.roles[1], None)
                self.prompt_len = len(self.tokenizer_image_token(conversation_temp.get_prompt(), self.tokenizer)) - 1
            conversation.append_message(conversation.roles[1], gt_strings[b])
            prompt = conversation.get_prompt()
            input_ids.append(self.tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt'))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id).cuda()
        attention_mask = (input_ids != pad_token_id).to(input_ids.device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = IGNORE_INDEX
        labels[:, :self.prompt_len] = IGNORE_INDEX

        self.model.train()
        self.model.config.use_cache = False

        results = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            images=x,
            return_dict=True
        )
        logits = results.logits
        loss = results.loss
        return {'LLM_loss': loss}

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.time_marker_window_size != 0 and self.model.time_markers is None:
            self.model.set_time_markers(
                [self.tokenizer.encode(f"{i}", add_special_tokens=False, return_tensors='pt').cuda() for i in
                 range(0, self.time_marker_window_size)])

        # the training forward function is not implemented
        return self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            return_dict=return_dict
        )

    def forward_test(self, x, masks, metas, prompt_base, llm_proposal_order, **kwargs):
        if self.time_marker_window_size != 0 and self.model.time_markers is None:
            self.model.set_time_markers(
                [self.tokenizer.encode(f"{i}", add_special_tokens=False, return_tensors='pt').cuda() for i in
                 range(0, self.time_marker_window_size)])

        # prepare the prompts for VLLMs
        (B, T, D) = x.shape
        conversation = self.conv_vicuna_v1.copy()
        prompt_example = f"{metas[0]['class_map'][0]}: 0-10, {metas[0]['class_map'][-1]}: 21-36, {metas[0]['class_map'][0]}: 16-25."
        conversation.append_message(conversation.roles[0], prompt_base)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()

        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt').unsqueeze(0).expand(B, -1)
        input_ids = input_ids.cuda()

        # # if not the sliding windows dataset, fill the mask with UNK tokens
        # UNK_VECTOR = torch.zeros(x.shape[2]).to(x.device)
        # x[~masks] = UNK_VECTOR  # [B,T,D]

        # limit the available tokens for the model
        if self.logits_processor is None or self.logits_processor_T != T:
            allowed_words = [prompt_example, ' ', ':', '-', ',', '.', ", "]
            allowed_words.extend(metas[0]['class_map'])
            allowed_words.extend([f"{i}" for i in range(T)])
            allowed_words.extend(self.keywords)
            allowed_token_ids = []
            for word in allowed_words:
                tokens = self.tokenizer.encode(word, add_special_tokens=False)
                allowed_token_ids.extend(tokens)
            self.allowed_token_ids = list(set(allowed_token_ids))
            dot_token_id = self.tokenizer.convert_tokens_to_ids(".")
            # giving logit bias to minimize the probability of "." for generating more proposals
            self.logits_processor = WhiteListLogitsProcessor(self.allowed_token_ids, [dot_token_id], [0])
            self.logits_processor_T = T

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                images=x,
                do_sample=False,
                temperature=0,
                num_beams=1,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
                logits_processor=[self.logits_processor] if self.logits_processor is not None else None,
                output_scores=True,
                return_dict_in_generate=True
            )
            output_ids = outputs.sequences
            logits = outputs.scores  # [T][B, V]
            logits = [torch.log_softmax(logit, dim=-1) for logit in logits]
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)

            dot_token_id = self.tokenizer.convert_tokens_to_ids(".")
            comma_token_id = self.tokenizer.convert_tokens_to_ids(",")
            colon_token_id = self.tokenizer.convert_tokens_to_ids(":")
            bar_token_id = self.tokenizer.convert_tokens_to_ids("-")
            stop_token_id = self.tokenizer.convert_tokens_to_ids(self.stop_str)

            proposals = []
            scores = []
            actions = []
            for batch, output in enumerate(outputs):
                # arrange the output in the text format
                output = output.strip()
                if output.endswith(self.stop_str):
                    output = output[:-len(self.stop_str)]

                # arrange the output in the token format
                token_length = torch.where(output_ids[batch, :] == stop_token_id)[0]
                if len(token_length) > 0:
                    token_length = token_length[0].item()
                else:
                    token_length = output_ids.shape[1] - 1
                output_tokens = output_ids[batch, input_token_len:token_length]
                last_token_index = torch.where(output_tokens == dot_token_id)[0]
                if len(last_token_index) > 0:
                    last_token_index = last_token_index[0].item()
                else:
                    last_token_index = output_tokens.shape[0]
                output_tokens = output_tokens[:last_token_index]
                if len(output_tokens) == 0:
                    proposals.append([])
                    scores.append([])
                    actions.append([])
                    continue
                # split the output into segments by the comma
                segments = []
                segment_indexes = []
                comma_indices = torch.where(output_tokens == comma_token_id)[0]
                start = 0
                for comma_index in comma_indices:
                    if comma_index <= start:
                        continue
                    segments.append(output_tokens[start:comma_index])
                    comma_index = comma_index.item()
                    segment_indexes.append((start, comma_index))
                    start = comma_index + 1
                if start < last_token_index:
                    segments.append(output_tokens[start:last_token_index])
                    segment_indexes.append((start, last_token_index))
                # split the segments into the action and the start-end frame
                colon_indices = [torch.where(seg == colon_token_id)[0] for seg in segments]
                colon_indices = [seg[0].item() if len(seg) > 0 else None for seg in colon_indices]
                # [N_Seg][L_action]
                action_segments = []
                action_segments_indexes = []
                for seg, colon_index, segment_index in zip(segments, colon_indices, segment_indexes):
                    if colon_index is not None and colon_index > 0:
                        action_segments.append(seg[:colon_index])
                        action_segments_indexes.append((segment_index[0], segment_index[0] + colon_index))
                    else:
                        action_segments.append(None)
                        action_segments_indexes.append((None, None))
                bar_indices = [torch.where(seg == bar_token_id)[0] for seg in segments]
                bar_indices = [seg[0].item() if len(seg) > 0 else None for seg in bar_indices]
                # [N_Seg][L_tokenNum]
                start_segments = []
                end_segments = []
                start_segments_indexes = []
                end_segments_indexes = []
                for seg, colon_index, bar_index, segment_index in zip(segments, colon_indices, bar_indices,
                                                                      segment_indexes):
                    if colon_index is not None and bar_index is not None and bar_index > colon_index + 1 and \
                            segment_index[
                                0] + bar_index + 1 < segment_index[1]:
                        start_segments.append(seg[colon_index + 1:bar_index])
                        start_segments_indexes.append(
                            (segment_index[0] + colon_index + 1, segment_index[0] + bar_index))
                        end_segments.append(seg[bar_index + 1:])
                        end_segments_indexes.append((segment_index[0] + bar_index + 1, segment_index[1]))
                    else:
                        start_segments.append(None)
                        start_segments_indexes.append((None, None))
                        end_segments.append(None)
                        end_segments_indexes.append((None, None))

                # compute the probabilities of each prediction
                scores_action = []
                scores_start = []
                scores_end = []
                last_action_score = 0
                proposals.append([])
                scores.append([])
                actions.append([])
                for seg_i in range(len(segments)):
                    try:
                        if action_segments[seg_i] is None or start_segments[seg_i] is None or end_segments[
                            seg_i] is None:
                            continue
                        start_num = int(self.tokenizer.decode(start_segments[seg_i]))
                        end_num = int(self.tokenizer.decode(end_segments[seg_i]))
                        if start_num >= end_num or start_num < 0 or end_num >= T:
                            continue
                        scores_action.extend(
                            compute_scores(logits, batch, action_segments_indexes[seg_i][0],
                                           action_segments_indexes[seg_i][1],
                                           target=[action_segments[seg_i]]))
                        start_boundaries = compute_acceptable_boundaries(start_num)
                        start_boundaries = [
                            self.tokenizer.encode(str(i), return_tensors='pt', add_special_tokens=False)[0]
                            for i in start_boundaries]
                        start_scores = compute_scores(logits, batch, start_segments_indexes[seg_i][0],
                                                      start_segments_indexes[seg_i][1],
                                                      target=start_boundaries)
                        start_scores = torch.cat([t.unsqueeze(0) for t in start_scores], dim=0)
                        a = len(start_boundaries)
                        weights = torch.linspace(-3, 3, a)
                        weights = torch.exp(-0.5 * weights ** 2)
                        weights = weights / weights.sum()
                        weights = weights.cuda()
                        start_scores = torch.sum(start_scores * weights)
                        scores_start.append(start_scores)
                        end_boundaries = compute_acceptable_boundaries(end_num)
                        end_boundaries = [
                            self.tokenizer.encode(str(i), return_tensors='pt', add_special_tokens=False)[0]
                            for i in end_boundaries]
                        end_scores = compute_scores(logits, batch, end_segments_indexes[seg_i][0],
                                                    end_segments_indexes[seg_i][1],
                                                    target=end_boundaries)
                        end_scores = torch.cat([t.unsqueeze(0) for t in end_scores], dim=0)
                        end_scores = torch.sum(end_scores * weights)
                        scores_end.append(end_scores)

                        # score = HasActionProb * ThisActionProb * StartProb * EndProb
                        scores[-1].append(last_action_score + scores_action[-1] + start_scores + end_scores)
                        last_action_score = scores_action[-1]
                        proposals[-1].append(torch.tensor([start_num, end_num]))
                        actions[-1].append(self.tokenizer.decode(action_segments[seg_i]))
                    except ValueError or IndexError as e:
                        print("error while sorting the predictions:",e)
                        traceback.print_exc()
                        continue

                output = output.strip()
                print(f"Video: {metas[batch]['video_name']} & Frame: {metas[batch]['window_start_frame']}")
                print(output)

            # end of for batch
            for i, sco in enumerate(scores):
                if llm_proposal_order == 'certainty':
                    scores[i] = torch.cat([t.unsqueeze(0) for t in sco], dim=0) if len(sco) > 0 else torch.tensor([])
                    scores[i] = torch.exp(scores[i])
                    scores[i] = torch.softmax(scores[i], dim=0) / 2 + 0.5
                    # scores[i] = torch.linspace(0.5, 1.0, scores[i].shape[0]).cuda()
                else:
                    scores[i] = torch.ones((len(sco), 1), device=sco[0].device,
                                           dtype=sco[0].dtype) if sco else torch.tensor([])
            for i, prop in enumerate(proposals):
                proposals[i] = torch.cat([t.unsqueeze(0) for t in prop], dim=0) if len(prop) > 0 else torch.tensor([])

        ret_dict={
            'proposals': proposals,
            'scores': scores,
            'actions': actions
        }
        return ret_dict

    def eval(self: T) -> T:
        super().eval()
        self.model.eval()
        return self


def compute_acceptable_boundaries(x, boundary=2):
    """
    Compute the acceptable boundaries for the given number.
    e.g. 25 -> 23, 24, 25, 26, 27
    e.g. 20 -> 20, 21, 22, 21, 22
    e.g. 29 -> 27, 28, 29, 28, 27
    """
    x_a = x % 10

    result = list(range(x - boundary, x + boundary + 1))
    length = len(result)

    for i, r in enumerate(result):
        if r % 10 != x_a:
            result[i] = result[length - i - 1]
    return result


def compute_scores(logits, batch, index_from, index_to, target=[]):
    if index_from is None or index_to is None:
        return [None for _ in target]
    target_scores = [torch.tensor(0.0).cuda() for _ in target]
    logits = logits[index_from:index_to]
    for target_i in range(len(target)):
        for index_i in range(target[target_i].shape[0]):
            if index_i >= len(logits):
                break
            target_scores[target_i] += logits[index_i][batch, target[target_i][index_i]]
    return target_scores


class WhiteListLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids, offset_token_ids, offsets):
        self.allowed_token_ids = allowed_token_ids
        self.offset_token_ids = offset_token_ids
        self.offsets = offsets

    def __call__(self, input_ids, scores):
        mask = torch.ones_like(scores) * -float("inf")
        mask[:, self.allowed_token_ids] = 0
        for offset_token_id, offset in zip(self.offset_token_ids, self.offsets):
            mask[:, offset_token_id] = offset
        scores = scores + mask
        return scores

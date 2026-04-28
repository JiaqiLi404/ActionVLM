from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import torch
from peft import get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, GenerationConfig, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast
from mmengine.model import BaseModel

from opentad.models.builder import LANGUAGE_MODELS
from opentad.models.language_models.utils import (
    enable_gradient_checkpointing,
    find_all_linear_names,
    guess_load_checkpoint,
    make_inputs_require_grad,
)


@LANGUAGE_MODELS.register_module()
class Qwen3VL_Slowfast(BaseModel):
    def __init__(
        self,
        model_path,
        quantization_vit=False,
        quantization_llm=False,
        pretrained_pth=None,
        special_tokens=None,
        downsample_ratio=None,
        image_size=None,
    ):
        super().__init__()
        self.freeze_llm = False
        self.freeze_visual_encoder = False
        self.use_llm_lora = False
        self.use_visual_encoder_lora = False
        self.quantization_vit = quantization_vit
        self.quantization_llm = quantization_llm
        self.special_tokens = list(special_tokens or [])

        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="right")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            config=self.config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        if downsample_ratio is not None:
            self.config.downsample_ratio = downsample_ratio
        if image_size is not None:
            self.config.force_image_size = image_size

        self._initialize(pretrained_pth)

    def _initialize(self, pretrained_pth=None):
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        if self.special_tokens:
            self.add_special_tokens(self.special_tokens)

        llm_module = self.get_llm_module()
        if hasattr(llm_module, "enable_input_require_grads"):
            llm_module.enable_input_require_grads()
        else:
            input_embeddings = (
                llm_module.get_input_embeddings() if hasattr(llm_module, "get_input_embeddings") else self.model.get_input_embeddings()
            )
            input_embeddings.register_forward_hook(make_inputs_require_grad)

        self.gradient_checkpointing_enable()

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)

    def add_special_tokens(self, special_tokens):
        new_tokens = [token for token in special_tokens if token not in self.special_tokens]
        if not new_tokens:
            return 0
        self.special_tokens.extend(new_tokens)
        num_new_tokens = self.tokenizer.add_tokens(new_tokens, special_tokens=True)
        if num_new_tokens > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        return num_new_tokens

    def get_llm_module(self):
        if hasattr(self.model, "language_model"):
            return self.model.language_model
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            return self.model.model.language_model
        return self.model

    def set_llm_module(self, module):
        if hasattr(self.model, "language_model"):
            self.model.language_model = module
            return
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            self.model.model.language_model = module
            return
        raise AttributeError("Qwen3VL model does not expose a replaceable language model module")

    def get_visual_module(self):
        if hasattr(self.model, "visual"):
            return self.model.visual
        if hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
            return self.model.model.visual
        return None

    def set_visual_module(self, module):
        if hasattr(self.model, "visual"):
            self.model.visual = module
            return
        if hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
            self.model.model.visual = module
            return
        raise AttributeError("Qwen3VL model does not expose a replaceable visual module")

    def get_llm_state_prefixes(self):
        prefixes = []
        if hasattr(self.model, "language_model"):
            prefixes.append("model.language_model.")
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            prefixes.append("model.model.language_model.")
        return prefixes

    def get_visual_state_prefixes(self):
        prefixes = []
        if hasattr(self.model, "visual"):
            prefixes.append("model.visual.")
        if hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
            prefixes.append("model.model.visual.")
        return prefixes

    def call_freeze_llm(self):
        self.get_llm_module().requires_grad_(False)
        self.freeze_llm = True

    def call_freeze_visual_encoder(self):
        visual_module = self.get_visual_module()
        if visual_module is not None:
            visual_module.requires_grad_(False)
        self.freeze_visual_encoder = True

    def call_unfreeze_llm(self):
        self.get_llm_module().requires_grad_(True)
        self.freeze_llm = False

    def call_unfreeze_visual_encoder(self):
        visual_module = self.get_visual_module()
        if visual_module is not None:
            visual_module.requires_grad_(True)
        self.freeze_visual_encoder = False

    def call_llm_lora(self, llm_lora):
        assert not self.quantization_llm, "Quantization is not supported in LORA"
        self.use_llm_lora = True
        self._prepare_llm_for_lora(llm_lora)

    def call_visual_encoder_lora(self, visual_encoder_lora):
        assert not self.quantization_vit, "Quantization is not supported in LORA"
        self.use_visual_encoder_lora = True
        self._prepare_visual_encoder_for_lora(visual_encoder_lora)

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        llm_module = prepare_model_for_kbit_training(self.get_llm_module(), use_activation_checkpointing)
        if lora_config.target_modules is None:
            lora_config.target_modules = find_all_linear_names(llm_module)
        self.set_llm_module(get_peft_model(llm_module, lora_config))

    def _prepare_visual_encoder_for_lora(self, lora_config):
        visual_module = self.get_visual_module()
        if visual_module is None:
            raise AttributeError("Qwen3VL model does not expose a visual module for LoRA")
        if lora_config.target_modules is None:
            lora_config.target_modules = find_all_linear_names(visual_module)
        self.set_visual_module(get_peft_model(visual_module, lora_config))

    def _post_init(self, fast_pool_size=4, fast_pool=False):
        return

    def _build_inputs_embeds(self, input_ids, visual_features):
        if visual_features is None:
            return self.model.get_input_embeddings()(input_ids)

        visual_features = visual_features.to(self.model.dtype)
        input_embeds = self.model.get_input_embeddings()(input_ids).clone().to(visual_features.dtype)
        batch_size, seq_len, hidden_size = input_embeds.shape
        input_embeds = input_embeds.reshape(batch_size * seq_len, hidden_size)
        flat_ids = input_ids.reshape(batch_size * seq_len).to(input_embeds.device)
        selected = flat_ids == self.model.img_context_token_id

        visual_features = visual_features.reshape(-1, hidden_size)
        selected_total = int(selected.sum().item())
        if selected_total > visual_features.shape[0]:
            repeat_times = (selected_total + visual_features.shape[0] - 1) // visual_features.shape[0]
            visual_features = visual_features.repeat(repeat_times, 1)
        input_embeds[selected] = visual_features[:selected_total]
        return input_embeds.reshape(batch_size, seq_len, hidden_size)

    def _pool_split_visual_features(self, split_features, grid_thw):
        if split_features is None or grid_thw is None:
            return None

        pooled_features = []
        for embeds, grid in zip(split_features, grid_thw.detach().cpu().tolist()):
            if embeds is None or embeds.numel() == 0:
                continue
            time_steps = max(int(grid[0]), 1)
            time_steps = min(time_steps, embeds.shape[0])
            if embeds.shape[0] % time_steps == 0:
                tokens_per_frame = max(embeds.shape[0] // time_steps, 1)
                pooled = embeds.reshape(time_steps, tokens_per_frame, embeds.shape[-1]).mean(dim=1)
            else:
                pooled = torch.stack([chunk.mean(dim=0) for chunk in torch.chunk(embeds, time_steps, dim=0)], dim=0)
            pooled_features.append(pooled)

        if not pooled_features:
            return None
        return pad_sequence(pooled_features, batch_first=True)

    def _split_flat_visual_features(self, flat_features, grid_thw):
        if flat_features is None or grid_thw is None or flat_features.dim() != 2:
            return None

        token_counts = [max(int(grid[0] * grid[1] * grid[2]), 1) for grid in grid_thw.detach().cpu().tolist()]
        total_tokens = sum(token_counts)
        if total_tokens <= 0:
            return None

        if flat_features.shape[0] == total_tokens:
            scaled_counts = token_counts
        else:
            scale = total_tokens / max(int(flat_features.shape[0]), 1)
            scaled_counts = [max(int(round(count / scale)), 1) for count in token_counts]
            diff = flat_features.shape[0] - sum(scaled_counts)
            if scaled_counts:
                scaled_counts[-1] += diff
            if scaled_counts[-1] <= 0 or sum(scaled_counts) != flat_features.shape[0]:
                return None

        return list(torch.split(flat_features, scaled_counts, dim=0))

    def encode_video_visual(self, videos):
        if videos is None or not hasattr(self, "processor") or getattr(self.processor, "video_processor", None) is None:
            return None, None

        if videos.dim() == 5:
            videos = [video for video in videos.detach().cpu()]
        else:
            videos = videos.detach().cpu()

        video_inputs = self.processor.video_processor(videos=videos, return_tensors="pt")
        pixel_values_videos = video_inputs.get("pixel_values_videos", None)
        video_grid_thw = video_inputs.get("video_grid_thw", None)
        if pixel_values_videos is None or video_grid_thw is None:
            return None, None

        device = self.model.device
        pixel_values_videos = pixel_values_videos.to(device)
        video_grid_thw = video_grid_thw.to(device)
        video_output = self.model.get_video_features(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        if isinstance(video_output, tuple):
            video_features = video_output[0]
        elif hasattr(video_output, "pooler_output") and video_output.pooler_output is not None:
            video_features = video_output.pooler_output
        elif hasattr(video_output, "last_hidden_state") and video_output.last_hidden_state is not None:
            video_features = video_output.last_hidden_state
        else:
            video_features = video_output

        if isinstance(video_features, torch.Tensor):
            video_features = self._split_flat_visual_features(video_features, video_grid_thw)
        elif not isinstance(video_features, (list, tuple)):
            video_features = None

        pooled_features = self._pool_split_visual_features(video_features, video_grid_thw)
        return pooled_features, video_grid_thw

    def forward(
        self,
        x=None,
        visual_features=None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        visual_features = visual_features if visual_features is not None else x
        input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        if position_ids is not None:
            position_ids = position_ids.to(self.model.device)
        if labels is not None:
            labels = labels.to(self.model.device)
        if visual_features is not None:
            visual_features = visual_features.to(self.model.device)

        inputs_embeds = self._build_inputs_embeds(input_ids, visual_features)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=False if use_cache is None else use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values=None,
        visual_features=None,
        input_ids=None,
        attention_mask=None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
    ):
        visual_features = visual_features if visual_features is not None else pixel_values
        input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        if visual_features is not None:
            visual_features = visual_features.to(self.model.device)

        inputs_embeds = self._build_inputs_embeds(input_ids, visual_features)
        return self.model.generate(
            inputs=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=True,
            use_cache=False,
            **generate_kwargs,
        )

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        enable_gradient_checkpointing(self.get_llm_module())

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.get_llm_module().gradient_checkpointing_disable()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        if self.use_visual_encoder_lora:
            visual_module = self.get_visual_module()
            if visual_module is not None:
                to_return.update(get_peft_model_state_dict(visual_module, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            visual_prefixes = tuple(self.get_visual_state_prefixes())
            to_return.update({k: v for k, v in state_dict.items() if k.startswith(visual_prefixes)})

        if self.use_llm_lora:
            to_return.update(get_peft_model_state_dict(self.get_llm_module(), state_dict=state_dict))
        elif not self.freeze_llm:
            llm_prefixes = tuple(self.get_llm_state_prefixes())
            to_return.update({k: v for k, v in state_dict.items() if k.startswith(llm_prefixes)})

        to_return.update({k: v for k, v in state_dict.items() if "model.lm_head." in k})
        return to_return

    def init_weights(self):
        pass

from collections import OrderedDict
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from einops import rearrange, reduce
from torch.nn import CrossEntropyLoss
from peft import get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import (AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig, GenerationConfig)
from mmengine import print_log
from mmengine.model import BaseModel

from opentad.models.builder import LANGUAGE_MODELS
from opentad.models.language_models.utils import (
    enable_gradient_checkpointing,
    find_all_linear_names,
    guess_load_checkpoint,
    make_inputs_require_grad,
)


class InternVL_V1_5(BaseModel):
    '''
    Modified from InternVL_V1_5 in xtuner.models.
    '''

    def __init__(
            self,
            model_path,
            quantization_vit=False,
            quantization_llm=False,
            pretrained_pth=None,
            downsample_ratio=None,
            image_size=None,
    ):
        print_log("Start to load InternVL_V1_5 model.", logger="current")
        super().__init__()
        self.freeze_llm = False
        self.freeze_visual_encoder = False
        self.use_llm_lora = False
        self.use_visual_encoder_lora = False
        self.quantization_vit = quantization_vit
        self.quantization_llm = quantization_llm

        if model_path is not None:
            self.config = config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='right')
        else:
            self.config = config = AutoConfig.from_pretrained('OpenGVLab/InternVL3-2B', trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL3-2B', trust_remote_code=True,
                                                           padding_side='right')
        if downsample_ratio is not None:
            config.downsample_ratio = downsample_ratio
        if image_size is not None:
            config.force_image_size = image_size
            config.vision_config.image_size = image_size
        if hasattr(config, "llm_config") and config.llm_config.model_type == "internlm2":
            config.llm_config.attn_implementation = "flash_attention_2"

        self.model = AutoModel.from_config(
            torch_dtype=torch.bfloat16,
            config=config,
            trust_remote_code=True
        )

        self._initialize(pretrained_pth)
        self._count = 0

    def _initialize(self, pretrained_pth=None):
        img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = img_context_token_id

        if hasattr(self.model.language_model, "enable_input_require_grads"):
            self.model.language_model.enable_input_require_grads()
        else:
            self.model.language_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

        self.gradient_checkpointing_enable()

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f"Load pretrained weight from {pretrained_pth}")

    def call_freeze_llm(self):
        self.model.language_model.requires_grad_(False)
        self.freeze_llm = True

    def call_freeze_visual_encoder(self):
        self.model.vision_model.requires_grad_(False)
        self.freeze_visual_encoder = True

    def call_unfreeze_llm(self):
        self.model.language_model.requires_grad_(True)
        self.freeze_llm = False

    def call_unfreeze_visual_encoder(self):
        self.model.vision_model.requires_grad_(True)
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
        self.model.language_model = prepare_model_for_kbit_training(
            self.model.language_model, use_activation_checkpointing
        )
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.language_model)
            lora_config.target_modules = modules
        self.model.language_model = get_peft_model(
            self.model.language_model, lora_config
        )

    def _prepare_visual_encoder_for_lora(self, lora_config):
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.vision_model)
            lora_config.target_modules = modules
        self.model.vision_model = get_peft_model(self.model.vision_model, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        enable_gradient_checkpointing(self.model.language_model)

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.model.language_model.gradient_checkpointing_disable()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.model.vision_model, state_dict=state_dict
                )
            )
        elif not self.freeze_visual_encoder:
            to_return.update(
                {k: v for k, v in state_dict.items() if "model.vision_model." in k}
            )
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.model.language_model, state_dict=state_dict
                )
            )
        elif not self.freeze_llm:
            to_return.update(
                {k: v for k, v in state_dict.items() if "model.language_model." in k}
            )
        # Step 3. Projector
        to_return.update({k: v for k, v in state_dict.items() if "model.mlp1." in k})
        return to_return

    def init_weights(self):
        pass

    def forward(self, data, data_samples=None, mode="loss"):
        pixel_values = data["pixel_values"]

        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            # b*n, c, h, w
            concat_images = torch.cat(
                [image.to(self.model.vision_model.dtype) for image in pixel_values],
                dim=0,
            )
        else:
            raise NotImplementedError()

        input_ids = data["input_ids"]
        position_ids = data["position_ids"]
        attention_mask = data["attention_mask"]
        # sum is 0 are text
        image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
        image_flags = image_flags.long()

        labels = data["labels"]
        use_cache = False

        # Directly calling this code in LORA fine-tuning
        # will result in an error,so we must rewrite it.
        # TODO: Once the official is fixed, we can remove it.
        # outputs = self.model(input_ids=input_ids,
        #                      position_ids=position_ids,
        #                      attention_mask=attention_mask,
        #                      image_flags=image_flags,
        #                      pixel_values=concat_images,
        #                      labels=labels,
        #                      use_cache=use_cache)
        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=concat_images,
            labels=labels,
            use_cache=use_cache,
        )
        loss_dict = {"loss": outputs.loss}
        return loss_dict

    def _llm_forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict
            if return_dict is not None
            else self.model.config.use_return_dict
        )

        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error.
        input_embeds = self.model.language_model.get_input_embeddings()(
            input_ids
        ).clone()

        vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.get_rank() == 0 and self._count % 100 == 0:
            print(
                f"dynamic ViT batch size: {vit_batch_size}, "
                f"images per sample: {vit_batch_size / B}, "
                f"dynamic token length: {N}"
            )
        self._count += 1

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.model.img_context_token_id
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
                -1, C
            )
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(
                f"warning: {e}, input_embeds[selected].shape="
                f"{input_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.model.language_model.config.vocab_size
            )
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@LANGUAGE_MODELS.register_module()
class InternVL_Slowfast(InternVL_V1_5):
    '''
    Modified from Sa2VA.
    '''

    def __init__(self,
                 model_path,
                 quantization_vit=False,
                 quantization_llm=False,
                 pretrained_pth=None,
                 special_tokens=None,
                 downsample_ratio=None,
                 image_size=None,
                 ):
        self.special_tokens = special_tokens
        super().__init__(
            model_path=model_path,
            quantization_vit=quantization_vit,
            quantization_llm=quantization_llm,
            pretrained_pth=pretrained_pth,
            downsample_ratio=downsample_ratio,
            image_size=image_size,
        )

        self.transfer_to_hf = False

    def _initialize(self, pretrained_pth=None):
        super()._initialize(pretrained_pth)
        if self.special_tokens is not None:
            self._add_special_tokens(self.special_tokens)

        if hasattr(self.model.language_model, 'enable_input_require_grads'):
            self.model.language_model.enable_input_require_grads()
        else:
            self.model.language_model.get_input_embeddings(
            ).register_forward_hook(make_inputs_require_grad)

    def _add_special_tokens(self, special_tokens):
        num_new_tokens = self.tokenizer.add_tokens(
            special_tokens, special_tokens=True)

        if num_new_tokens > 0:
            self.model.language_model.resize_token_embeddings(len(self.tokenizer))

    def _post_init(self, fast_pool_size=4, fast_pool=True):
        if fast_pool:
            self.fast_pool = nn.AdaptiveAvgPool2d((fast_pool_size, fast_pool_size))
        return

    def forward(self,
                x,
                input_ids,
                position_ids,
                attention_mask,
                labels,
                vp_overall_mask=None,
                prompt_masks=None,
                fast_pixel_values=None,
                fast_token_idx=None
                ):
        if fast_pixel_values:
            assert fast_token_idx is not None
            if type(fast_pixel_values) is list or fast_pixel_values.ndim == 5:
                if type(fast_pixel_values) is list:
                    fast_pixel_values = [
                        x.unsqueeze(0) if x.ndim == 3 else x for x in fast_pixel_values
                    ]
                # b*n, c, h, w
                fast_concat_images = torch.cat(
                    [image.to(self.model.vision_model.dtype) for image in fast_pixel_values], dim=0)
            else:
                raise NotImplementedError()
        else:
            fast_concat_images = None

        if isinstance(x, list) or x.ndim == 5:
            if isinstance(x, list):
                x = [img.unsqueeze(0) if img.ndim == 3 else img for img in x]
            x = torch.cat(
                [img.to(self.model.vision_model.dtype) for img in x], dim=0
            )
            image_flags = torch.sum(x, dim=(1,2,3)) != 0
        else:
            # sum is 0 are text
            image_flags = torch.sum(x, dim=2) != 0

        image_flags = image_flags.long()

        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=x,
            labels=labels,
            use_cache=False,
            output_hidden_states=True,
            fast_pixel_values=fast_concat_images,
            fast_token_idx=fast_token_idx,
            vp_overall_mask=vp_overall_mask,
            prompt_masks=prompt_masks,
        )

        return outputs

    def _llm_forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            fast_pixel_values=None,
            fast_token_idx=None,
            vp_overall_mask=None,
            prompt_masks=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.model.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error.
        input_embeds = self.model.language_model.get_input_embeddings()(
            input_ids).clone().to(pixel_values.dtype)

        if fast_pixel_values is not None:
            n_fast_images = fast_pixel_values.shape[0]
            whole_pixel_values = torch.cat([fast_pixel_values, pixel_values], dim=0)
            vit_embeds = self.model.extract_feature(whole_pixel_values)
            vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
            fast_vit_embeds = vit_embeds[:n_fast_images]  # (n_fast_images, hw, c)
            _size = int(fast_vit_embeds.shape[1] ** 0.5)
            fast_vit_embeds = fast_vit_embeds.reshape(fast_vit_embeds.shape[0], _size, _size, fast_vit_embeds.shape[-1])
            # pooling
            fast_vit_embeds = fast_vit_embeds.permute(0, 3, 1, 2)  # (n_fast_images, c, h, w)
            fast_vit_embeds = self.fast_pool(fast_vit_embeds).flatten(2)  # (n_fast_images, c, hw)
            fast_vit_embeds = fast_vit_embeds.permute(0, 2, 1)
            vit_embeds = vit_embeds[n_fast_images:]
        elif pixel_values.dim() == 3:
            vit_embeds = pixel_values
            fast_vit_embeds = None
        else:
            vit_embeds = self.model.extract_feature(pixel_values)
            vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
            fast_vit_embeds = None

        vit_embeds = vit_embeds[image_flags == 1]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        self._count += 1

        if vp_overall_mask is not None and prompt_masks is not None:
            vp_embeds = []
            vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
            prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

            vp_overall_mask = vp_overall_mask[image_flags == 1]
            overall_tile_vit_embeds = vit_embeds[vp_overall_mask]  # (n_img, hw, c)

            i_vp_img = 0
            for i_img in range(len(vit_embeds)):
                vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                if vp_overall_mask[i_img]:
                    tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
                    objects_prompt_masks = prompt_masks[i_vp_img]
                    n_obj = len(objects_prompt_masks)
                    tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                    objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                    vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                    i_vp_img += 1
            vp_embeds = torch.cat(vp_embeds, dim=0)
        else:
            vp_embeds = None

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.model.img_context_token_id)

        if vp_embeds is None:
            try:
                input_embeds[selected] = vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vit_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vit_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vit_embeds) + 1
                    vit_embeds = torch.cat([vit_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vit_embeds[:n_token]
        else:
            try:
                input_embeds[selected] = vp_embeds.reshape(-1, C)
            except Exception as e:
                vp_embeds = vp_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vp_embeds.shape={vp_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vp_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vp_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vp_embeds) + 1
                    vp_embeds = torch.cat([vp_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vp_embeds[:n_token]

        if fast_vit_embeds is not None:
            selected = (input_ids == fast_token_idx)
            selected_tot = selected.sum().item()
            if selected_tot > fast_vit_embeds.shape[0] * fast_vit_embeds.shape[1]:
                assert selected_tot % (fast_vit_embeds.shape[0] * fast_vit_embeds.shape[1]) == 0
                repeat_times = selected_tot / (fast_vit_embeds.shape[0] * fast_vit_embeds.shape[1])
                fast_vit_embeds = fast_vit_embeds.repeat(int(repeat_times), 1, 1)
            try:
                input_embeds[selected] = fast_vit_embeds.reshape(-1, C)
            except Exception as e:
                fast_vit_embeds = fast_vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[fast_selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'fast_vit_embeds.shape={fast_vit_embeds.shape}')
                n_token = selected.sum()
                input_embeds[selected] = fast_vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.model.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            fast_token_idx=None,
            fast_pixel_values=None,
            prompt_masks=None,
            vp_overall_mask=None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        device = self.model.device
        assert self.model.img_context_token_id is not None

        if fast_pixel_values is not None:
            assert fast_token_idx is not None
            if type(fast_pixel_values) is list or fast_pixel_values.ndim == 5:
                if type(fast_pixel_values) is list:
                    fast_pixel_values = [
                        x.unsqueeze(0) if x.ndim == 3 else x for x in fast_pixel_values
                    ]
                # b*n, c, h, w
                fast_pixel_values = torch.cat(
                    [image.to(self.model.vision_model.dtype) for image in fast_pixel_values], dim=0)

        if pixel_values is not None:
            fast_vit_embeds = None
            if visual_features is not None:
                vit_embeds = visual_features
                image_flags = torch.sum(vit_embeds, dim=2) != 0
            else:
                if type(pixel_values) is list or pixel_values.ndim == 5:
                    if type(pixel_values) is list:
                        pixel_values = [
                            x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                        ]
                    # b*n, c, h, w
                    pixel_values = torch.cat(
                        [image.to(self.model.vision_model.dtype) for image in pixel_values], dim=0)

                if fast_pixel_values is not None:
                    n_fast_images = fast_pixel_values.shape[0]
                    whole_pixel_values = torch.cat([fast_pixel_values, pixel_values], dim=0)
                    vit_embeds = self.model.extract_feature(whole_pixel_values.to(device))
                    # vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
                    fast_vit_embeds = vit_embeds[:n_fast_images]  # (n_fast_images, hw, c)
                    _size = int(fast_vit_embeds.shape[1] ** 0.5)
                    fast_vit_embeds = fast_vit_embeds.reshape(fast_vit_embeds.shape[0], _size, _size,
                                                              fast_vit_embeds.shape[-1])
                    # pooling
                    fast_vit_embeds = fast_vit_embeds.permute(0, 3, 1, 2)  # (n_fast_images, c, h, w)
                    fast_vit_embeds = self.fast_pool(fast_vit_embeds).flatten(2)  # (n_fast_images, c, hw)
                    fast_vit_embeds = fast_vit_embeds.permute(0, 2, 1)
                    vit_embeds = vit_embeds[n_fast_images:]
                else:
                    vit_embeds = self.model.extract_feature(pixel_values.to(device))
                image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
            image_flags = image_flags.long()
            vit_embeds = vit_embeds[image_flags == 1]

            input_embeds = self.model.language_model.get_input_embeddings()(input_ids.to(device))
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            if vp_overall_mask is not None and prompt_masks is not None:
                vp_embeds = []
                vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
                prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

                vp_overall_mask = vp_overall_mask[image_flags == 1]
                overall_tile_vit_embeds = vit_embeds[vp_overall_mask]  # (n_img, hw, c)

                i_vp_img = 0
                for i_img in range(len(vit_embeds)):
                    vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                    if vp_overall_mask[i_img]:
                        tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
                        objects_prompt_masks = prompt_masks[i_vp_img]
                        n_obj = len(objects_prompt_masks)
                        tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                        objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                        vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                        i_vp_img += 1
                vp_embeds = torch.cat(vp_embeds, dim=0)
            else:
                vp_embeds = None

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.model.img_context_token_id)
            assert selected.sum() != 0
            if vp_embeds is None:
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            else:
                if len(input_embeds[selected]) != len(vp_embeds.reshape(-1, C)):
                    print("Shape mismatch, selected is {}, vp embeds is {} !!!" \
                          .format(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C))))
                    min_tokens = min(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C)))
                    input_embeds[selected][:min_tokens] = vp_embeds.reshape(-1, C)[:min_tokens].to(input_embeds.device)
                else:
                    input_embeds[selected] = vp_embeds.reshape(-1, C).to(input_embeds.device)

            if fast_vit_embeds is not None:
                selected = (input_ids == fast_token_idx)
                # FIXME, add repeat.
                assert selected.sum() != 0
                input_embeds[selected] = fast_vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        outputs = self.model.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=False,
            **generate_kwargs,
        )

        outputs.vit_embeds = vp_embeds if vp_embeds is not None else vit_embeds
        return outputs

    def vision_forward(self, x: torch.FloatTensor, temporal_positional_embedding=None):
        # (b n c h w) or (b c h w)
        device = self.model.device
        B = None
        if x.dim() == 5:
            B = x.shape[0]
            x = rearrange(x, 'b n c h w -> (b n) c h w')

        res = self.model.vision_model(pixel_values=x.to(device))
        x = res.pooler_output

        # vit_embeds = res.last_hidden_state[:,1:,:].mean(dim=1)  # [B, C]

        # x=self.model.extract_feature(x)
        # x=x.mean(dim=1)  # [B, C]

        if B is not None:
            x = rearrange(x, '(b n) c -> b n c', b=B)
            T = x.shape[1]
        else:
            T = x.shape[0]

        if temporal_positional_embedding is None:
            pos_embed = temporal_positional_embedding[:T, :].unsqueeze(0).to(device)
            x = x + pos_embed
        return x

    def state_dict(self, *args, **kwargs):
        if self.transfer_to_hf:
            state_dict = super(InternVL_V1_5, self).state_dict(*args, **kwargs)
            return state_dict
        else:
            return super().state_dict(*args, **kwargs)

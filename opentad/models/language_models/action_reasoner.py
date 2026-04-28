import os
import re
from typing import Literal

import torch
import torch.nn.functional as F
from mmaction.models import ActionDataPreprocessor
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, get_peft_model_state_dict
from mmengine.model import BaseModel
from einops import rearrange, reduce
from torch.nn.utils.rnn import pad_sequence

from opentad.models.backbones.vit import VisionTransformerCP_Large, VisionTransformerCP_Base, VisionTransformerCP_Small
from opentad.models.builder import LANGUAGE_MODELS
from opentad.models.language_models.llava.llava_arch import DEFAULT_VID_START_TOKEN, DEFAULT_VIDEO_PATCH_TOKEN, \
    DEFAULT_VID_END_TOKEN, DEFAULT_SLOW_VID_START_TOKEN, DEFAULT_SLOW_VID_END_TOKEN, IGNORE_INDEX
from opentad.models.language_models.llava_qwen import LlavaQwenForCausalLM
from opentad.models.language_models.utils import enable_gradient_checkpointing
from opentad.utils import rank0_print


@LANGUAGE_MODELS.register_module()
class ActionReasoner(BaseModel):
    def __init__(self,
                 videomae_version: Literal['v1_s', 'v1_b', 'v1_l', 'v2_s', 'v2_b', "iv_2_5"] = 'v2_s',
                 mllm_hf_name_or_path: str = None,
                 mllm_lora_path: str = None,
                 mllm_embedding_dim: int = 384,  # for mllm embedding dimension
                 clip_window=768,  # clip the sequence before VLM for saving the VRM
                 torch_dtype='bf16',
                 **kwargs
                 ):
        super(ActionReasoner, self).__init__()

        # Initialize the VideoMAE model
        vit_pretrained_files = {
            'v1_s': 'pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth',
            'v1_b': 'pretrained/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth',
            'v1_l': 'pretrained/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth',
            'v2_s': 'pretrained/vit_s_k710_dl_from_giant.pth',
            'v2_b': 'pretrained/vit_b_k710_dl_from_giant.pth',
            'iv_2_5': "/mnt/e/OneDrive - wqa/Models/InternVL_2_5_HiCo_R16"
        }

        # Load InternVideo2.5
        if videomae_version == "iv_2_5":
            self.backbone = "InternVideo_2_5"
            self.vit = AutoModel.from_pretrained(vit_pretrained_files[videomae_version],
                                                 trust_remote_code=True).half().cuda()
            self.vit = self.vit.vision_model
            self.backbone_embedding_dim = 1024
            mllm_embedding_dim=1024
        # Load VideoMAE
        else:
            self.backbone = 'VideoMAE'
            if videomae_version.endswith('l'):
                self.vit = VisionTransformerCP_Large()
            elif videomae_version.endswith('b'):
                self.vit = VisionTransformerCP_Base()
            elif videomae_version.endswith('s'):
                self.vit = VisionTransformerCP_Small()
            else:
                raise ValueError(f"Unknown videomae_version: {videomae_version}")

            state_dict = torch.load(vit_pretrained_files[videomae_version])
            state_dict = state_dict['module'] if videomae_version.startswith('v2_') else state_dict
            for key in list(state_dict.keys()):
                if key.startswith('backbone.'):
                    new_key = key.replace('backbone.', '')
                    state_dict[new_key] = state_dict.pop(key)
                if "fc1" in key:
                    new_key = key.replace("fc1", "layers.0.0")
                    state_dict[new_key] = state_dict.pop(key)
                if "fc2" in key:
                    new_key = key.replace("fc2", "layers.1")
                    state_dict[new_key] = state_dict.pop(key)
                if "patch_embed.proj." in key:
                    new_key = key.replace("patch_embed.proj.", "patch_embed.projection.")
                    state_dict[new_key] = state_dict.pop(key)
            res = self.vit.load_state_dict(state_dict, strict=False)
            print("Missing keys for VideoMAE:", res.missing_keys)
            print("Unexpected keys for VideoMAE:", res.unexpected_keys)
            self.backbone_embedding_dim = self.vit.embed_dims

        torch.cuda.empty_cache()
        self.mllm_embedding_dim = mllm_embedding_dim
        self.clip_window = clip_window

        self.data_preprocessor = ActionDataPreprocessor(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW")

        self.torch_dtype = torch.float32

        # self.conversations = Conversations.__getattribute__(Conversations, stage.upper())
        self.mllm = None
        # if mllm_hf_name_or_path is not None:
        #     self.tokenizer, self.mllm, self.image_processor, self.max_length = load_lora_model(
        #         mllm_lora_path,
        #         mllm_hf_name_or_path,
        #         "llava_qwen",
        #         device_map="cpu",
        #         overwrite_config={
        #             "num_spatial_tokens": 100,
        #             "num_temporal_tokens": 100
        #         }
        #     )  # Add any other thing you want to pass in llava_model_args
        #     self.vision_config = self.mllm.model.vision_config
        #     self.mllm.config.max_frame = 100

    def forward_train(self,
                      x,
                      masks,
                      gt_segments,
                      gt_labels,
                      metas,
                      curr_epoch,
                      prompt_base,
                      stage,
                      dtype,
                      **kwargs):
        self.torch_dtype = dtype
        if stage == 0:
            x = self.extract_feat_from_video(x, masks)
        else:
            x = self.forward(x, metas, prompt_base, **kwargs)

        pass_dict = {
            'x': x,
            'masks': masks,
            'gt_segments': gt_segments,
            'gt_labels': gt_labels,
            'metas': metas,
            'curr_epoch': curr_epoch,
            'prompt_base': prompt_base,
            'dtype': dtype,
        }
        pass_dict.update(kwargs)
        return pass_dict

    def forward_test(self,
                     x,
                     masks,
                     metas,
                     prompt_base,
                     stage,
                     dtype,
                     **kwargs):
        self.torch_dtype = dtype
        if stage == 0:
            x = self.extract_feat_from_video(x, masks)
        else:
            x = self.forward(x, metas, prompt_base, **kwargs)

        pass_dict = {
            'x': x,
            'masks': masks,
            'metas': metas,
            'prompt_base': prompt_base,
            'dtype': dtype,
        }
        pass_dict.update(kwargs)

        return pass_dict

    def forward(self,
                x,
                metas,
                prompt_base,
                answers=None,
                **kwargs):

        if self.mllm is None:
            return x

        ''' Prepare the input for MLLM '''
        return x

    def llava_forward(self, x_query, text_query):
        ''' Prepare the input for MLLM '''
        system_message: str = "You are a helpful assistant."
        replace_token = ""
        replace_token += (DEFAULT_VID_START_TOKEN +
                          DEFAULT_VIDEO_PATCH_TOKEN *
                          (self.vision_config.fast_token_num *
                           self.vision_config.fast_frame_num - 2) +
                          DEFAULT_VID_END_TOKEN)
        replace_token += (DEFAULT_SLOW_VID_START_TOKEN +
                          DEFAULT_VIDEO_PATCH_TOKEN *
                          (self.vision_config.slow_token_num *
                           self.vision_config.slow_frame_num - 2) +
                          DEFAULT_SLOW_VID_END_TOKEN)

        roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

        im_start, im_end = self.tokenizer.additional_special_tokens_ids
        nl_tokens = self.tokenizer("\n").input_ids
        _system = self.tokenizer("system").input_ids + nl_tokens

        # Apply prompt templates
        input_ids, targets = [], []
        for j, sentence in enumerate(text_query):
            input_id, target = [], []
            system = [im_start] + _system + self.tokenizer(system_message).input_ids + [im_end] + nl_tokens
            input_id += system
            target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
            assert len(input_id) == len(target)

            role = "<|im_start|>user"
            sentence = sentence.replace(DEFAULT_VIDEO_TOKEN, replace_token)
            _input_id = (self.tokenizer(role).input_ids + nl_tokens
                         + self.tokenizer(sentence).input_ids + [im_end] + nl_tokens)
            input_id += _input_id

            # if role == "<|im_start|>user":
            #     _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
            # elif role == "<|im_start|>assistant":
            #     _target = ([im_start] + [IGNORE_INDEX] * len(self.tokenizer(role).input_ids)
            #                + _input_id[len(self.tokenizer(role).input_ids) + 1:-2] + [im_end] + nl_tokens)
            # else:
            #     raise NotImplementedError
            # target += _target

            input_ids.append(input_id)
            targets.append(target)

        input_ids = [torch.tensor(x, dtype=torch.long, device=x_query[0].device) for x in input_ids]
        targets = [torch.tensor(x, dtype=torch.long, device=x_query[0].device) for x in targets]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        targets = pad_sequence(targets, batch_first=True, padding_value=IGNORE_INDEX)

        variables = {
            "temporal_input_locations": [],
            "temporal_output_locations": [],
            "spatial_height_input_locations": [],
            "spatial_height_output_locations": [],
            "spatial_width_input_locations": [],
            "spatial_width_output_locations": []
        }
        variables = [variables for _ in range(len(input_ids))]
        modalities = ['video' for _ in range(len(input_ids))]

        output_ids = self.mllm.generate(
            input_ids,
            images=x_query,  # type: ignore
            do_sample=True,
            temperature=0.01,
            top_p=None,
            num_beams=1,
            # no_repeat_ngram_size=3,
            variables=variables,
            modalities=modalities,
            max_new_tokens=1024,
            use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids,
                                              skip_special_tokens=False)[0]
        outputs = outputs.replace("<|im_end|>", "")

        def replace_and_normalize(input_str, return_token=False):
            pattern = re.compile(r'(<WIDTH-(\d+)>|<HEIGHT-(\d+)>|<TEMP-(\d+)>)')

            def normalize(match):
                if match.group(2):
                    value = int(match.group(2))
                elif match.group(3):
                    value = int(match.group(3))
                elif match.group(4):
                    value = int(match.group(4))

                normalized_value = value / 99.0

                if return_token:
                    return '{:d},'.format(value)
                return '{:.5f},'.format(normalized_value)

            # 使用 re.sub 进行替换，调用 normalize 函数进行处理
            result_str = re.sub(pattern, normalize, input_str)

            return result_str.replace(",]", "]").replace(",}", "}")

        try:
            converted_outputs = replace_and_normalize(outputs)
        except Exception:
            converted_outputs = None

    def extract_feat_from_video(self, x, masks):
        window_size = x.shape[3]
        if self.clip_window <= 0 or window_size % self.clip_window != 0:
            print(f"Warning: window size {window_size} is not divisible by clip window {self.clip_window}, "
                  f"the clip window will be used as the window size.")
            self.clip_window = window_size

        x = x.squeeze(1)  # [B, C, T, H, W]
        frames = self.get_frames_from_video(x, sample_type='none', resize=None, norm=True).to(self.torch_dtype)
        B, C, T, H, W = x.shape

        if self.backbone == 'VideoMAE':
            ''' VidewoMAE '''
            chunk_num = window_size // 16
            frames = rearrange(frames, "b (t1 t) c h w -> (b t1) c t h w", t1=chunk_num)
            x = self.vit(frames)
            x = reduce(x, 'b c t h w -> b c t', 'mean')  # [B, C, T]
            x = rearrange(x, '(b t1) c t -> b c (t1 t)', t1=chunk_num)
            x = F.interpolate(x, size=window_size, mode="linear", align_corners=False)
        elif self.backbone == 'InternVideo_2_5':
            frames = rearrange(frames, "b t c h w -> (b t) c h w")
            pixel_values = frames

            # InternVideo2.5 forward
            with torch.no_grad():
                vision_out = self.vit(pixel_values)

            x = vision_out.last_hidden_state[:, 1:, :]
            x = x.mean(dim=1)
            x = rearrange(x, "(b t) c -> b c t", b=B, t=T)
            if x.shape[-1] != window_size:
                x = F.interpolate(x, size=window_size, mode="linear", align_corners=False)

        return x

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(state_dict, strict, assign)

    def all_state_dict(self, *args, **kwargs):
        state_dict = super(self).state_dict(*args, **kwargs)
        return state_dict

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        from collections import OrderedDict

        if self.mllm is None:
            return state_dict

        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.mllm.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.mllm.model.vision_model, state_dict=state_dict))
        elif not self.mllm.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
        # Step 2. LLM
        if self.mllm.use_llm_lora:
            if self.arch_type == 'intern_vl':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict)
                )
            elif self.arch_type == 'qwen':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.model, state_dict=state_dict)
                )
            elif self.arch_type == 'llava':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict)
                )
        elif not self.mllm.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mlp1.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'model.multi_modal_projector.' in k})

        # Step 4. mask decoder of grounding model (SAM/SAM2)
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mask_decoder' in k})

        # Step 5. others (fcs)
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'text_hidden_fcs.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'text_exist_fcs.' in k}
        )
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'lm_head.weight' in k or 'output' in k and 'sam2_model' not in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'embed_tokens.weight' in k or 'tok_embeddings' in k})
        return to_return

    def get_frames_from_video(self, frames, n_frames=5, sample_type="uniform", resize=None, norm=False):
        """
        frames: [B, C, T, H, W]
        return: [B, T, C, H, W]
        """
        B, C, T, H, W = frames.shape
        device = frames.device

        def process(frames):
            # frames: [B, T, C, H, W]
            if resize is not None and (H, W) != resize:
                frames = frames.reshape(B * T, C, H, W)  # [B*T, C, H, W]
                frames = F.interpolate(frames, size=resize, mode='bilinear', align_corners=False)
                frames = frames.view(B, T, C, *resize)  # [B, T, C, H, W]
            if norm:
                frames = rearrange(frames, 'N T C H W -> N C T H W')
                frames = self.data_preprocessor.preprocess(list(frames), data_samples=None, training=False)[0]
                frames = rearrange(frames, ' N C T H W -> N T C H W')
            return frames

        if sample_type == "uniform":
            idxs = torch.linspace(0, T - 1, steps=n_frames, device=device).long()
            frames = frames.index_select(2, idxs)

        frames = frames.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]

        return process(frames)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        enable_gradient_checkpointing(self.mllm.model.language_model)

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        if self.arch_type == 'qwen':
            self.mllm.model.model.gradient_checkpointing_disable()
        else:
            self.mllm.model.language_model.gradient_checkpointing_disable()

    def process_rpn_head(self, rpn_head):
        if 'projection' in rpn_head and rpn_head['projection'] is not None:
            rpn_head['projection']['in_channels'] = self.mllm_embedding_dim
            print(f"RPN head in_channels is modified to: {rpn_head['projection']['in_channels']}", flush=True)
        return rpn_head


def load_pretrained_model(model_path, load_8bit=False, load_4bit=False, device_map="auto",
                          attn_implementation="flash_attention_2", customized_config=None, overwrite_config=None,
                          **kwargs):
    kwargs["device_map"] = device_map

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                           bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    else:
        kwargs["torch_dtype"] = torch.float16

    if customized_config is not None:
        kwargs["config"] = customized_config

    # Load LLaVA model
    rank0_print(f"Loaded LLaVA model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llava_cfg = AutoConfig.from_pretrained(model_path)
    llava_cfg.vocab_size = llava_cfg.text_config.vocab_size
    if overwrite_config is not None:
        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(llava_cfg, k, v)
    model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                                 attn_implementation=attn_implementation,
                                                 config=llava_cfg, **kwargs)
    rank0_print(f"Model Class: {model.__class__.__name__}")

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != "auto":
        vision_tower.to(device="cuda", dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    model.model.init_vision_config(tokenizer)

    return tokenizer, model, image_processor, context_len


def load_lora_model(lora_path, model_base, load_8bit=False, load_4bit=False,
                    device_map="auto", attn_implementation="flash_attention_2", customized_config=None,
                    overwrite_config=None, **kwargs):
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_base, load_8bit,
                                                                           load_4bit, device_map, attn_implementation,
                                                                           customized_config, overwrite_config,
                                                                           **kwargs)
    if lora_path is None:
        lora_path = []
    if type(lora_path) == str:
        lora_path = [lora_path, ]
    for lora in lora_path:
        print(f"Loading LoRA: {lora}")
        print(f"Loading additional LLaVA weights...")
        if "checkpoint-" in lora:
            # model.initialize_embedings(6,model.config.vocab_size+6)
            pattern = r"checkpoint-(\d*)"
            num = re.search(pattern, lora).group(1)
            non_lora_trainables = torch.load(os.path.join(lora, f"global_step{num}/mp_rank_00_model_states.pt"),
                                             map_location=next(model.parameters()).device)["module"]

        else:
            if os.path.exists(os.path.join(lora, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(os.path.join(lora, "non_lora_trainables.bin"),
                                                 map_location=next(model.parameters()).device)
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(lora, "non_lora_trainables.bin")
        # non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
        # if any(k.startswith("model.model.") for k in non_lora_trainables):
        #     non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
        # info = model.load_state_dict(non_lora_trainables, strict=False)

        if "base_model.model.model.embed_tokens.weight" in non_lora_trainables:
            if model.model.embed_tokens.weight.shape[0] != \
                    non_lora_trainables["base_model.model.model.embed_tokens.weight"].shape[0]:
                new_token_num = non_lora_trainables["base_model.model.model.embed_tokens.weight"].shape[0] - \
                                model.model.embed_tokens.weight.shape[0]
                cur_token_num = non_lora_trainables["base_model.model.model.embed_tokens.weight"].shape[0]
                model.initialize_embedings(new_token_num, cur_token_num)

        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(model, lora)
        info = model.load_state_dict(non_lora_trainables, strict=False)
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
        print("Model is loaded...")

    return tokenizer, model, image_processor, context_len


class Conversations:
    DATASET_SYNTHESIS = {
        "BaseballPitch": [
            "<video>\nWhen does the athlete prepare and winds up with the baseball in hand in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete initiate the pitching motion in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the ball is released and thrown toward the batter in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "BasketballDunk": [
            "<video>\nWhen does the athlete jump towards the hoop while holding the basketball in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the ball is forcefully pushed through the basket in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "Billiards": [
            "<video>\nWhen does the athlete position the cue stick aiming at the cue ball in the video? Please describe the location of the corresponding player in this video.",
            "<video>\nWhen does the athlete strike the cue ball with the stick in the video? Please describe the location of the corresponding player in this video.",
            "<video>\nWhen does the ball move forward by the striking in the video? Please describe the location of the corresponding player in this video.",
        ],
        "CleanAndJerk": [
            "<video>\nWhen does the athlete grip the barbell on the ground in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete lift the barbell to the shoulders in the 'clean' phase in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete explosively lift the barbell overhead in the 'jerk' phase in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete hold the barbell steady overhead with full control in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "CliffDiving": [
            "<video>\nWhen does the athlete stand at the edge of a cliff or platform in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete jump off and perform acrobatic movements while descending in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete enter the water below in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "CricketBowling": [
            "<video>\nWhen does the athlete take a run-up toward the wicket in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete release the ball with a straight arm towards the batsman in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the ball reach the batsman or wicketkeeper in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "CricketShot": [
            "<video>\nWhen does the athlete prepare the bat stance as the ball approaches in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete swing the bat to hit the ball in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the ball is struck and sent away from the batsman in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "Diving": [
            "<video>\nWhen does the athlete run or stand at the edge of a diving board or platform in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete jump and perform a controlled descent into the water in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete enter the water in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "FrisbeeCatch": [
            "<video>\nWhen does the athlete track the flying frisbee in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete position their hands or body to intercept the frisbee in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete successfully grasp or trap the frisbee in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "GolfSwing": [
            "<video>\nWhen does the athlete address the golf ball in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete swing the club in a controlled arc to strike the ball in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the ball launch toward the target in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "HammerThrow": [
            "<video>\nWhen does the athlete grip the hammer in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete spin and build momentum in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete release the hammer into the throwing sector in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "HighJump": [
            "<video>\nWhen does the athlete run towards the bar in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete jump towards the bar at the designated point in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete land on the mat after clearing the bar in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "JavelinThrow": [
            "<video>\nWhen does the athlete run forward with the javelin in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete begin the throwing motion in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete release the javelin into the air in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "LongJump": [
            "<video>\nWhen does the athlete begin running along the track in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete take off from the takeoff board in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete land in the sandpit in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "PoleVault": [
            "<video>\nWhen does the athlete sprint down the runway holding the pole in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete plant the pole into the vault box and propel upward over the bar in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete land safely on the mat in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "Shotput": [
            "<video>\nWhen does the athlete position the shot near the neck in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete push and launch the shot forward in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the shot land on the ground in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "SoccerPenalty": [
            "<video>\nWhen does the athlete approach the ball placed at the penalty mark in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete strike the ball towards the goal in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "TennisSwing": [
            "<video>\nWhen does the athlete prepare their stance and track the incoming ball in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete swing the racket to strike the ball in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the ball get hit and sent back over the net in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "ThrowDiscus": [
            "<video>\nWhen does the athlete grip the discus in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete perform a spinning motion to gain momentum in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete release the discus into the air in the video? Please describe the location of the corresponding athlete in this video.",
        ],
        "VolleyballSpiking": [
            "<video>\nWhen does the athlete jump near the net in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the athlete swing their arm forcefully to hit the ball downward over the net in the video? Please describe the location of the corresponding athlete in this video.",
            "<video>\nWhen does the ball cross into the opponent’s court in the video? Please describe the location of the corresponding athlete in this video.",
        ],
    }

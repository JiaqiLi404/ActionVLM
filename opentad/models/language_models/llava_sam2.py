import copy
import os.path
from collections import OrderedDict
from typing import Literal
from types import MethodType

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig, AutoModel, AutoConfig
from transformers.cache_utils import Cache
from peft import get_peft_model_state_dict, LoraConfig
from mmengine.model import BaseModel
from mmdet.models.losses import CrossEntropyLoss, DiceLoss
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmcv.ops import point_sample
from pycocotools import mask as _mask
from flash_attn.modules.mha import MHA
from einops import rearrange

from opentad.models.builder import LANGUAGE_MODELS, build_backbone, build_language_model
from opentad.models.language_models.utils import (
    enable_gradient_checkpointing,
    get_stop_criteria,
    guess_load_checkpoint,
)
from opentad.models.language_models.utils.templates import PROMPT_TEMPLATE


@LANGUAGE_MODELS.register_module()
class VideoLLaVASAMModel(BaseModel):
    def __init__(self,
                 mllm,
                 grounding_encoder,
                 tune_llm=False,
                 tune_visual_encoder=False,
                 lora_llm_enable=True,
                 lora_visual_enable=False,
                 lora_r: int = 64,
                 lora_alpha: int = 128,
                 lora_dropout: float = 0.05,
                 lora_bias: Literal["none", "all", "lora_only"] = "none",
                 torch_dtype='bf16',
                 pretrained_pth=None,
                 hf_model_path=None,
                 frozen_sam2_decoder=False,
                 loss_sample_points=True,
                 num_points=12544,
                 fast_pool=False,  # for slow fast arch
                 fast_pool_size=4,
                 use_fast_supervision=False,
                 phi3=True,  # for inference
                 template=None,
                 arch_type: Literal['intern_vl', 'qwen', 'llava'] = 'intern_vl',  # for arch selection
                 bs: int = 0,  # bs
                 sam2_prediction_path: str = None,  # If need to save the Sa2VA prediction
                 llm_feature=False,
                 aggregated_features=['sam2_maskmem_features'],
                 **kwargs
                 ):
        super(VideoLLaVASAMModel, self).__init__()

        # for multi image process
        self.downsample_ratio = 0.5  # 0.5
        self.image_size = 448  # 448
        self.patch_size = 14
        mllm['downsample_ratio'] = self.downsample_ratio
        mllm['image_size'] = self.image_size
        self.model = self.mllm = build_language_model(mllm)
        self.hf_model_path = hf_model_path
        self.tune_llm = tune_llm
        self.tune_visual_encoder = tune_visual_encoder

        self.lora_llm_enable = lora_llm_enable
        self.lora_visual_enable = lora_visual_enable
        self.lora_enable = self.lora_llm_enable or self.lora_visual_enable
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.lora_config = LoraConfig(r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout,
                                      bias=self.lora_bias)

        self.fast_pool = fast_pool
        self.fast_pool_size = fast_pool_size
        self.arch_type = arch_type
        self.tokenizer = self.mllm.tokenizer
        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        self.grounding_encoder = build_backbone(grounding_encoder, with_wrapper=False)
        self.grounding_encoder.requires_grad_(False)
        self.frozen_sam2_decoder = frozen_sam2_decoder
        if not frozen_sam2_decoder:
            self.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)

        if torch_dtype == 'bf16':
            self.torch_dtype = torch.bfloat16
        elif torch_dtype == 'fp16':
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
        if self.arch_type == 'intern_vl':
            in_dim = self.mllm.model.config.llm_config.hidden_size
        elif self.arch_type == 'qwen':
            in_dim = self.mllm.model.config.hidden_size
        elif self.arch_type == 'llava':
            # for llava, the hidden size is in language model
            in_dim = self.mllm.model.language_model.config.hidden_size
        out_dim = self.grounding_encoder.hidden_dim
        self.patch_token = int((self.image_size // self.patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = ''
            self.IMG_END_TOKEN = ''
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )
        self.transformer_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(self.torch_dtype)
        self.transformer_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(self.torch_dtype)
        self.loss_mask = CrossEntropyLoss(use_sigmoid=True, reduction='mean', loss_weight=2.0)
        self.loss_dice = DiceLoss(
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=0.5)
        if use_fast_supervision:
            self.text_exist_fcs = nn.Sequential(
                nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
                nn.Linear(in_dim, 1), nn.Dropout(0.0)
            )
            self.loss_exists = CrossEntropyLoss(use_sigmoid=True, reduction='mean', loss_weight=1.0)
        if fast_pool:
            self.fast_token_idx = self.tokenizer("<FAST_IMG_CONTEXT>", add_special_tokens=False).input_ids[0]
        else:
            self.fast_token_idx = None

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.num_frames = 5
        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75
        self.use_fast_supervision = use_fast_supervision
        self.phi3 = phi3
        self.template = template
        self.init_prediction_config = False
        self.bs = bs
        self.sam2_prediction_path = sam2_prediction_path
        if self.sam2_prediction_path is not None:
            os.makedirs(self.sam2_prediction_path, exist_ok=True)
        self.sam2_prediction_files = None
        self.aggregated_features = aggregated_features

        self.hf_config = None
        if self.hf_model_path is not None:
            self.hf_model = AutoModel.from_pretrained(
                self.hf_model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            ).eval()
            self.hf_config = AutoConfig.from_pretrained(self.hf_model_path, trust_remote_code=True)
            self.hf_state_dict = self.hf_model.state_dict()
            self.hf_mllm_state_dict = OrderedDict()
            self.hf_sam_state_dict = OrderedDict()
            self.hf_text_fcs_state_dict = OrderedDict()
            self.hf_other_state_dict = OrderedDict()
            for k, v in self.hf_state_dict.items():
                if 'language_model' in k or 'vision_model' in k or 'mlp1' in k:
                    self.hf_mllm_state_dict[k] = v
                elif 'sam2' in k:
                    self.hf_sam_state_dict[k] = v
                elif 'text_hidden_fcs' in k:
                    self.hf_text_fcs_state_dict[k] = v
                else:
                    self.hf_other_state_dict[k] = v
            assert len(self.hf_other_state_dict) == 0, "hf model has unused state_dict: {}".format(
                self.hf_other_state_dict.keys())
            self.mllm.model.load_state_dict(self.hf_mllm_state_dict)

        if not self.tune_llm:
            self.mllm.call_freeze_llm()
        if not self.tune_visual_encoder:
            self.mllm.call_freeze_visual_encoder()
        if self.lora_llm_enable:
            self.mllm.call_llm_lora(self.lora_config)
        if self.lora_visual_enable:
            self.mllm.call_visual_encoder_lora(self.lora_config)

        if hasattr(self.mllm, '_post_init'):
            self.mllm._post_init(
                fast_pool_size=self.fast_pool_size,
                fast_pool=self.fast_pool
            )
        else:
            print("No _post_init() in mllm !!!")
        if self.lora_llm_enable:
            if self.arch_type == 'intern_vl':
                self.mllm.model.language_model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.language_model.base_model.model.get_output_embeddings().requires_grad_(True)
            elif self.arch_type == 'qwen':
                self.mllm.model.model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.get_output_embeddings().weight.requires_grad_(True)
            elif self.arch_type == 'llava':
                self.mllm.model.language_model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.language_model.base_model.model.get_output_embeddings().requires_grad_(True)

        if self.hf_model_path is not None:
            self.load_state_dict(self.hf_sam_state_dict, strict=False)
            self.load_state_dict(self.hf_text_fcs_state_dict, strict=False)
            print(f'Load hf model weight from {self.hf_model_path}')
            del self.hf_model
            del self.hf_state_dict
            del self.hf_mllm_state_dict
            del self.hf_other_state_dict
            del self.hf_text_fcs_state_dict
            del self.hf_sam_state_dict
            if llm_feature:
                del self.grounding_encoder, self.text_hidden_fcs
                self.grounding_encoder = None
                self.text_hidden_fcs = None
            torch.cuda.empty_cache()
        else:
            pass
            # del self.model, self.mllm, self.grounding_encoder, self.text_hidden_fcs
            # self.mllm = None
            # self.grounding_encoder = None
            # self.text_hidden_fcs = None

        self.aggregation_dim = 1024
        if 'sam2_maskmem_features' in self.aggregated_features:
            # self.maskmem_attn=nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
            maskmem_embed_dim = 128
            self.maskmem_attn = MHA(
                embed_dim=maskmem_embed_dim,
                num_heads=8,
                dropout=0.1,
                causal=False,
                use_flash_attn=True
            )
            self.maskmem_proj1 = nn.Linear(1024, maskmem_embed_dim)
            self.maskmem_proj2 = nn.Linear(maskmem_embed_dim, self.aggregation_dim)
        if 'sam2_vision_features' in self.aggregated_features:
            self.sam_vision_proj = nn.Linear(4096, self.aggregation_dim)

        self.llm_feature = llm_feature

    def preparing_for_generation(self, max_new_tokens=2048, torch_dtype=torch.bfloat16):
        # set stop criteria and generation configs for model
        self.bot_name = 'BOT'
        if self.hf_config is not None:
            self.template = self.hf_config.template
            self.template = self.template.replace('-', '_')
        else:
            self.template = 'phi3_chat'
        self.template = self.conv_template = PROMPT_TEMPLATE[self.template]
        stop_words = self.template.get('STOP_WORDS', [])
        self.stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)

        default_generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
            temperature=0.01
        )

        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True
        self.torch_dtype = torch_dtype
        self.mllm.to(self.torch_dtype)
        self.text_hidden_fcs.to(self.torch_dtype) if self.text_hidden_fcs is not None else None

        # change phi3 prepare for generation fuction
        if (self.hf_config is None and self.phi3) or self.hf_config.llm_config.architectures[0] == 'Phi3ForCausalLM':
            self.mllm.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_phi3, self.mllm)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.seg_token_idx = self.tokenizer.convert_tokens_to_ids('[SEG]')

    def forward_train(self,
                      x,
                      masks,
                      gt_segments,
                      gt_labels,
                      metas,
                      curr_epoch,
                      raw_inputs,
                      prompt_base,
                      **kwargs):
        if self.sam2_prediction_path is not None:
            ret_dict = self.predict_video(metas, raw_inputs, prompt_base, whole_video=True, **kwargs)
            torch.cuda.empty_cache()
            raw_inputs = ret_dict['features']
        if self.llm_feature:
            ret_dict = self.llm_process_video(metas, raw_inputs, prompt_base, whole_video=True, **kwargs)

        x = self.aggregate_features(x, raw_inputs, metas, trunc=False) if self.aggregated_features is not None and len(
            self.aggregated_features) > 0 else x

        pass_dict = {
            'x': x,
            'masks': masks,
            'gt_segments': gt_segments,
            'gt_labels': gt_labels,
            'metas': metas,
            'curr_epoch': curr_epoch,
            'prompt_base': prompt_base,
        }
        pass_dict.update(kwargs)
        return pass_dict

    def forward_test(self,
                     x,
                     masks,
                     metas,
                     raw_inputs,
                     prompt_base,
                     **kwargs):
        if self.sam2_prediction_path is not None:
            ret_dict = self.predict_video(metas, raw_inputs, prompt_base, whole_video=True, **kwargs)
            torch.cuda.empty_cache()
            raw_inputs = ret_dict['features']
        x = self.aggregate_features(x, raw_inputs, metas, trunc=False) if self.aggregated_features is not None and len(
            self.aggregated_features) > 0 else x

        pass_dict = {
            'x': x,
            'masks': masks,
            'metas': metas,
            'prompt_base': prompt_base,
        }
        pass_dict.update(kwargs)

        return pass_dict

    def aggregate_features(self, x, feat_dict, metas, trunc=False):
        """ pass a proper x that aggregates the SA2VA features """
        win_size = x.shape[2]
        windows = [(max(meta['trunc_start'], 0), meta['trunc_start'] + meta['trunc_len']) for meta in
                   metas] if trunc else None
        new_features = []  # [Feat] [B,C,T]
        for key in self.aggregated_features:
            assert key == "x" or key in feat_dict[0], f"Key {key} not found in features dict"
            if key == 'sam2_maskmem_features':
                feat = []
                for b_i in range(len(feat_dict)):
                    feat_temp = feat_dict[b_i][key]
                    feat_temp = rearrange(feat_temp, 'b c (h p1) (w p2) -> b (c p1 p2) (h w)', p1=4, p2=4)
                    feat.append(feat_temp)
                    del feat_dict[b_i][key]
                feat = torch.stack(feat, dim=0)  # (B,T,C,S)
                feat = feat.permute(1, 0, 3, 2)  # (T, B, S, C)
                feat = torch.nan_to_num(feat, nan=0.0)
                T, B, S, C = feat.shape
                new_feat = []
                T_b = min(T, 2304)
                for t in range(0, T, T_b):
                    feat_temp = feat[t:t + T_b, :, :, :].to(dtype=x.dtype, device=x.device)  # (T_b,B, S, C)
                    t_t = feat_temp.shape[0]
                    feat_temp = rearrange(feat_temp, 't b s c -> (t b) s c')
                    feat_temp = self.maskmem_proj1(feat_temp)  # (B, S, C‘)
                    feat_temp = self.maskmem_attn(feat_temp)
                    feat_temp = self.maskmem_proj2(feat_temp.mean(dim=1))  # (B, C)
                    feat_temp = rearrange(feat_temp, '(t b) c -> b c t', t=t_t)
                    new_feat.append(feat_temp)
                new_feat = torch.cat(new_feat, dim=2)
                new_features.append(new_feat)
            elif key == 'x':
                continue
            else:
                feat = []
                for b_i in range(len(feat_dict)):
                    feat.append(feat_dict[b_i][key])
                    del feat_dict[b_i][key]
                feat = torch.stack(feat, dim=0).to(dtype=x.dtype, device=x.device)  # ( B,T,C)
                feat = self.sam_vision_proj(feat)
                new_features.append(feat.permute(0, 2, 1))  # (B,C,T)
            if trunc:
                new_features[-1] = [t[windows[i][0]:windows[i][1]] for i, t in enumerate(new_features[-1])]
        # concat the new features after x
        new_features = [x] + new_features if 'x' in self.aggregated_features else new_features
        x = torch.cat(new_features, dim=1)
        # check if nan in x
        if torch.isnan(x).any():
            print("x has nan !!!")
        return x

    def forward(self, data, data_samples=None, mode='loss'):
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        input_ids = data['input_ids']
        fast_exists = data.pop('fast_exists', None)
        # if self.arch_type == 'llava' and data.get('pixel_values', None) is not None:
        #     data['pixel_values'] = data['pixel_values'].to(self.torch_dtype)
        if self.fast_pool:
            output = self.mllm(data, data_samples, mode, fast_token_idx=self.fast_token_idx)
        else:
            output = self.mllm(data, data_samples, mode)
        if gt_masks is None:
            # require zero seg datas
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype,
                device=input_ids.device,
            )
        else:
            seg_valid = True

        assert frames_per_batch, "Video Lisa require frames_per_batch !!!"
        # print('frmaes_per_batch: ', frames_per_batch)
        ori_size_list = []
        for i_bs, mask in enumerate(gt_masks):
            mask_shape = mask.shape[-2:]
            ori_size_list += [mask_shape] * frames_per_batch[i_bs]

        seg_token_mask = input_ids == self.seg_token_idx

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])

        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero

        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += 5

        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = []
        for item in pred_embeddings_list_:
            if len(item) != 0:
                pred_embeddings_list.append(item)
        pred_embeddings_list_video, success = self.genetate_video_pred_embeddings(
            pred_embeddings_list, frames_per_batch)
        if not success:
            raise NotImplementedError

        if self.use_fast_supervision and fast_exists is not None:
            # gt_exists = []
            # for id_x, _fast_exists in enumerate(fast_exists):
            #     num_tot = _fast_exists.shape[0]
            #     num_conv = gt_masks[id_x].shape[0] // frames_per_batch[id_x]
            #     assert num_tot % num_conv == 0
            #     gt_exists.append(_fast_exists.reshape(num_conv, num_tot // num_conv))
            fast_flag = input_ids == self.fast_token_idx
            fast_tokens = output.hidden_states[-1][fast_flag]
            exists_logit = self.text_exist_fcs(fast_tokens[self.fast_pool_size ** 2 - 1::self.fast_pool_size ** 2])
            gt_exists = torch.cat(fast_exists)
            loss_exists = self.loss_exists(exists_logit, gt_exists)
        else:
            loss_exists = None

        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
            pred_embeddings_list_video, gt_masks_video
        )
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])
        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video)
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings,
                                                                 nf_nobj=(num_frames, num_objs))

        gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for
                    gt_mask in gt_masks_video]
        gt_masks = torch.cat(gt_masks, dim=0)
        pred_masks = pred_masks.flatten(0, 1)

        loss_mask, loss_dice = 0, 0
        if len(pred_masks) != len(gt_masks):
            # drop this data
            print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
            min_num = min(len(pred_masks), len(gt_masks))
            pred_masks = pred_masks[:min_num]
            gt_masks = gt_masks[:min_num]
            seg_valid = False

        if self.loss_sample_points:
            sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(
                sampled_pred_mask,
                sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
            sam_loss_mask = self.loss_mask(
                sampled_pred_mask.reshape(-1),
                sampled_gt_mask.reshape(-1),
                avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
        else:
            sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(pred_masks, gt_masks)
        loss_mask += sam_loss_mask
        loss_dice += sam_loss_dice

        if not seg_valid:
            _scale = 0.0
        else:
            _scale = 1.0
        loss_mask = loss_mask * _scale
        loss_dice = loss_dice * _scale

        loss_dict = {
            'loss_mask': loss_mask,
            'loss_dice': loss_dice,
            'llm_loss': output.loss,
        }
        if loss_exists is not None:
            loss_dict['loss_exists'] = loss_exists
        return loss_dict

    def predict(self, data):
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        eos_token_id = self.tokenizer.convert_tokens_to_ids('<|end|>')
        generation_config['eos_token_id'] = eos_token_id
        pixel_values = data.pop('pixel_values')
        attention_mask = data.pop('attention_mask', None)
        input_ids = data['input_ids']
        generate_output = self.mllm.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **generation_config,
        )
        device = self.mllm.model.device

        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1] for item in hidden_states[1:]]  # remove input_ids
        last_hidden_states = torch.cat(last_hidden_states, dim=1)
        last_hidden_states = last_hidden_states[0]  # remove batch dim
        output_ids = generate_output.sequences[0][:-1]  # remove batch dim and eos token
        output_text = self.tokenizer.decode(output_ids)
        seg_mask = output_ids == self.seg_token_idx
        if seg_mask.sum() == 0:
            return dict(
                pred_mask_logits=None,
                output_text=output_text,
            )
        seg_embeds = self.text_hidden_fcs(last_hidden_states[seg_mask])

        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data['masks']

        ori_size_list = [mask.shape[-2:] for mask in gt_masks]
        resize_list = [pixel.shape[-2:] for pixel in g_pixel_values]
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess(pixel.to(device)) for pixel in g_pixel_values
        ])
        image_embeddings = self.grounding_encoder.image_encoder(g_pixel_values)
        pred_masks = self._generate_and_postprocess_masks(
            [seg_embeds], image_embeddings, resize_list, ori_size_list)

        return dict(
            pred_mask_logits=pred_masks[0],  # remove batch dim
            output_text=output_text,
        )

    def _generate_and_postprocess_masks(self, pred_embeddings, image_embeddings, resize_list=None, orig_size_list=None):
        pred_masks = []
        for i, pred_embedding in enumerate(pred_embeddings):
            sparse_embeddings, dense_embeddings = self.grounding_encoder.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, _ = self.grounding_encoder.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.grounding_encoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                multimask_output=False)

            pred_mask = self.grounding_encoder.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=orig_size_list[i], )
            pred_masks.append(pred_mask[:, 0])
        return pred_masks

    def _merge_lora(self):
        # print('pre merge lora: ', self.mllm.model.language_model.base_model.model.get_input_embeddings().weight.shape)
        try:
            self.mllm.model.language_model = self.mllm.model.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        try:
            self.mllm.model.vision_model = self.mllm.model.vision_model.merge_and_unload()
        except:
            print("Skip vision encoder, no LoRA in it !!!")
        # print('after merge lora: ', self.mllm.model.language_model.get_input_embeddings().weight.shape)
        return

    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        mask_pred = mask_pred.unsqueeze(1)
        # (N, 1, h, w)

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred.to(torch.float32), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
        return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)

    def genetate_video_pred_embeddings(self, pred_embeddings_list, frames_per_batch):
        if len(pred_embeddings_list) == len(frames_per_batch):
            success = True
        else:
            success = False
            print("len(pred_embeddings_list):{} is not equal to len(frames_per_batch):{} !!!".format(
                len(pred_embeddings_list), len(frames_per_batch)))
        pred_embeddings_list_video = []
        for pred_embedding_batch, frame_nums in zip(pred_embeddings_list, frames_per_batch):
            pred_embeddings_list_video += [pred_embedding_batch] * frame_nums
        return pred_embeddings_list_video, success

    def process_video_gt_masks(self, gt_masks, frames_per_batch):
        gt_masks_video = []

        assert len(gt_masks) == len(frames_per_batch)
        for gt_masks_batch, frames_num in zip(gt_masks, frames_per_batch):
            N, H, W = gt_masks_batch.shape
            assert N % frames_num == 0
            gt_masks_batch = gt_masks_batch.reshape(
                N // frames_num, frames_num, H, W)
            for i in range(frames_num):
                gt_masks_video.append(gt_masks_batch[:, i])
        return gt_masks_video

    def llm_process_video(self,
                          metas,
                          raw_inputs,
                          prompt_base,
                          whole_video=False,
                          **kwargs):
        # load and init the pretrained model
        if not self.init_prediction_config:
            self.preparing_for_generation()

        past_text = kwargs.get('past_text', "")
        num_image_tokens = self.patch_token

        predictions = []
        for b_i in range(self.bs):
            pixel_values = self.get_frames_from_video(raw_inputs[b_i], sample_type='uniform', resize=(448,448),
                                                      norm=True).to(self.torch_dtype)

            vp_token_str = ''
            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            image_token_str = image_token_str + '\n'
            image_token_str = image_token_str * pixel_values.shape[1]
            image_token_str = image_token_str.strip()
            text = "<image>" + prompt_base
            if '<image>' in text:
                assert past_text is None or len(past_text) == 0
            text = text.replace('<image>', image_token_str + vp_token_str)
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).cuda().unsqueeze(0)
            attention_mask = torch.ones_like(ids, dtype=torch.bool)
            mm_inputs = {
                'pixel_values': pixel_values[b_i, :, :, :, :],
                'input_ids': ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'prompt_masks': None,
                'vp_overall_mask': None,
                'fast_pixel_values': None,
                'fast_token_idx': None,
            }
            torch.cuda.empty_cache()
            generate_output = self.mllm.generate(
                **mm_inputs,
                generation_config=self.gen_config,
                streamer=None,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.stop_criteria,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            predict = self.tokenizer.decode(generate_output.sequences[0], skip_special_tokens=False).strip()
            predictions.append(predict)

    def predict_video(self,
                      metas,
                      raw_inputs,
                      prompt_base,
                      whole_video=False,
                      **kwargs):
        def get_ori_bi(ori_metas, video_name):
            for i, ori_meta in enumerate(ori_metas):
                if ori_meta['video_name'] in video_name:
                    return i
            return -1

        def batch_wrapper(metas, ret_dict, ori_metas):
            features = ret_dict['features']
            # merge the split batch into original batch
            new_features = [{} for i in range(len(ori_metas))]
            for b_i, sample_feats in enumerate(features):
                video_name = metas[b_i]['video_name']
                ori_bi = get_ori_bi(ori_metas, video_name)
                for feat_name, feat in sample_feats.items():
                    if feat_name not in new_features[ori_bi]:
                        new_features[ori_bi][feat_name] = feat if type(feat) == list else feat
                    elif type(feat) == list:
                        new_features[ori_bi][feat_name].extend(feat)
                    elif type(feat) == torch.Tensor:
                        new_features[ori_bi][feat_name] = 0.5 * new_features[ori_bi][feat_name] + 0.5 * feat
            # collect the features with same name
            features = {}
            for b_i, sample_feats in enumerate(new_features):
                for feat_name, feat in sample_feats.items():
                    if feat_name not in features:
                        features[feat_name] = [feat]
                    else:
                        features[feat_name].append(feat)
            # stack the features
            for feat_name, feat in features.items():
                if type(feat[0]) == list and len(feat[0])>0 and type(feat[0][0]) == torch.Tensor:
                    features[feat_name] = [torch.stack(feat[b_i], dim=0) for b_i in range(len(feat))]
            ret_dict['features'] = features
            return ret_dict

        rle = False
        self.bs = len(raw_inputs)
        ori_metas = copy.deepcopy(metas)

        # To avoid OOM, we need to split the video into several parts
        new_raw_inputs = []
        new_metas = []
        split_length = 2500
        for b_i, raw in enumerate(raw_inputs):
            num_frames = min(self.num_frames, raw.shape[2])
            if raw.shape[2] > split_length:
                add_bs = raw.shape[2] // split_length + 1
                self.bs = (self.bs - 1) + add_bs
                length = raw.shape[2] // add_bs
                raw = raw[:, :, :length * add_bs, :, :]
                raw = list(torch.split(raw, length, dim=2))
                # for i in range(add_bs):
                #     if i == 0:
                #         feat = raw[i]  # B,3,T,W,H
                #         last_feat = feat[:, :, -num_frames:, :, :]
                #         raw[i] = torch.cat([feat, last_feat], dim=2)
                #     else:
                #         feat = raw[i]
                #         raw[i] = torch.cat([last_feat, feat], dim=2)
                #         last_feat = feat[:, :, -num_frames:, :, :]
                new_raw_inputs.extend(raw)
                metas_temp = [metas[b_i].copy() for _ in range(add_bs)]
                video_name = metas[b_i]['video_name']
                for i in range(add_bs):
                    metas_temp[i]['video_name'] = video_name + '_' + str(i)
                    metas_temp[i]['trunc_start'] = i * length
                    new_metas.append(metas_temp[i])
            else:
                new_metas.append(metas[b_i])
                new_raw_inputs.append(raw)
        raw_inputs = new_raw_inputs
        del new_raw_inputs, raw
        metas = new_metas
        past_text = kwargs.get('past_text', "")

        # check if the prediction already cached
        existed = True
        ret_dict = {'features': []}
        for b_i in range(self.bs):
            file_name = metas[b_i]['video_name'] + '_' + str(metas[b_i]['trunc_start']) if not whole_video else \
                metas[b_i]['video_name']
            file_path = os.path.join(self.sam2_prediction_path, f"{file_name}.pth")
            if not os.path.exists(file_path):
                existed = False
                print(f"Warning, {file_path} not in cache !!!")
                break
            loaded_features = torch.load(file_path)
            ret_dict['features'].append(loaded_features)
        if len(past_text) == 0 and existed:
            return batch_wrapper(metas, ret_dict, ori_metas)

        # check if the video length is same
        T = raw_inputs[0].shape[2]
        assert all(raw.shape[2] == T for raw in raw_inputs), "The video lengths within a batch must be same!!!"
        raw_inputs = torch.cat(raw_inputs, dim=0)
        raw_inputs = raw_inputs.to(device='cuda', dtype=self.torch_dtype)

        # load and init the pretrained model
        if not self.init_prediction_config:
            self.preparing_for_generation()

        # (n_f, 3, h, w)
        pixel_values = self.get_frames_from_video(raw_inputs[:, :, :num_frames, :, :], sample_type='begin',
                                                  resize=(self.image_size, self.image_size), norm=True).to(
            self.torch_dtype)
        num_image_tokens = self.patch_token
        ori_image_size = raw_inputs.shape[-2:]

        vp_token_str = ''
        image_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{self.IMG_END_TOKEN}'
        image_token_str = image_token_str + '\n'
        image_token_str = image_token_str * num_frames
        image_token_str = image_token_str.strip()

        text = "<image>" + prompt_base
        if '<image>' in text:
            assert past_text is None or len(past_text) == 0
        text = text.replace('<image>', image_token_str + vp_token_str)
        input_text = ''
        input_text += self.template['INSTRUCTION'].format(
            input=text, round=1, bot_name=self.bot_name)
        input_text = past_text + input_text
        ids = self.tokenizer.encode(input_text)
        ids = torch.tensor(ids).cuda().unsqueeze(0)
        attention_mask = torch.ones_like(ids, dtype=torch.bool)

        raw_inputs = raw_inputs.cpu()

        predictions = []
        pred_masks = []
        is_exists_list = []
        seg_feat_list = []
        sam2_maskmem_feat_list = []
        sam2_vision_feat_list = []
        for b_i in range(self.bs):
            mm_inputs = {
                'pixel_values': pixel_values[b_i, :, :, :, :],
                'input_ids': ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'prompt_masks': None,
                'vp_overall_mask': None,
                'fast_pixel_values': None,
                'fast_token_idx': None,
            }
            torch.cuda.empty_cache()
            generate_output = self.mllm.generate(
                **mm_inputs,
                generation_config=self.gen_config,
                streamer=None,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.stop_criteria,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            predict = self.tokenizer.decode(generate_output.sequences[0], skip_special_tokens=False).strip()
            predictions.append(predict)

            hidden_states = generate_output.hidden_states
            last_hidden_states = [item[-1][0] for item in hidden_states]
            last_hidden_states = torch.cat(last_hidden_states, dim=0)
            seg_hidden_states = get_seg_hidden_states(
                last_hidden_states, generate_output.sequences[0][:-1],
                seg_id=self.seg_token_idx
            )  # (1,2048)
            del hidden_states
            if not self.use_fast_supervision or (ids == self.fast_token_idx).sum() <= 0:
                del last_hidden_states, generate_output
            torch.cuda.empty_cache()
            g_pixel_val = self.get_frames_from_video(raw_inputs[b_i:b_i + 1, :, :, :, :], sample_type='begin',
                                                     resize=(1024, 1024), norm=True)
            g_pixel_val = g_pixel_val.cuda()
            g_pixel_val = g_pixel_val.squeeze(0)
            if len(seg_hidden_states) == 0:
                print("Warning, no [SEG] tokens !!!")
                pred_masks.append(
                    torch.zeros((g_pixel_val.shape[0], ori_image_size[0], ori_image_size[1]), dtype=torch.int))
                continue
            elif len(seg_hidden_states) > 1:
                print("Warning, {} [SEG] tokens !!!".format(len(seg_hidden_states)))
                seg_hidden_states = seg_hidden_states[:1]
            seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

            seg_hidden_states = seg_hidden_states.to(dtype=torch.float32)

            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_val)
            pred_mask = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * num_frames)
            pred_mask = F.interpolate(
                pred_mask,
                size=ori_image_size,
                mode='bilinear',
                align_corners=False,
            )
            pred_mask = pred_mask[:, 0]
            pred_mask = pred_mask.sigmoid() > 0.5
            pred_mask = pred_mask.int()
            # supervision
            if self.use_fast_supervision and (ids == self.fast_token_idx).sum() > 0:
                fast_flag = ids.squeeze(0) == self.fast_token_idx
                len_out = generate_output.sequences[0][:-1].shape[0]
                fast_tokens = last_hidden_states[:-len_out][fast_flag].to(dtype=torch.float32)
                exists_logit = self.text_exist_fcs(fast_tokens[self.fast_pool_size ** 2 - 1::self.fast_pool_size ** 2])
                is_exists = exists_logit.squeeze(-1).sigmoid() > 0.5
                is_exists_list.append(is_exists)
                not_exists = torch.logical_not(is_exists)
                if torch.any(not_exists):
                    pred_mask[not_exists] = pred_mask[not_exists] * 0

            pred_masks.append(pred_mask)

            seg_feat_list.append(seg_hidden_states.cpu())
            sam2_vision_features = sam_states['cached_vision_features']
            sam2_vision_feat_list.append([])
            sam2_maskmem_feat_list.append([])
            for t_i, feats in sam2_vision_features.items():
                if b_i == 0 and t_i >= len(sam2_vision_features) - num_frames:
                    break
                # Step 1: reshape to (B, 64, 64, 256)
                B = feats.shape[1]
                feats = feats.permute(1, 0, 2)  # -> (B, 4096, 256)
                feats = feats.view(B, 64, 64, -1)  # -> (B, 64, 64, 256)

                # Step 2: group into (B, 16, 16, 4, 4, 256)
                G = 16
                H, W = 64 // G, 64 // G
                feats = feats.view(B, H, G, W, G, -1)  # (B, 4, 16, 4, 16, 256)
                feats = feats.permute(1, 3, 0, 2, 4, 5)  # (4, 4, B, 16, 16, 256)
                feats = feats.contiguous().view(H * W, B, G * G, -1)  # (16, B, 256, 256)

                # Step 3: mean pooling over group (4x4 = 16 patches)
                feats = feats.mean(dim=2)  # -> (16, B, 256)
                feats = feats.squeeze(1).flatten()
                sam2_vision_feat_list[b_i].append(feats.cpu())
                if t_i < num_frames:
                    mem_feat = [
                        sam_states['output_dict_per_obj'][obj_i]['cond_frame_outputs'][t_i]['maskmem_features']
                        for obj_i in range(len(sam_states['output_dict_per_obj']))
                    ]
                else:
                    mem_feat = [
                        sam_states['output_dict_per_obj'][obj_i]['non_cond_frame_outputs'][t_i]['maskmem_features']
                        for obj_i in range(len(sam_states['output_dict_per_obj']))
                    ]

                mem_feat_mean = torch.cat(mem_feat, dim=0).mean(dim=0)
                sam2_maskmem_feat_list[b_i].append(mem_feat_mean.cpu())
            del sam_states

        ret_dict = {
            'prediction': predictions,
            'prediction_masks': [mask_to_rle(_item.cpu().numpy()) for _item in pred_masks] if rle else pred_masks,
        }

        if len(is_exists_list) > 0:
            ret_dict['is_exists'] = is_exists_list
        ret_dict['features'] = []

        del g_pixel_val, pixel_values

        # todo: save the video
        ori_pixel_values = self.get_frames_from_video(raw_inputs[:, :, :, :, :], sample_type='begin')
        for b_i in range(self.bs):
            if len(ret_dict['prediction_masks']) >= self.bs:
                file_name = metas[b_i]['video_name'] + '_' + str(metas[b_i]['trunc_start']) if not whole_video else \
                    metas[b_i]['video_name']
                ori_pixel_val = [f.permute(1, 2, 0) for f in ori_pixel_values[b_i]]
                video_mask_show, selected_colors = show_mask_pred_video(ori_pixel_val,
                                                                        [ret_dict['prediction_masks'][b_i]], )
                predict_dict = {
                    'sam2_vision_features': sam2_vision_feat_list[b_i],
                    'sam2_maskmem_features': sam2_maskmem_feat_list[b_i],
                    'seg_features': seg_feat_list[b_i],
                    'prediction': predictions[b_i],
                } if len(sam2_vision_feat_list) > 0 else None
                ret_dict['features'].append(predict_dict)
                video_save_path = os.path.join(self.sam2_prediction_path, f"{file_name}.mp4")
                image2video_and_save(video_mask_show, save_path=video_save_path)
                pth_save_path = os.path.join(self.sam2_prediction_path, f"{file_name}.pth")
                torch.save(predict_dict, pth_save_path) if predict_dict else None
            else:
                selected_colors = []

            ret_dict['prediction'] = [x.strip() for x in ret_dict['prediction']]
        return batch_wrapper(metas, ret_dict, ori_metas) if predict_dict is not None else ret_dict

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
            raise NotImplementedError
        elif not self.mllm.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
            raise NotImplementedError
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
            raise NotImplementedError
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
        # frames: [B, C, T, H, W]
        self.transformer_mean = self.transformer_mean.to(frames.device)
        self.transformer_std = self.transformer_std.to(frames.device)

        def resize_frame(frame, resize):
            if resize is not None:
                frame = F.interpolate(frame, size=resize, mode='bicubic', align_corners=False)  # Resize
            if norm:
                ndim = frame.dim()
                if ndim == 3:  # [C, H, W]
                    return frame - self.transformer_mean.squeeze(0) / self.transformer_std.squeeze(0)
                elif ndim == 4:  # [T, C, H, W]
                    return frame - self.transformer_mean / self.transformer_std
                elif ndim == 5:  # [B, T, C, H, W]
                    return frame - self.transformer_mean.unsqueeze(0) / self.transformer_std.unsqueeze(0)
            return frame

        B, C, T, H, W = frames.shape
        if sample_type == "uniform":
            stride = T / (n_frames + 1e-4)
            ret = []
            for i in range(n_frames):
                idx = int(i * stride)
                frame = frames[:, :, idx, :, :]  # [B, C, H, W]
                # frame = frame.flip(1)  # to RGB
                ret.append(resize_frame(frame, resize))
            frames = torch.stack(ret, dim=0)  # [n_frames,B, C, H, W]
            frames=frames.permute(1, 0, 2, 3,4)  # [B, n_frames, C, H, W]
        else:
            frames = frames.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
            if resize is not None:
                frames = frames.view(B * T, C, H, W)  # [B*T, C, H, W]
            frames = resize_frame(frames, resize)
            if resize is not None:
                frames = frames.view(B, T, C, resize[0], resize[1])
        return frames

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

    def check_obj_number(self, pred_embeddings_list_video, gt_masks_video, fix_number=5):
        assert len(pred_embeddings_list_video) == len(gt_masks_video)
        ret_pred_embeddings_list_video = []
        ret_gt_masks_video = []
        for pred_mebeds, gt_masks in zip(pred_embeddings_list_video, gt_masks_video):
            # assert len(pred_mebeds) == len(gt_masks)
            if len(pred_mebeds) != len(gt_masks):
                min_num = min(len(pred_mebeds), len(gt_masks))
                pred_mebeds = pred_mebeds[:min_num]
                gt_masks = gt_masks[:min_num]
            if len(pred_mebeds) != fix_number:
                if len(pred_mebeds) > fix_number:
                    _idxs = torch.randperm(pred_mebeds.shape[0])
                    _idxs = _idxs[:fix_number]
                    pred_mebeds = pred_mebeds[_idxs]
                    gt_masks = gt_masks[_idxs]
                else:
                    n_repeat = fix_number // len(pred_mebeds) + 1
                    pred_mebeds = torch.cat([pred_mebeds] * n_repeat, dim=0)[:fix_number]
                    gt_masks = torch.cat([gt_masks] * n_repeat, dim=0)[:fix_number]
            ret_pred_embeddings_list_video.append(pred_mebeds)
            ret_gt_masks_video.append(gt_masks)
        return ret_pred_embeddings_list_video, ret_gt_masks_video

    def _get_pesudo_data(self, dtype, device):
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((5, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks

    def process_rpn_head(self, rpn_head):
        additional_channels = {
            'sam2_vision_features': self.aggregation_dim,
            'seg_features': 256,
            'sam2_maskmem_features': self.aggregation_dim,
            'x': 0
        }
        in_channels = rpn_head['projection']['in_channels'] if 'x' in self.aggregated_features else 0
        rpn_head['projection']['in_channels'] = (in_channels +
                                                 sum([additional_channels[k] for k in self.aggregated_features]))
        print(f"RPN head in_channels is modified to: {rpn_head['projection']['in_channels']}", flush=True)
        return rpn_head


def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]


def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle


def prepare_inputs_for_generation_phi3(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and (past_key_values is None or len(past_key_values) == 0):
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids}

    model_inputs.update(
        {
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        }
    )
    return model_inputs


def show_mask_pred_video(video, masks):
    ret_video = []
    selected_colors = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 128, 255), [255, 192, 203],  # Pink
              [165, 42, 42],  # Brown
              [255, 165, 0],  # Orange
              [128, 0, 128],  # Purple
              [0, 0, 128],  # Navy
              [128, 0, 0],  # Maroon
              [128, 128, 0],  # Olive
              [70, 130, 180],  # Steel Blue
              [173, 216, 230],  # Light Blue
              [255, 192, 0],  # Gold
              [255, 165, 165],  # Light Salmon
              [255, 20, 147],  # Deep Pink
              ]
    if isinstance(video, torch.Tensor):
        video = video.to(torch.float32).cpu().numpy()
    elif isinstance(video, list) and isinstance(video[0], torch.Tensor):
        video = [frame.to(torch.float32).cpu().numpy() for frame in video]
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    elif isinstance(masks, list) and isinstance(masks[0], torch.Tensor):
        masks = [mask.cpu().numpy() for mask in masks]
    for i_frame in range(len(video)):
        frame_masks = [mask[i_frame:i_frame + 1] for mask in masks]
        frame_masks = np.concatenate(frame_masks, axis=0)
        _mask_image = np.zeros((frame_masks.shape[1], frame_masks.shape[2], 3), dtype=np.uint8)

        for i, mask in enumerate(frame_masks):
            if i_frame == 0:
                color = colors[i % len(colors)]
                selected_colors.append(color)
            else:
                color = selected_colors[i]
            _mask_image[:, :, 0] = _mask_image[:, :, 0] + mask.astype(np.uint8) * color[0]
            _mask_image[:, :, 1] = _mask_image[:, :, 1] + mask.astype(np.uint8) * color[1]
            _mask_image[:, :, 2] = _mask_image[:, :, 2] + mask.astype(np.uint8) * color[2]

        image = np.array(video[i_frame])
        image = image * 0.5 + _mask_image * 0.5
        image = image.astype(np.uint8)
        ret_video.append(image)
    return ret_video, selected_colors


def image2video_and_save(frames, save_path):
    success = frames_to_video(frames, save_path)
    return save_path


def frames_to_video(
        frames,
        output_path: str,
        fps: int = 15,
) -> bool:
    try:
        frames = [frame[:, :, ::-1] for frame in frames]
        # Use provided frame size or get from first frame
        height, width = frames[0].shape[:2]

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process each frame
        for frame in frames:
            out.write(frame)

        # Release video writer
        out.release()
        print(f"Video saved successfully to {output_path}")
        return True

    except Exception as e:
        print(f"Error converting frames to video: {str(e)}")
        return False

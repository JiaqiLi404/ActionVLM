from collections import OrderedDict
from types import MethodType
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from mmaction.models import ActionDataPreprocessor
from mmengine.model import BaseModel
from peft import LoraConfig, get_peft_model_state_dict
from torch import nn
from transformers import AutoConfig, AutoModel, GenerationConfig
from transformers.cache_utils import Cache

from opentad.models import VisionTransformerCP
from opentad.models.backbones.vit import VisionTransformerCP_Base, VisionTransformerCP_Large, VisionTransformerCP_Small
from opentad.models.builder import LANGUAGE_MODELS, build_language_model
from opentad.models.language_models.utils import (
    enable_gradient_checkpointing,
    get_stop_criteria,
    guess_load_checkpoint,
)
from opentad.models.language_models.utils.templates import PROMPT_TEMPLATE


@LANGUAGE_MODELS.register_module()
class ActionVLM(BaseModel):
    def __init__(
        self,
        mllm,
        tune_llm=False,
        tune_visual_encoder=False,
        lora_llm_enable=True,
        lora_visual_enable=False,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        lora_bias: Literal["none", "all", "lora_only"] = "none",
        torch_dtype="bf16",
        pretrained_pth=None,
        hf_model_path=None,
        fast_pool=False,
        fast_pool_size=4,
        phi3=True,
        template=None,
        arch_type: Literal["intern_vl", "qwen", "qwen3", "llava"] = "intern_vl",
        clip_window=128,
        backbone_type: Literal[
            "videomaes",
            "videomaeb",
            "videomael",
            "vision",
            "videomaes_language",
            "videomaeb_language",
            "videomael_language",
            "vision_language",
            "videomaes_language_token",
            "videomaeb_language_token",
            "videomael_language_token",
        ] = "vision_language",
        loss_weight=1.0,
        tune_video_backbone=None,
        enable_debiasing=True,
        adv_loss_weight=0.1,
        vision_only_interval=2,
        language_warmup_epochs=0,
        fusion_mode: Literal[
            "feature",
            "cls_logit_bias",
            "language_conditioned_cls_head",
            "dual_visual_cls_head",
            "cross_modal_cls_head",
            "expert_router_cls_head",
            "candidate_reranker_cls_head",
            "distill_cls_head",
        ] = "feature",
        use_delta_fusion=True,
        lang_residual_dropout=0.1,
        fallback_lang_weight=0.2,
        lang_gate_floor=0.0,
        cls_lang_scale=1.0,
        loc_lang_scale=0.6,
        use_similarity_gate=True,
        similarity_gate_scale=8.0,
        semantic_prior_scale=0.5,
        semantic_loss_weight=0.5,
        semantic_temperature=0.07,
        qwen_visual_num_frames=8,
        teacher_loss_weight=0.2,
        cls_token_feature_mode: Literal["raw", "contextual_residual", "query_bank"] = "raw",
        lang_weight_mode: Literal["confidence", "visual", "hybrid"] = "confidence",
        rerank_topk=0,
        cls_query_bank_size=4,
        prototype_align_loss_weight=0.0,
        **kwargs,
    ):
        super().__init__()

        self.downsample_ratio = 0.5
        self.image_size = 448
        self.patch_size = 14
        mllm["downsample_ratio"] = self.downsample_ratio
        mllm["image_size"] = self.image_size

        self.model = self.mllm = build_language_model(mllm)
        self.hf_model_path = hf_model_path
        self.tune_llm = tune_llm
        self.tune_visual_encoder = tune_visual_encoder
        self.tune_video_backbone = tune_visual_encoder if tune_video_backbone is None else tune_video_backbone
        self.lora_llm_enable = lora_llm_enable
        self.lora_visual_enable = lora_visual_enable
        self.lora_enable = self.lora_llm_enable or self.lora_visual_enable
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.fast_pool = fast_pool
        self.fast_pool_size = fast_pool_size
        self.arch_type = arch_type
        self.backbone_type = backbone_type
        self.loss_weight = loss_weight
        self.enable_debiasing = enable_debiasing
        self.adv_loss_weight = adv_loss_weight
        self.vision_only_interval = max(int(vision_only_interval), 0)
        self.language_warmup_epochs = max(int(language_warmup_epochs), 0)
        self.fusion_mode = fusion_mode
        self.use_delta_fusion = use_delta_fusion
        self.fallback_lang_weight = float(min(max(fallback_lang_weight, 0.0), 1.0))
        self.lang_gate_floor = float(min(max(lang_gate_floor, 0.0), 0.99))
        self.cls_lang_scale = float(max(cls_lang_scale, 0.0))
        self.loc_lang_scale = float(max(loc_lang_scale, 0.0))
        self.use_similarity_gate = use_similarity_gate
        self.similarity_gate_scale = float(max(similarity_gate_scale, 0.0))
        self.semantic_prior_scale = float(max(semantic_prior_scale, 0.0))
        self.semantic_loss_weight = float(max(semantic_loss_weight, 0.0))
        self.semantic_temperature = float(max(semantic_temperature, 1e-4))
        self.qwen_visual_num_frames = max(int(qwen_visual_num_frames), 0)
        self.teacher_loss_weight = float(max(teacher_loss_weight, 0.0))
        self.cls_token_feature_mode = cls_token_feature_mode
        self.lang_weight_mode = lang_weight_mode
        self.rerank_topk = max(int(rerank_topk), 0)
        self.cls_query_bank_size = max(int(cls_query_bank_size), 1)
        self.prototype_align_loss_weight = float(max(prototype_align_loss_weight, 0.0))
        self._last_logged_epoch_mode = None

        self.tokenizer = self.mllm.tokenizer
        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN = "</img>"
        if self.arch_type in {"qwen", "qwen3"}:
            self.IMG_CONTEXT_TOKEN = "<|image_pad|>"
            if self.tokenizer.convert_tokens_to_ids("<|vision_start|>") is not None:
                self.IMG_START_TOKEN = "<|vision_start|>"
                self.IMG_END_TOKEN = "<|vision_end|>"
            else:
                self.IMG_START_TOKEN = ""
                self.IMG_END_TOKEN = ""

        if torch_dtype == "bf16":
            self.torch_dtype = torch.bfloat16
        elif torch_dtype == "fp16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        self.patch_token = int((self.image_size // self.patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        self.img_cls_token_id = self.img_context_token_id
        self.seg_token_idx = self.tokenizer.convert_tokens_to_ids("[SEG]")
        self.fast_token_idx = (
            self.tokenizer("<FAST_IMG_CONTEXT>", add_special_tokens=False).input_ids[0] if fast_pool else None
        )

        self.llm_lora_config = LoraConfig(
            r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout, bias=self.lora_bias
        )
        self.vision_lora_config = LoraConfig(
            r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout, bias=self.lora_bias
        )

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)

        self.phi3 = phi3
        self.template = template
        self.init_prediction_config = False
        self.bot_name = "BOT"

        self.hf_config = None
        if self.hf_model_path is not None and ("vision" in backbone_type or "language" in backbone_type):
            self.hf_model = AutoModel.from_pretrained(
                self.hf_model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval()
            self.hf_config = AutoConfig.from_pretrained(self.hf_model_path, trust_remote_code=True)
            self.hf_state_dict = self.hf_model.state_dict()
            self.hf_mllm_state_dict = OrderedDict()
            self.hf_sam_state_dict = OrderedDict()
            self.hf_text_fcs_state_dict = OrderedDict()
            for k, v in self.hf_state_dict.items():
                if "sam2" in k:
                    self.hf_sam_state_dict[k] = v
                elif "text_hidden_fcs" in k:
                    self.hf_text_fcs_state_dict[k] = v
                else:
                    self.hf_mllm_state_dict[k] = v
            load_result = self.mllm.model.load_state_dict(self.hf_mllm_state_dict, strict=False)
            print("Missing keys (model expects but not in hf_model):")
            print(set(key.split(".")[0] for key in load_result.missing_keys))
            print("Unexpected keys (in hf_model but not in model):")
            print(load_result.unexpected_keys)

        if not self.tune_llm:
            self.mllm.call_freeze_llm()
        if not self.tune_visual_encoder:
            self.mllm.call_freeze_visual_encoder()
        if self.lora_llm_enable:
            self.mllm.call_llm_lora(self.llm_lora_config)
        if self.lora_visual_enable:
            self.mllm.call_visual_encoder_lora(self.vision_lora_config)

        if hasattr(self.mllm, "_post_init"):
            self.mllm._post_init(fast_pool_size=self.fast_pool_size, fast_pool=self.fast_pool)

        if self.lora_llm_enable:
            if self.arch_type == "intern_vl":
                self.mllm.model.language_model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.language_model.base_model.model.get_output_embeddings().requires_grad_(True)
            elif self.arch_type in {"qwen", "qwen3"}:
                llm_module = (
                    self.mllm.get_llm_module()
                    if hasattr(self.mllm, "get_llm_module")
                    else self.mllm.model.language_model
                )
                llm_module.get_input_embeddings().requires_grad_(True)
                output_embeddings = self.mllm.model.get_output_embeddings()
                if output_embeddings is not None and hasattr(output_embeddings, "weight"):
                    output_embeddings.weight.requires_grad_(True)
            elif self.arch_type == "llava":
                self.mllm.model.language_model.base_model.model.get_input_embeddings().requires_grad_(True)
                self.mllm.model.language_model.base_model.model.get_output_embeddings().requires_grad_(True)

        if self.hf_model_path is not None and ("vision" in backbone_type or "language" in backbone_type):
            del self.hf_model
            del self.hf_state_dict
            del self.hf_mllm_state_dict
            del self.hf_text_fcs_state_dict
            del self.hf_sam_state_dict
            torch.cuda.empty_cache()

        self.data_preprocessor = ActionDataPreprocessor(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        )
        self.clip_window = clip_window

        self.vit = None
        if "videomaeb" in backbone_type:
            self.vit = VisionTransformerCP_Base()
            state_dict = torch.load("pretrained/vit_b_k710_dl_from_giant.pth")
            self.mllm_embedding_dim = 768
        elif "videomaes" in backbone_type:
            self.vit = VisionTransformerCP_Small()
            state_dict = torch.load("pretrained/vit_s_k710_dl_from_giant.pth")
            self.mllm_embedding_dim = 384
        elif "videomael" in backbone_type:
            self.vit = VisionTransformerCP_Large()
            state_dict = torch.load(
                "pretrained/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth"
            )
            self.mllm_embedding_dim = 1024

        if self.vit is not None:
            state_dict = state_dict["module"] if "module" in state_dict else state_dict
            for key in list(state_dict.keys()):
                if key.startswith("backbone."):
                    state_dict[key.replace("backbone.", "")] = state_dict.pop(key)
                if "fc1" in key:
                    state_dict[key.replace("fc1", "layers.0.0")] = state_dict.pop(key)
                if "fc2" in key:
                    state_dict[key.replace("fc2", "layers.1")] = state_dict.pop(key)
                if "patch_embed.proj." in key:
                    state_dict[key.replace("patch_embed.proj.", "patch_embed.projection.")] = state_dict.pop(key)
            res = self.vit.load_state_dict(state_dict, strict=False)
            print("Missing keys for VideoMAE:", res.missing_keys)
            print("Unexpected keys for VideoMAE:", res.unexpected_keys)
            self.backbone_embedding_dim = self.vit.embed_dims
            if not self.tune_video_backbone:
                self.vit.requires_grad_(False)
            del state_dict
            if hasattr(self.mllm.model, "vision_model"):
                del self.mllm.model.vision_model
            torch.cuda.empty_cache()
        else:
            self.backbone_embedding_dim = 1024
            self.mllm_embedding_dim = 1024
            self.temporal_positional_embedding = nn.Parameter(torch.zeros(2048, self.backbone_embedding_dim))

        if "language" in self.backbone_type:
            if self.arch_type == "intern_vl":
                self.mllm_embedding_dim = self.mllm.model.config.llm_config.hidden_size
            elif self.arch_type in {"qwen", "qwen3"}:
                self.mllm_embedding_dim = getattr(
                    self.mllm.model.config, "hidden_size", self.mllm.model.config.text_config.hidden_size
                )
            elif self.arch_type == "llava":
                self.mllm_embedding_dim = self.mllm.model.language_model.config.hidden_size

            self.backbone_language_mlp = nn.Linear(self.backbone_embedding_dim, self.mllm_embedding_dim, bias=False)
            self.norm = nn.LayerNorm(self.mllm_embedding_dim, eps=1e-6, elementwise_affine=False)
            if self.fusion_mode == "feature":
                self.lang_cls_adapter = nn.Linear(self.mllm_embedding_dim, self.mllm_embedding_dim, bias=False)
                self.lang_loc_adapter = nn.Linear(self.mllm_embedding_dim, self.mllm_embedding_dim, bias=False)
                nn.init.eye_(self.lang_cls_adapter.weight)
                nn.init.eye_(self.lang_loc_adapter.weight)
                self.lang_cls_gate = nn.Linear(self.mllm_embedding_dim * 2, 1)
                self.lang_loc_gate = nn.Linear(self.mllm_embedding_dim * 2, 1)
                nn.init.zeros_(self.lang_cls_gate.weight)
                nn.init.zeros_(self.lang_loc_gate.weight)
                nn.init.constant_(self.lang_cls_gate.bias, -1.0)
                nn.init.constant_(self.lang_loc_gate.bias, -2.0)
                self.lang_cls_norm = nn.LayerNorm(self.mllm_embedding_dim, eps=1e-6, elementwise_affine=False)
                self.lang_loc_norm = nn.LayerNorm(self.mllm_embedding_dim, eps=1e-6, elementwise_affine=False)
                self.lang_residual_dropout = nn.Dropout(lang_residual_dropout)
            else:
                self.lang_cls_adapter = None
                self.lang_loc_adapter = None
                self.lang_cls_gate = None
                self.lang_loc_gate = None
                self.lang_cls_norm = None
                self.lang_loc_norm = None
                self.lang_residual_dropout = nn.Identity()
            self.semantic_query_proj = nn.Linear(self.mllm_embedding_dim, self.mllm_embedding_dim, bias=False)
            self.semantic_context_proj = nn.Linear(self.mllm_embedding_dim, self.mllm_embedding_dim, bias=False)
            nn.init.eye_(self.semantic_query_proj.weight)
            nn.init.eye_(self.semantic_context_proj.weight)
            self.semantic_context_norm = nn.LayerNorm(self.mllm_embedding_dim, eps=1e-6, elementwise_affine=False)
            cls_query_heads = min(8, self.mllm_embedding_dim)
            while self.mllm_embedding_dim % cls_query_heads != 0 and cls_query_heads > 1:
                cls_query_heads -= 1
            self.cls_context_gate = None
            self.cls_context_norm = None
            self.cls_query_bank = None
            self.cls_query_attn = None
            self.cls_query_proj = None
            self.cls_query_gate = None
            self.cls_query_norm = None
            if self.cls_token_feature_mode == "contextual_residual":
                self.cls_context_gate = nn.Linear(self.mllm_embedding_dim * 2, 1)
                nn.init.zeros_(self.cls_context_gate.weight)
                nn.init.constant_(self.cls_context_gate.bias, -1.0)
                self.cls_context_norm = nn.LayerNorm(self.mllm_embedding_dim, eps=1e-6, elementwise_affine=False)
            elif self.cls_token_feature_mode == "query_bank":
                self.cls_query_bank = nn.Parameter(
                    torch.randn(self.cls_query_bank_size, self.mllm_embedding_dim) * (self.mllm_embedding_dim ** -0.5)
                )
                self.cls_query_attn = nn.MultiheadAttention(
                    self.mllm_embedding_dim, cls_query_heads, batch_first=True
                )
                self.cls_query_proj = nn.Linear(self.mllm_embedding_dim * 2, self.mllm_embedding_dim, bias=False)
                self.cls_query_gate = nn.Linear(self.mllm_embedding_dim * 2, 1)
                nn.init.zeros_(self.cls_query_gate.weight)
                nn.init.constant_(self.cls_query_gate.bias, -1.0)
                self.cls_query_norm = nn.LayerNorm(self.mllm_embedding_dim, eps=1e-6, elementwise_affine=False)
            else:
                self.cls_context_gate = None
                self.cls_context_norm = None
            self.adv_predictor = nn.Linear(self.mllm_embedding_dim, 1) if self.enable_debiasing else None
        else:
            self.adv_predictor = None
            self.lang_cls_adapter = None
            self.lang_loc_adapter = None
            self.lang_cls_gate = None
            self.lang_loc_gate = None
            self.lang_cls_norm = None
            self.lang_loc_norm = None
            self.lang_residual_dropout = nn.Identity()
            self.semantic_query_proj = None
            self.semantic_context_proj = None
            self.semantic_context_norm = None
            self.cls_context_gate = None
            self.cls_context_norm = None
            self.cls_query_bank = None
            self.cls_query_attn = None
            self.cls_query_proj = None
            self.cls_query_gate = None
            self.cls_query_norm = None
            if hasattr(self.mllm.model, "language_model"):
                del self.mllm.model.language_model

        if hasattr(self.mllm.model, "grounding_encoder"):
            del self.mllm.model.grounding_encoder

        self.cls_token_id = None
        self.loc_token_id = None
        self.adv_token_id = None
        if self.backbone_type.endswith("token"):
            special_tokens = ["<cls>", "<loc>"]
            if self.enable_debiasing:
                special_tokens.append("<adv>")
            if hasattr(self.mllm, "add_special_tokens"):
                self.mllm.add_special_tokens(special_tokens)
            else:
                num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)
                if num_new_tokens > 0:
                    if hasattr(self.mllm.model, "resize_token_embeddings"):
                        self.mllm.model.resize_token_embeddings(len(self.tokenizer))
                    else:
                        self.mllm.model.language_model.resize_token_embeddings(len(self.tokenizer))
            self.cls_token_id = self.tokenizer.convert_tokens_to_ids("<cls>")
            self.loc_token_id = self.tokenizer.convert_tokens_to_ids("<loc>")
            if self.enable_debiasing:
                self.adv_token_id = self.tokenizer.convert_tokens_to_ids("<adv>")

    def _uses_language_conditioned_classifier(self):
        return self.fusion_mode in {
            "language_conditioned_cls_head",
            "dual_visual_cls_head",
            "cross_modal_cls_head",
            "expert_router_cls_head",
            "candidate_reranker_cls_head",
        }

    def is_vision_only_epoch(self, curr_epoch):
        if "language" not in self.backbone_type:
            return False
        if curr_epoch is None or curr_epoch < 0:
            return False
        if curr_epoch < self.language_warmup_epochs:
            return True
        if self.vision_only_interval <= 0:
            return False
        return (curr_epoch - self.language_warmup_epochs) % self.vision_only_interval == 0

    def preparing_for_generation(self, max_new_tokens=2048):
        if self.hf_config is not None and getattr(self.hf_config, "template", None) is not None:
            self.template = self.hf_config.template.replace("-", "_")
        elif self.arch_type in {"qwen", "qwen3"}:
            self.template = "qwen_chat"
        else:
            self.template = "phi3_chat"
        self.template = self.conv_template = PROMPT_TEMPLATE[self.template]
        stop_words = self.template.get("STOP_WORDS", [])
        self.stop_criteria = get_stop_criteria(tokenizer=self.tokenizer, stop_words=stop_words)

        self.gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
            temperature=0.01,
        )
        self.init_prediction_config = True

        hf_archs = []
        if self.hf_config is not None:
            if hasattr(self.hf_config, "llm_config"):
                hf_archs = getattr(self.hf_config.llm_config, "architectures", [])
            else:
                hf_archs = getattr(self.hf_config, "architectures", [])
        if ((self.hf_config is None and self.phi3) or ("Phi3ForCausalLM" in hf_archs)) and hasattr(
            self.mllm, "prepare_inputs_for_generation"
        ):
            self.mllm.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_phi3, self.mllm)

    def forward_train(self, x, masks, gt_segments, gt_labels, metas, curr_epoch, prompt_base, **kwargs):
        answers = None
        vision_only_epoch = self.is_vision_only_epoch(curr_epoch)
        if curr_epoch != self._last_logged_epoch_mode:
            epoch_mode = "vision-only" if vision_only_epoch else "vision-language"
            print(f"[ActionVLM] Epoch {curr_epoch}: {epoch_mode} epoch", flush=True)
            self._last_logged_epoch_mode = curr_epoch
        if not vision_only_epoch:
            answers = self._build_description_answers(gt_segments, gt_labels, metas, prompt_base)

        x, llm_loss, _, aux = self.forward(
            x,
            metas,
            prompt_base,
            answers=answers,
            use_language=not vision_only_epoch,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            **kwargs,
        )

        pass_dict = {
            "x": x,
            "masks": masks if not self.backbone_type.endswith("token") else masks.repeat(2, 1),
            "gt_segments": gt_segments,
            "gt_labels": gt_labels,
            "metas": metas,
            "curr_epoch": curr_epoch,
            "prompt_base": prompt_base,
            "llm_loss": llm_loss,
            "semantic_loss": aux["semantic_loss"],
            "prototype_align_loss": aux["prototype_align_loss"],
            "branched": self.backbone_type.endswith("token"),
            "vision_only_epoch": vision_only_epoch,
            "collect_vision_stats": self.enable_debiasing,
            "adv_predictions": aux["adv_predictions"],
            "lambda_lang": aux["lambda_lang"],
            "cls_logit_bias": aux["cls_logit_bias"],
            "class_prototypes": aux["class_prototypes"],
            "cls_token_features": aux["cls_token_features"],
            "cls_token_confidence": aux["cls_token_confidence"],
            "qwen_visual_features": aux["qwen_visual_features"],
            "qwen_visual_confidence": aux["qwen_visual_confidence"],
            "teacher_cls_targets": aux["teacher_cls_targets"],
            "cls_lang_scale": self.cls_lang_scale,
            "adv_loss_weight": self.adv_loss_weight,
            "input_masks": masks,
        }
        pass_dict.update(kwargs)
        return pass_dict

    def forward_test(self, x, masks, metas, prompt_base, **kwargs):
        x, _, predictions, aux = self.forward(x, metas, prompt_base, answers=None, use_language=True, **kwargs)

        pass_dict = {
            "x": x,
            "masks": masks if not self.backbone_type.endswith("token") else masks.repeat(2, 1),
            "metas": metas,
            "prompt_base": prompt_base,
            "branched": self.backbone_type.endswith("token"),
            "predictions": predictions,
            "adv_predictions": aux["adv_predictions"],
            "lambda_lang": aux["lambda_lang"],
            "cls_logit_bias": aux["cls_logit_bias"],
            "class_prototypes": aux["class_prototypes"],
            "cls_token_features": aux["cls_token_features"],
            "cls_token_confidence": aux["cls_token_confidence"],
            "qwen_visual_features": aux["qwen_visual_features"],
            "qwen_visual_confidence": aux["qwen_visual_confidence"],
            "teacher_cls_targets": aux["teacher_cls_targets"],
            "cls_lang_scale": self.cls_lang_scale,
            "input_masks": masks,
        }
        pass_dict.update(kwargs)
        return pass_dict

    def _build_vision_only_features(self, x_in):
        if self.backbone_type.endswith("token"):
            x_in = self.norm(x_in)
            x = torch.cat([x_in, x_in], dim=0)
            return rearrange(x, "b t c -> b c t")
        x = self.norm(x_in)
        return rearrange(x, "b t c -> b c t")

    def _build_visual_token_str(self, time_steps):
        visual_token_str = f"{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN * time_steps}{self.IMG_END_TOKEN}".strip()
        return visual_token_str + "\n"

    def _build_query_token_str(self, time_steps):
        if not self.backbone_type.endswith("token"):
            return ""

        query_blocks = ["<cls>" * time_steps]
        if self.fusion_mode == "feature":
            query_blocks.append("<loc>" * time_steps)
        if self.enable_debiasing and self.adv_token_id is not None:
            query_blocks.append("<adv>" * time_steps)
        return "\n".join(query_blocks) + "\n"

    def _build_language_prompt(self, prompt_base, time_steps):
        prompt_body = prompt_base.replace(
            "You should locate the start and end frames of the action in the video provided.",
            "You should describe the actions present in the video using the provided action definitions.",
        )
        task_instruction = (
            "Describe the actions present in this video using the provided action definitions. "
            "Mention only actions that appear in the video and do not output timestamps or frame numbers."
        )
        return (
            self._build_visual_token_str(time_steps)
            + self._build_query_token_str(time_steps)
            + prompt_body.strip()
            + "\n"
            + task_instruction
        )

    def _extract_token_states(self, hidden_states, input_ids, token_id, batch_size):
        if token_id is None:
            return None
        token_mask = input_ids == token_id
        selected = hidden_states[:, : input_ids.shape[1], :][token_mask]
        if selected.numel() == 0:
            return None
        if selected.dim() == 3:
            selected = selected.squeeze(0)
        return rearrange(selected, "(b t) c -> b t c", b=batch_size)

    def _build_cls_token_features(self, cls_tokens, semantic_context=None, qwen_semantic_context=None):
        if cls_tokens is None:
            return None
        if self.cls_token_feature_mode == "query_bank" and self.cls_query_attn is not None:
            query_bank = self.cls_query_bank.to(device=cls_tokens.device, dtype=cls_tokens.dtype)
            query_bank = query_bank.unsqueeze(0).expand(cls_tokens.shape[0], -1, -1)
            context_terms = []
            if semantic_context is not None:
                context_terms.append(semantic_context.mean(dim=1, keepdim=True).to(dtype=cls_tokens.dtype))
            if qwen_semantic_context is not None:
                resized_qwen_context = self._resize_temporal_condition(qwen_semantic_context, cls_tokens.shape[1])
                if resized_qwen_context is not None:
                    context_terms.append(resized_qwen_context.mean(dim=1, keepdim=True).to(dtype=cls_tokens.dtype))
            if context_terms:
                query_bank = query_bank + torch.stack(context_terms, dim=0).mean(dim=0)
            summary_tokens, _ = self.cls_query_attn(query_bank, cls_tokens, cls_tokens, need_weights=False)
            summary = summary_tokens.mean(dim=1, keepdim=True).expand(-1, cls_tokens.shape[1], -1)
            if context_terms:
                summary = summary + torch.stack(
                    [context.expand(-1, cls_tokens.shape[1], -1) for context in context_terms], dim=0
                ).mean(dim=0)
            fused = self.cls_query_proj(torch.cat([cls_tokens, summary], dim=-1))
            gate = torch.sigmoid(self.cls_query_gate(torch.cat([cls_tokens, summary], dim=-1)))
            return self.cls_query_norm(cls_tokens + gate * fused)
        if self.cls_token_feature_mode != "contextual_residual" or self.cls_context_gate is None or semantic_context is None:
            return cls_tokens

        context_terms = [semantic_context.to(dtype=cls_tokens.dtype)]
        if qwen_semantic_context is not None:
            resized_qwen_context = self._resize_temporal_condition(qwen_semantic_context, cls_tokens.shape[1])
            if resized_qwen_context is not None:
                context_terms.append(resized_qwen_context.to(dtype=cls_tokens.dtype))

        fused_context = context_terms[0] if len(context_terms) == 1 else torch.stack(context_terms, dim=0).mean(dim=0)
        gate = torch.sigmoid(self.cls_context_gate(torch.cat([cls_tokens, fused_context], dim=-1)))
        return self.cls_context_norm(cls_tokens + gate * fused_context)

    def _compute_language_weight(self, adv_predictions):
        if adv_predictions is None:
            return None
        return 2.0 * torch.sigmoid(F.relu(adv_predictions)) - 1.0

    def _scale_language_weight(self, shared_weight, branch_scale):
        scaled_weight = (shared_weight.clamp(min=0.0, max=1.0) * branch_scale).clamp(min=0.0, max=1.0)
        return self.lang_gate_floor + (1.0 - self.lang_gate_floor) * scaled_weight

    def _fallback_language_tensor(self, batch_size, time_steps, device, dtype):
        return torch.full((batch_size, time_steps, 1), self.fallback_lang_weight, device=device, dtype=dtype)

    def _compute_similarity_gate(self, lang_tokens, visual_tokens):
        if not self.use_similarity_gate:
            return torch.ones(lang_tokens.shape[:-1] + (1,), device=lang_tokens.device, dtype=lang_tokens.dtype)

        output_dtype = lang_tokens.dtype
        lang_tokens = F.normalize(lang_tokens.float(), dim=-1)
        visual_tokens = F.normalize(visual_tokens.float(), dim=-1)
        similarity = (lang_tokens * visual_tokens).sum(dim=-1, keepdim=True)
        gate = torch.sigmoid(self.similarity_gate_scale * similarity)
        return gate.to(dtype=output_dtype)

    def _compute_branch_gate(self, lang_tokens, visual_tokens, gate_proj, semantic_confidence=None):
        if gate_proj is None:
            return torch.ones(lang_tokens.shape[:-1] + (1,), device=lang_tokens.device, dtype=lang_tokens.dtype)

        gate_input = torch.cat([visual_tokens, lang_tokens], dim=-1)
        gate = torch.sigmoid(gate_proj(gate_input))
        if semantic_confidence is not None:
            gate = gate * semantic_confidence.to(dtype=gate.dtype)
        return gate

    def _build_language_residual(self, lang_tokens, visual_tokens, adapter, residual_norm):
        lang_tokens = adapter(lang_tokens)
        if self.use_delta_fusion:
            lang_tokens = lang_tokens - visual_tokens
        lang_tokens = residual_norm(lang_tokens)
        lang_tokens = self.lang_residual_dropout(lang_tokens)
        return lang_tokens

    def _format_segment_value(self, value):
        rounded = round(float(value))
        if abs(float(value) - rounded) < 1e-3:
            return str(int(rounded))
        return f"{float(value):.1f}"

    def _extract_action_descriptions(self, prompt_base, class_map):
        descriptions = {class_name: class_name for class_name in class_map}
        marker = "Here we provide the descriptions of each action:"
        description_block = prompt_base.split(marker, 1)[1] if marker in prompt_base else prompt_base
        for raw_line in description_block.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            class_name, description = line.split(":", 1)
            class_name = class_name.strip()
            description = description.strip()
            if class_name in descriptions and description:
                descriptions[class_name] = description
        return descriptions

    def _build_description_answers(self, gt_segments, gt_labels, metas, prompt_base):
        answers = []
        id_action_map = {i: v for i, v in enumerate(metas[0]["class_map"])}
        descriptions = self._extract_action_descriptions(prompt_base, metas[0]["class_map"])
        for batch_segments, batch_labels in zip(gt_segments, gt_labels):
            if batch_segments.numel() == 0:
                answers.append("The video does not contain any of the listed actions.")
                continue

            order = torch.argsort(batch_segments[:, 0])
            seen = set()
            lines = ["The video shows:"]
            for idx in order.tolist():
                lab = batch_labels[idx]
                label_idx = int(lab.item())
                if label_idx in seen:
                    continue
                seen.add(label_idx)
                action_name = id_action_map[label_idx]
                action_description = descriptions.get(action_name, action_name)
                if not action_description.endswith("."):
                    action_description += "."
                lines.append(f"{action_name}: {action_description}")
            answers.append("\n".join(lines))
        return answers

    def _build_clip_targets(self, gt_labels, num_classes, device):
        clip_targets = torch.zeros((len(gt_labels), num_classes), device=device, dtype=torch.float32)
        for batch_idx, labels in enumerate(gt_labels):
            if labels.numel() == 0:
                continue
            clip_targets[batch_idx, torch.unique(labels.long())] = 1.0
        return clip_targets

    def _build_class_texts(self, class_map, prompt_base):
        descriptions = self._extract_action_descriptions(prompt_base, class_map)
        return [f"{class_name}: {descriptions.get(class_name, class_name)}" for class_name in class_map]

    def _build_dense_class_targets(self, gt_segments, gt_labels, time_steps, num_classes, device):
        dense_targets = torch.zeros((len(gt_labels), time_steps, num_classes), device=device, dtype=torch.float32)
        for batch_idx, (segments, labels) in enumerate(zip(gt_segments, gt_labels)):
            if segments.numel() == 0 or labels.numel() == 0:
                continue
            for seg, label in zip(segments, labels.long()):
                start = max(int(torch.floor(seg[0]).item()), 0)
                end = min(int(torch.ceil(seg[1]).item()), time_steps)
                if end <= start:
                    end = min(start + 1, time_steps)
                dense_targets[batch_idx, start:end, label] = 1.0
        return dense_targets

    def _compute_dense_semantic_loss(self, semantic_logits, gt_segments, gt_labels):
        dense_targets = self._build_dense_class_targets(
            gt_segments,
            gt_labels,
            semantic_logits.shape[1],
            semantic_logits.shape[2],
            semantic_logits.device,
        )
        pos_mask = dense_targets > 0
        loss = F.binary_cross_entropy_with_logits(semantic_logits, dense_targets, reduction="none")
        weights = torch.ones_like(loss)
        if pos_mask.any():
            pos_count = pos_mask.sum().float().clamp(min=1.0)
            neg_count = (~pos_mask).sum().float().clamp(min=1.0)
            pos_weight = torch.clamp(neg_count / pos_count, max=20.0)
            weights = weights.masked_fill(pos_mask, pos_weight)
        return (loss * weights).sum() / weights.sum().clamp(min=1.0)

    def _compute_prototype_alignment_loss(
        self, cls_token_features, class_prototypes, gt_segments, gt_labels, qwen_visual_features=None
    ):
        if cls_token_features is None or class_prototypes is None:
            return None

        target_len = cls_token_features.shape[1]
        dense_targets = self._build_dense_class_targets(
            gt_segments, gt_labels, target_len, class_prototypes.shape[0], cls_token_features.device
        )
        pos_mask = dense_targets.sum(dim=-1) > 0
        if not pos_mask.any():
            return None

        prototype_bank = F.normalize(class_prototypes.float(), dim=-1)
        target_dist = dense_targets / dense_targets.sum(dim=-1, keepdim=True).clamp(min=1.0)
        losses = []

        for features in (cls_token_features, qwen_visual_features):
            resized_features = self._resize_temporal_condition(features, target_len)
            if resized_features is None:
                continue
            resized_features = resized_features.to(dtype=cls_token_features.dtype)
            feature_bank = F.normalize(resized_features.float(), dim=-1)
            logits = torch.matmul(feature_bank, prototype_bank.transpose(0, 1)) / self.semantic_temperature
            log_prob = torch.log_softmax(logits, dim=-1)
            per_pos_loss = -(target_dist * log_prob)[pos_mask].sum(dim=-1)
            if per_pos_loss.numel() > 0:
                losses.append(per_pos_loss.mean())

        if not losses:
            return None
        return sum(losses) / len(losses)

    def _resize_temporal_condition(self, tensor, target_len):
        if tensor is None or tensor.dim() != 3:
            return None
        if tensor.shape[1] != target_len:
            tensor = F.interpolate(
                tensor.permute(0, 2, 1), size=target_len, mode="linear", align_corners=False
            ).permute(0, 2, 1)
        return tensor

    def _extract_qwen_visual_features(self, video_frames):
        if (
            video_frames is None
            or self.qwen_visual_num_frames <= 0
            or self.arch_type not in {"qwen", "qwen3"}
            or not hasattr(self.mllm, "encode_video_visual")
        ):
            return None

        sampled_frames = self.get_frames_from_video(
            video_frames, n_frames=self.qwen_visual_num_frames, sample_type="uniform", resize=None, norm=False
        )
        if sampled_frames is None:
            return None

        sampled_frames = sampled_frames.float()
        if sampled_frames.max() <= 1.0:
            sampled_frames = sampled_frames * 255.0
        sampled_frames = sampled_frames.clamp(0.0, 255.0)

        qwen_visual_features, _ = self.mllm.encode_video_visual(sampled_frames)
        if qwen_visual_features is None:
            return None
        return qwen_visual_features.to(device=video_frames.device)

    def _encode_texts(self, texts, device):
        ids = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").data
        input_ids = ids["input_ids"].to(device=device)
        attention_mask = ids["attention_mask"].to(device=device)

        was_training = self.mllm.training
        self.mllm.eval()
        with torch.no_grad():
            outputs = self.mllm(
                visual_features=None,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        self.mllm.train(was_training)

        hidden_states = outputs.hidden_states[-1]
        last_token_idx = attention_mask.long().sum(dim=1) - 1
        pooled = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), last_token_idx]
        return pooled

    def _compute_semantic_prior(self, query_tokens, class_prototypes):
        if (
            class_prototypes is None
            or self.semantic_query_proj is None
            or self.semantic_context_proj is None
            or self.semantic_context_norm is None
        ):
            return None, None

        query = F.normalize(self.semantic_query_proj(query_tokens).float(), dim=-1)
        prototypes = F.normalize(class_prototypes.float(), dim=-1)
        semantic_logits = torch.matmul(query, prototypes.transpose(0, 1)) / self.semantic_temperature
        semantic_weights = torch.softmax(semantic_logits, dim=-1)
        semantic_context = torch.matmul(semantic_weights.to(class_prototypes.dtype), class_prototypes)
        semantic_context = self.semantic_context_proj(semantic_context.to(query_tokens.dtype))
        semantic_context = self.semantic_context_norm(semantic_context)
        return semantic_context, semantic_logits

    def _compute_semantic_confidence(self, semantic_logits, dtype):
        if semantic_logits is None:
            return None
        semantic_confidence = torch.softmax(semantic_logits.detach(), dim=-1).amax(dim=-1, keepdim=True)
        return semantic_confidence.to(dtype=dtype)

    def _build_cls_logit_bias(self, semantic_logits, semantic_confidence, dtype, shared_lang_weight=None):
        if semantic_logits is None:
            return None
        cls_logit_bias = semantic_logits - semantic_logits.mean(dim=-1, keepdim=True)
        cls_logit_bias = cls_logit_bias / cls_logit_bias.detach().std(dim=-1, keepdim=True).clamp(min=1.0)
        cls_logit_bias = torch.tanh(cls_logit_bias)
        if semantic_confidence is not None:
            cls_logit_bias = cls_logit_bias * semantic_confidence.to(dtype=cls_logit_bias.dtype)
        if shared_lang_weight is not None:
            cls_logit_bias = cls_logit_bias * shared_lang_weight.to(dtype=cls_logit_bias.dtype)
        return cls_logit_bias.to(dtype=dtype) * self.cls_lang_scale

    def _tokenize_language_supervision(self, input_text, batch_size, answers, device):
        prompt_texts = [input_text for _ in range(batch_size)]
        prompt_ids = self.tokenizer(prompt_texts, padding=True, truncation=True, return_tensors="pt").data

        input_texts = prompt_texts
        if answers is not None:
            suffix = self.template.get("SUFFIX", "")
            input_texts = [prompt_texts[i] + answers[i] + suffix for i in range(batch_size)]

        ids = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").data
        input_ids = ids["input_ids"].to(device=device)
        attention_mask = ids["attention_mask"].to(device=device)

        labels = None
        if answers is not None:
            prompt_lengths = prompt_ids["attention_mask"].sum(dim=1).tolist()
            labels = input_ids.clone()
            for idx, prompt_len in enumerate(prompt_lengths):
                labels[idx, : int(prompt_len)] = -100

        return input_ids, attention_mask, labels

    def _get_module_dtype(self, module, fallback=None):
        if module is None:
            return fallback if fallback is not None else self.torch_dtype
        for param in module.parameters():
            return param.dtype
        for buffer in module.buffers():
            return buffer.dtype
        return fallback if fallback is not None else self.torch_dtype

    def forward(self, x, metas, prompt_base, answers=None, use_language=True, gt_segments=None, gt_labels=None, **kwargs):
        if not self.init_prediction_config:
            self.preparing_for_generation(max_new_tokens=1)
            trainable = sum(p.numel() for p in self.mllm.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.mllm.parameters())
            print(f"Trainable params: {trainable} / {total} ({trainable / total:.2%})")

        window_size = x.shape[3]
        if self.clip_window <= 0 or window_size % self.clip_window != 0:
            self.clip_window = window_size

        predictions = []
        video_frames = x.squeeze(1)
        qwen_visual_features = None
        if (
            self.qwen_visual_num_frames > 0
            and self.arch_type in {"qwen", "qwen3"}
            and self.fusion_mode
            in {
                "dual_visual_cls_head",
                "cross_modal_cls_head",
                "expert_router_cls_head",
                "candidate_reranker_cls_head",
                "distill_cls_head",
            }
        ):
            qwen_visual_features = self._extract_qwen_visual_features(video_frames)

        x = video_frames
        size = (160, 160) if self.vit is not None else (self.image_size, self.image_size)
        x = self.get_frames_from_video(x, sample_type="none", resize=size, norm=True)

        if self.vit is not None:
            chunk_num = window_size // 16
            x = rearrange(x, "b (t1 t) c h w -> (b t1) c t h w", t1=chunk_num)
            x = x.to(dtype=self._get_module_dtype(self.vit), device=next(self.vit.parameters()).device)
            x = self.vit(x)
            x = reduce(x, "b c t h w -> b c t", "mean")
            x = rearrange(x, "(b t1) c t -> b c (t1 t)", t1=chunk_num)
            x = F.interpolate(x, size=window_size, mode="linear", align_corners=False)
        elif "vision" in self.backbone_type:
            chunk_num = window_size // self.clip_window
            x = rearrange(x, "b (t1 t) c h w -> (b t1) t c h w", t1=chunk_num)
            vision_dtype = self._get_module_dtype(getattr(self.mllm.model, "visual", None))
            x = x.to(dtype=vision_dtype)
            x = self.mllm.vision_forward(x, self.temporal_positional_embedding.to(x.device))
            x = rearrange(x, "(b t1) t c -> b c (t1 t)", t1=chunk_num)

        batch_size, _, time_steps = x.shape
        loss = None
        aux = {
            "adv_predictions": None,
            "lambda_lang": None,
            "semantic_loss": None,
            "prototype_align_loss": None,
            "cls_logit_bias": None,
            "class_prototypes": None,
            "cls_token_features": None,
            "cls_token_confidence": None,
            "qwen_visual_features": None,
            "qwen_visual_confidence": None,
            "teacher_cls_targets": None,
        }

        if "language" not in self.backbone_type:
            return x, loss, predictions, aux

        x = rearrange(x, "b c t -> b t c")
        x = self.backbone_language_mlp(x)
        x_in = x.clone()

        if not use_language:
            return self._build_vision_only_features(x_in), None, predictions, aux

        text = self._build_language_prompt(prompt_base, time_steps)
        input_text = self.template["INSTRUCTION"].format(input=text, round=1, bot_name=self.bot_name)
        input_ids, attention_mask, labels = self._tokenize_language_supervision(
            input_text, batch_size, answers, x.device
        )

        mm_inputs = {
            "x": x,
            "visual_features": x,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": None,
            "labels": labels,
            "prompt_masks": None,
            "vp_overall_mask": None,
            "fast_pixel_values": None,
            "fast_token_idx": None,
        }

        generate_output = self.mllm(
            **mm_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        last_hidden_states = generate_output.hidden_states[-1]
        if generate_output.loss is not None:
            loss = generate_output.loss * self.loss_weight

        if self.backbone_type.endswith("token"):
            x_cls = self._extract_token_states(last_hidden_states, input_ids, self.cls_token_id, batch_size)
            x_loc = self._extract_token_states(last_hidden_states, input_ids, self.loc_token_id, batch_size)
            x_adv = self._extract_token_states(last_hidden_states, input_ids, self.adv_token_id, batch_size)

            if x_cls is None or (self.fusion_mode == "feature" and x_loc is None):
                return self._build_vision_only_features(x_in), loss, predictions, aux

            semantic_logits = None
            semantic_context = None
            class_prototypes = None
            qwen_semantic_context = None
            qwen_semantic_logits = None
            qwen_visual_confidence = None
            semantic_supervision_logits = None
            teacher_cls_targets = None
            if metas and self.semantic_query_proj is not None:
                class_texts = self._build_class_texts(metas[0]["class_map"], prompt_base)
                class_prototypes = self._encode_texts(class_texts, x.device).to(dtype=x_in.dtype)
                semantic_context, semantic_logits = self._compute_semantic_prior(x_cls, class_prototypes)

                if qwen_visual_features is not None:
                    qwen_visual_features = qwen_visual_features.to(device=x_in.device, dtype=x_in.dtype)
                    qwen_semantic_context, qwen_semantic_logits = self._compute_semantic_prior(
                        qwen_visual_features, class_prototypes
                    )
                    qwen_visual_confidence = self._compute_semantic_confidence(qwen_semantic_logits, x_in.dtype)

                semantic_supervision_logits = semantic_logits
                if qwen_semantic_logits is not None:
                    resized_qwen_logits = self._resize_temporal_condition(
                        qwen_semantic_logits, semantic_logits.shape[1] if semantic_logits is not None else time_steps
                    )
                    semantic_supervision_logits = (
                        resized_qwen_logits
                        if semantic_supervision_logits is None
                        else 0.5 * (semantic_supervision_logits + resized_qwen_logits)
                    )
                teacher_cls_targets = semantic_supervision_logits

                if gt_labels is not None and semantic_supervision_logits is not None and self.semantic_loss_weight > 0:
                    if self.fusion_mode in {
                        "cls_logit_bias",
                        "language_conditioned_cls_head",
                        "dual_visual_cls_head",
                        "cross_modal_cls_head",
                        "expert_router_cls_head",
                        "candidate_reranker_cls_head",
                        "distill_cls_head",
                    }:
                        aux["semantic_loss"] = (
                            self._compute_dense_semantic_loss(semantic_supervision_logits, gt_segments, gt_labels)
                            * self.semantic_loss_weight
                        )
                    else:
                        clip_logits = semantic_supervision_logits.max(dim=1).values
                        clip_targets = self._build_clip_targets(gt_labels, clip_logits.shape[-1], clip_logits.device)
                        aux["semantic_loss"] = (
                            F.binary_cross_entropy_with_logits(clip_logits, clip_targets) * self.semantic_loss_weight
                        )
            semantic_confidence = self._compute_semantic_confidence(
                semantic_supervision_logits if semantic_supervision_logits is not None else semantic_logits, x_in.dtype
            )
            cls_token_features = self._build_cls_token_features(x_cls, semantic_context, qwen_semantic_context)
            if (
                gt_labels is not None
                and class_prototypes is not None
                and cls_token_features is not None
                and self.prototype_align_loss_weight > 0
            ):
                prototype_align_loss = self._compute_prototype_alignment_loss(
                    cls_token_features, class_prototypes, gt_segments, gt_labels, qwen_visual_features
                )
                if prototype_align_loss is not None:
                    aux["prototype_align_loss"] = prototype_align_loss * self.prototype_align_loss_weight

            if x_adv is not None and self.adv_predictor is not None:
                adv_predictions = self.adv_predictor(x_adv).squeeze(-1)
                shared_lang_weight = self._compute_language_weight(adv_predictions).unsqueeze(-1)
                aux["adv_predictions"] = adv_predictions
            else:
                shared_lang_weight = None
                aux["adv_predictions"] = None

            if self.fusion_mode == "cls_logit_bias":
                aux["cls_logit_bias"] = self._build_cls_logit_bias(
                    semantic_supervision_logits, semantic_confidence, x_in.dtype, shared_lang_weight
                )
                if aux["cls_logit_bias"] is not None:
                    aux["lambda_lang"] = aux["cls_logit_bias"].abs().mean(dim=-1)
                x_vis = self.norm(x_in)
                x = torch.cat([x_vis, x_vis], dim=0)
                x = rearrange(x, "b t c -> b c t")
            elif self._uses_language_conditioned_classifier():
                aux["class_prototypes"] = class_prototypes
                aux["cls_token_features"] = cls_token_features
                aux["cls_token_confidence"] = semantic_confidence
                aux["qwen_visual_features"] = qwen_visual_features
                aux["qwen_visual_confidence"] = qwen_visual_confidence
                if semantic_confidence is not None:
                    aux["lambda_lang"] = semantic_confidence.squeeze(-1)
                elif qwen_visual_confidence is not None:
                    aux["lambda_lang"] = self._resize_temporal_condition(qwen_visual_confidence, time_steps).squeeze(-1)
                x_vis = self.norm(x_in)
                x = torch.cat([x_vis, x_vis], dim=0)
                x = rearrange(x, "b t c -> b c t")
            elif self.fusion_mode == "distill_cls_head":
                aux["teacher_cls_targets"] = teacher_cls_targets
                aux["qwen_visual_features"] = qwen_visual_features
                aux["qwen_visual_confidence"] = qwen_visual_confidence
                if teacher_cls_targets is not None:
                    aux["lambda_lang"] = teacher_cls_targets.sigmoid().amax(dim=-1)
                x_vis = self.norm(x_in)
                x = torch.cat([x_vis, x_vis], dim=0)
                x = rearrange(x, "b t c -> b c t")
            else:
                lang_cls = self._build_language_residual(x_cls, x_in, self.lang_cls_adapter, self.lang_cls_norm)
                lang_loc = self._build_language_residual(x_loc, x_in, self.lang_loc_adapter, self.lang_loc_norm)
                if shared_lang_weight is None:
                    shared_lang_weight = torch.ones((batch_size, time_steps, 1), device=x_in.device, dtype=x_in.dtype)
                cls_gate = self._compute_branch_gate(lang_cls, x_in, self.lang_cls_gate, semantic_confidence)
                loc_gate = self._compute_branch_gate(lang_loc, x_in, self.lang_loc_gate, semantic_confidence)
                lambda_cls = (shared_lang_weight * cls_gate).clamp(min=0.0, max=1.0) * self.cls_lang_scale
                lambda_loc = (shared_lang_weight * cls_gate.detach() * loc_gate).clamp(min=0.0, max=1.0) * self.loc_lang_scale
                aux["lambda_lang"] = lambda_cls.squeeze(-1)
                x_cls = x_in + lambda_cls * lang_cls
                x_cls = self.norm(x_cls)
                x_loc = self.norm(x_in + lambda_loc * lang_loc)
                x = torch.cat([x_cls, x_loc], dim=0)
                x = rearrange(x, "b t c -> b c t")
        else:
            frame_token_mask = input_ids == self.img_cls_token_id
            x = last_hidden_states[:, : input_ids.shape[1], :][frame_token_mask]
            if x.dim() == 2:
                x = x.unsqueeze(0)
            lang_residual = self._build_language_residual(x, x_in, self.lang_cls_adapter, self.lang_cls_norm)
            lambda_lang = self._compute_branch_gate(lang_residual, x_in, self.lang_cls_gate) * self.cls_lang_scale
            aux["lambda_lang"] = lambda_lang.squeeze(-1)
            x = self.norm(x_in + lambda_lang * lang_residual)
            x = rearrange(x, "b t c -> b c t")

        return x, loss, predictions, aux

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(state_dict, strict, assign)

    def all_state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if self.mllm is None:
            return state_dict

        to_return = OrderedDict()

        if self.mllm.use_visual_encoder_lora:
            if hasattr(self.mllm.model, "vision_model"):
                to_return.update(get_peft_model_state_dict(self.mllm.model.vision_model, state_dict=state_dict))
            elif hasattr(self.mllm, "get_visual_module"):
                visual_module = self.mllm.get_visual_module()
                if visual_module is not None:
                    to_return.update(get_peft_model_state_dict(visual_module, state_dict=state_dict))
            elif hasattr(self.mllm.model, "visual"):
                to_return.update(get_peft_model_state_dict(self.mllm.model.visual, state_dict=state_dict))
        elif not self.mllm.freeze_visual_encoder:
            visual_prefixes = ["visual_encoder.", "model.visual."]
            if hasattr(self.mllm, "get_visual_state_prefixes"):
                visual_prefixes.extend(self.mllm.get_visual_state_prefixes())
            to_return.update({k: v for k, v in state_dict.items() if any(k.startswith(prefix) for prefix in visual_prefixes)})

        if self.mllm.use_llm_lora:
            if self.arch_type == "intern_vl":
                to_return.update(get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict))
            elif self.arch_type in {"qwen", "qwen3"}:
                llm_module = (
                    self.mllm.get_llm_module()
                    if hasattr(self.mllm, "get_llm_module")
                    else self.mllm.model.language_model
                )
                to_return.update(get_peft_model_state_dict(llm_module, state_dict=state_dict))
            elif self.arch_type == "llava":
                to_return.update(get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict))
        elif not self.mllm.freeze_llm:
            llm_prefixes = ["llm.", "model.language_model."]
            if hasattr(self.mllm, "get_llm_state_prefixes"):
                llm_prefixes.extend(self.mllm.get_llm_state_prefixes())
            to_return.update(
                {k: v for k, v in state_dict.items() if any(k.startswith(prefix) for prefix in llm_prefixes)}
            )

        to_return.update({k: v for k, v in state_dict.items() if "mlp1." in k})
        to_return.update({k: v for k, v in state_dict.items() if "model.multi_modal_projector." in k})
        to_return.update({k: v for k, v in state_dict.items() if "mask_decoder" in k})
        to_return.update({k: v for k, v in state_dict.items() if "text_hidden_fcs." in k})
        to_return.update({k: v for k, v in state_dict.items() if "text_exist_fcs." in k})
        to_return.update(
            {k: v for k, v in state_dict.items() if "lm_head.weight" in k or ("output" in k and "sam2_model" not in k)}
        )
        to_return.update({k: v for k, v in state_dict.items() if "embed_tokens.weight" in k or "tok_embeddings" in k})
        to_return.update(
            {
                k: v
                for k, v in state_dict.items()
                if k.startswith("backbone_language_mlp.")
                or k.startswith("norm.")
                or k.startswith("lang_")
                or k.startswith("cls_context_")
                or k.startswith("cls_query_")
                or k.startswith("semantic_")
                or k.startswith("adv_predictor.")
                or k.startswith("vit.")
                or k == "temporal_positional_embedding"
            }
        )
        return to_return

    def get_frames_from_video(self, frames, n_frames=5, sample_type="uniform", resize=None, norm=False):
        batch_size, channels, time_steps, height, width = frames.shape
        device = frames.device

        def process(frames_):
            if resize is not None and (height, width) != resize:
                frames_ = frames_.reshape(batch_size * time_steps, channels, height, width)
                frames_ = F.interpolate(frames_, size=resize, mode="bilinear", align_corners=False)
                frames_ = frames_.view(batch_size, time_steps, channels, *resize)
            if norm:
                frames_ = rearrange(frames_, "n t c h w -> n c t h w")
                frames_ = self.data_preprocessor.preprocess(list(frames_), data_samples=None, training=False)[0]
                frames_ = rearrange(frames_, "n c t h w -> n t c h w")
            return frames_

        if sample_type == "uniform":
            idxs = torch.linspace(0, time_steps - 1, steps=n_frames, device=device).long()
            frames = frames.index_select(2, idxs)

        frames = frames.permute(0, 2, 1, 3, 4)
        return process(frames)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        if hasattr(self.mllm, "activation_checkpointing_enable"):
            self.mllm.activation_checkpointing_enable()
        elif hasattr(self.mllm.model, "language_model"):
            enable_gradient_checkpointing(self.mllm.model.language_model)
        elif hasattr(self.mllm, "get_llm_module"):
            enable_gradient_checkpointing(self.mllm.get_llm_module())

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        if hasattr(self.mllm, "activation_checkpointing_disable"):
            self.mllm.activation_checkpointing_disable()
        elif hasattr(self.mllm.model, "language_model"):
            self.mllm.model.language_model.gradient_checkpointing_disable()
        elif hasattr(self.mllm, "get_llm_module"):
            self.mllm.get_llm_module().gradient_checkpointing_disable()

    def process_rpn_head(self, rpn_head):
        if "projection" in rpn_head and rpn_head["projection"] is not None:
            rpn_head["projection"]["in_channels"] = self.mllm_embedding_dim
            print(f"RPN head in_channels is modified to: {rpn_head['projection']['in_channels']}", flush=True)

        if self.fusion_mode in {
            "language_conditioned_cls_head",
            "dual_visual_cls_head",
            "cross_modal_cls_head",
            "expert_router_cls_head",
            "candidate_reranker_cls_head",
            "distill_cls_head",
        }:
            dense_head_cfg = None
            if "rpn_head" in rpn_head and isinstance(rpn_head["rpn_head"], dict):
                dense_head_cfg = rpn_head["rpn_head"]
            elif "type" in rpn_head:
                dense_head_cfg = rpn_head
            if dense_head_cfg is not None:
                mode_map = {
                    "language_conditioned_cls_head": "similarity",
                    "dual_visual_cls_head": "dual_visual",
                    "cross_modal_cls_head": "cross_attention",
                    "expert_router_cls_head": "expert_router",
                    "candidate_reranker_cls_head": "candidate_reranker",
                    "distill_cls_head": "distill",
                }
                lang_classifier_cfg = dict(dense_head_cfg.get("lang_classifier_cfg", {}))
                lang_classifier_cfg.update(
                    dict(
                    enabled=self.fusion_mode != "distill_cls_head",
                    mode=mode_map[self.fusion_mode],
                    text_dim=self.mllm_embedding_dim,
                    embed_dim=dense_head_cfg.get("feat_channels", dense_head_cfg.get("in_channels", 256)),
                        num_heads=4,
                        distill_loss_weight=self.teacher_loss_weight,
                        weight_mode=self.lang_weight_mode,
                        rerank_topk=self.rerank_topk,
                        candidate_topk=self.rerank_topk,
                    )
                )
                dense_head_cfg["lang_classifier_cfg"] = lang_classifier_cfg
        return rpn_head


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

        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]

        if max_cache_length is not None and attention_mask is not None and cache_length + input_ids.shape[1] > max_cache_length:
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    if inputs_embeds is not None and (past_key_values is None or len(past_key_values) == 0):
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

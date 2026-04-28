import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..builder import HEADS, build_prior_generator, build_loss
from ..bricks import ConvModule, Scale
from ..utils.functions import gaussian_cdf, tan_curve


@HEADS.register_module()
class AnchorFreeHead(nn.Module):
    def __init__(
            self,
            num_classes,
            in_channels,
            feat_channels,
            num_convs=3,
            prior_generator=None,
            loss=None,
            loss_normalizer=100,
            loss_normalizer_momentum=0.9,
            center_sample="radius",
            center_sample_radius=1.5,
            label_smoothing=0,
            cls_prior_prob=0.01,
            loss_weight=1.0,
            filter_similar_gt=True,
            lang_classifier_cfg=None,
    ):
        super(AnchorFreeHead, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.cls_prior_prob = cls_prior_prob
        self.label_smoothing = label_smoothing
        self.filter_similar_gt = filter_similar_gt
        self.lang_classifier_cfg = lang_classifier_cfg or {}
        self.use_language_conditioned_classifier = bool(self.lang_classifier_cfg.get("enabled", False))
        self.lang_mode = str(self.lang_classifier_cfg.get("mode", "similarity"))
        self.lang_text_dim = int(self.lang_classifier_cfg.get("text_dim", self.feat_channels))
        self.lang_embed_dim = int(self.lang_classifier_cfg.get("embed_dim", self.feat_channels))
        requested_lang_heads = max(int(self.lang_classifier_cfg.get("num_heads", 4)), 1)
        self.lang_num_heads = requested_lang_heads
        while self.lang_embed_dim % self.lang_num_heads != 0 and self.lang_num_heads > 1:
            self.lang_num_heads -= 1
        self.distill_loss_weight = float(max(self.lang_classifier_cfg.get("distill_loss_weight", 0.0), 0.0))
        self.lang_weight_mode = str(self.lang_classifier_cfg.get("weight_mode", "confidence"))
        self.rerank_topk = max(int(self.lang_classifier_cfg.get("rerank_topk", 0)), 0)
        self.lang_candidate_topk = max(int(self.lang_classifier_cfg.get("candidate_topk", max(self.rerank_topk, 3))), 2)

        self.loss_weight = loss_weight
        self.center_sample = center_sample
        self.center_sample_radius = center_sample_radius
        self.loss_normalizer_momentum = loss_normalizer_momentum
        self.register_buffer("loss_normalizer", torch.tensor(loss_normalizer))  # save in the state_dict

        # point generator
        self.prior_generator = build_prior_generator(prior_generator)

        self._init_layers()

        self.cls_loss = build_loss(loss.cls_loss)
        self.reg_loss = build_loss(loss.reg_loss)

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        # self._init_prog_convs()
        self._init_heads()

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.cls_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.reg_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )

    def _init_prog_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.prog_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.prog_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )
        self.phase_pref_weight = nn.Parameter(torch.ones(self.num_classes) * 0.5, requires_grad=True)

    def _init_heads(self):
        """Initialize predictor layers of the head."""
        self.cls_head = nn.Conv1d(self.feat_channels, self.num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv1d(self.feat_channels, 2, kernel_size=3, padding=1)
        # self.prog_head = nn.Conv1d(self.feat_channels, 1, kernel_size=3, padding=1)
        self.scale = nn.ModuleList([Scale() for _ in range(len(self.prior_generator.strides))])

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if self.cls_prior_prob > 0:
            bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
            nn.init.constant_(self.cls_head.bias, bias_value)

        if self.use_language_conditioned_classifier:
            self._init_language_conditioned_classifier()
        else:
            self.lang_visual_proj = None
            self.lang_token_proj = None
            self.lang_qwen_proj = None
            self.lang_text_proj = None
            self.lang_fusion_gate = None
            self.lang_query_proj = None
            self.lang_cross_attn = None
            self.lang_cross_norm = None
            self.lang_cross_gate = None
            self.lang_cross_cls = None
            self.lang_logit_scale = None
            self.lang_expert_router = None
            self.lang_candidate_mlp = None

    def _init_language_conditioned_classifier(self):
        self.lang_visual_proj = nn.Conv1d(self.feat_channels, self.lang_embed_dim, kernel_size=1)
        self.lang_token_proj = nn.Linear(self.lang_text_dim, self.lang_embed_dim, bias=False)
        self.lang_qwen_proj = nn.Linear(self.lang_text_dim, self.lang_embed_dim, bias=False)
        self.lang_text_proj = nn.Linear(self.lang_text_dim, self.lang_embed_dim, bias=False)
        self.lang_fusion_gate = nn.Conv1d(self.lang_embed_dim * 3, self.num_classes, kernel_size=1)
        self.lang_query_proj = nn.Conv1d(self.feat_channels, self.lang_embed_dim, kernel_size=1)
        self.lang_cross_attn = nn.MultiheadAttention(self.lang_embed_dim, self.lang_num_heads, batch_first=True)
        self.lang_cross_norm = nn.LayerNorm(self.lang_embed_dim)
        self.lang_cross_gate = nn.Linear(self.lang_embed_dim * 2, self.num_classes)
        self.lang_cross_cls = nn.Linear(self.lang_embed_dim, self.num_classes)
        self.lang_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.lang_expert_router = nn.Conv1d(self.lang_embed_dim * 3, self.num_classes * 3, kernel_size=1)
        self.lang_candidate_mlp = nn.Sequential(
            nn.Linear(self.lang_embed_dim * 4 + 4, self.lang_embed_dim),
            nn.GELU(),
            nn.Linear(self.lang_embed_dim, 1),
        )

    def _resize_cls_logit_bias(self, cls_logit_bias, target_len, dtype, level_idx=None):
        if cls_logit_bias is None:
            return None
        bias = cls_logit_bias[level_idx] if isinstance(cls_logit_bias, (list, tuple)) else cls_logit_bias
        if bias.dim() != 3:
            return None
        if bias.shape[1] != self.num_classes and bias.shape[-1] == self.num_classes:
            bias = bias.permute(0, 2, 1)
        elif bias.shape[1] != self.num_classes:
            return None
        if bias.shape[-1] != target_len:
            bias = F.interpolate(bias, size=target_len, mode="linear", align_corners=False)
        return bias.to(dtype=dtype)

    def _resize_time_condition(self, tensor, target_len, dtype):
        if tensor is None or tensor.dim() != 3:
            return None
        if tensor.shape[1] != target_len:
            tensor = F.interpolate(
                tensor.permute(0, 2, 1), size=target_len, mode="linear", align_corners=False
            ).permute(0, 2, 1)
        return tensor.to(dtype=dtype)

    def _resize_teacher_targets(self, teacher_cls_targets, target_len, dtype):
        if teacher_cls_targets is None or teacher_cls_targets.dim() != 3:
            return None
        if teacher_cls_targets.shape[1] == self.num_classes:
            teacher = teacher_cls_targets
        elif teacher_cls_targets.shape[-1] == self.num_classes:
            teacher = teacher_cls_targets.permute(0, 2, 1)
        else:
            return None
        if teacher.shape[-1] != target_len:
            teacher = F.interpolate(teacher, size=target_len, mode="linear", align_corners=False)
        return teacher.to(dtype=dtype)

    def _compute_soft_visual_weight(self, base_cls_logits, dtype):
        if base_cls_logits is None or base_cls_logits.dim() != 3:
            return None

        detached_logits = base_cls_logits.detach().float()
        class_probs = torch.softmax(detached_logits, dim=1)
        if class_probs.shape[1] > 1:
            top2 = torch.topk(class_probs, k=2, dim=1).values
            margin = top2[:, 0, :] - top2[:, 1, :]
        else:
            margin = class_probs[:, 0, :]
        entropy = -(class_probs * class_probs.clamp(min=1e-6).log()).sum(dim=1)
        if self.num_classes > 1:
            entropy = entropy / math.log(self.num_classes)
        uncertainty = (0.5 * (1.0 - margin) + 0.5 * entropy).clamp(min=0.0, max=1.0)
        return (0.35 + 0.65 * uncertainty).unsqueeze(-1).to(dtype=dtype)

    def _build_candidate_mask(self, base_cls_logits, dtype):
        if self.rerank_topk <= 0 or base_cls_logits is None or base_cls_logits.dim() != 3:
            return None

        topk = min(self.rerank_topk, self.num_classes)
        topk_idx = torch.topk(base_cls_logits.detach(), k=topk, dim=1).indices
        candidate_mask = torch.zeros_like(base_cls_logits, dtype=dtype)
        candidate_mask.scatter_(1, topk_idx, 1.0)
        return candidate_mask

    def _combine_confidences(self, cls_token_confidence, qwen_visual_confidence, target_len, dtype, base_cls_logits=None):
        confidence_terms = []
        cls_conf = self._resize_time_condition(cls_token_confidence, target_len, dtype)
        if cls_conf is not None:
            confidence_terms.append(cls_conf)
        qwen_conf = self._resize_time_condition(qwen_visual_confidence, target_len, dtype)
        if qwen_conf is not None:
            confidence_terms.append(qwen_conf)
        confidence = torch.stack(confidence_terms, dim=0).mean(dim=0) if confidence_terms else None

        if self.lang_weight_mode == "confidence":
            return confidence

        visual_weight = self._compute_soft_visual_weight(base_cls_logits, dtype)
        if self.lang_weight_mode == "visual":
            return visual_weight if visual_weight is not None else confidence
        if self.lang_weight_mode == "hybrid":
            weight_terms = []
            if confidence is not None:
                weight_terms.append(confidence)
            if visual_weight is not None:
                weight_terms.append(visual_weight)
            if weight_terms:
                return torch.stack(weight_terms, dim=0).mean(dim=0)
        return confidence

    def _prepare_language_experts(self, cls_feat, cls_token_features, class_prototypes, qwen_visual_features=None):
        if class_prototypes is None or class_prototypes.dim() != 2:
            return None

        text_embed = self.lang_text_proj(class_prototypes.to(device=cls_feat.device, dtype=cls_feat.dtype))
        text_embed = F.normalize(text_embed.float(), dim=-1)
        visual_embed = F.normalize(self.lang_visual_proj(cls_feat).permute(0, 2, 1).float(), dim=-1)
        visual_logits = torch.einsum("bte,ce->btc", visual_embed, text_embed)

        token_embed = torch.zeros_like(visual_embed)
        token_logits = None
        token_features = self._resize_time_condition(cls_token_features, cls_feat.shape[-1], cls_feat.dtype)
        if token_features is not None and token_features.shape[0] == cls_feat.shape[0]:
            token_embed = F.normalize(self.lang_token_proj(token_features).float(), dim=-1)
            token_logits = torch.einsum("bte,ce->btc", token_embed, text_embed)

        qwen_embed = torch.zeros_like(visual_embed)
        qwen_logits = None
        qwen_features = self._resize_time_condition(qwen_visual_features, cls_feat.shape[-1], cls_feat.dtype)
        if qwen_features is not None and qwen_features.shape[0] == cls_feat.shape[0]:
            qwen_embed = F.normalize(self.lang_qwen_proj(qwen_features).float(), dim=-1)
            qwen_logits = torch.einsum("bte,ce->btc", qwen_embed, text_embed)

        return dict(
            text_embed=text_embed,
            visual_embed=visual_embed,
            visual_logits=visual_logits,
            token_embed=token_embed,
            token_logits=token_logits,
            qwen_embed=qwen_embed,
            qwen_logits=qwen_logits,
        )

    def _compute_scaled_conditioned_logits(self, conditioned_logits, confidence, cls_feat, cls_lang_scale):
        if conditioned_logits is None:
            return None
        if confidence is not None:
            conditioned_logits = conditioned_logits * confidence
        scale = cls_lang_scale
        if self.lang_logit_scale is not None:
            scale = scale * torch.clamp(self.lang_logit_scale, min=0.0, max=4.0)
        return conditioned_logits.to(dtype=cls_feat.dtype) * scale

    def _compute_similarity_conditioned_logits(
        self,
        cls_feat,
        cls_token_features,
        class_prototypes,
        cls_token_confidence=None,
        qwen_visual_features=None,
        qwen_visual_confidence=None,
        cls_lang_scale=1.0,
        base_cls_logits=None,
    ):
        experts = self._prepare_language_experts(cls_feat, cls_token_features, class_prototypes, qwen_visual_features)
        if experts is None:
            return None

        logits_list = [experts["visual_logits"]]
        token_embed = experts["token_embed"]
        qwen_embed = experts["qwen_embed"]
        if experts["token_logits"] is not None:
            logits_list.append(experts["token_logits"])
        if experts["qwen_logits"] is not None:
            logits_list.append(experts["qwen_logits"])

        conditioned_logits = sum(logits_list) / len(logits_list)
        gate_input = torch.cat([experts["visual_embed"], token_embed, qwen_embed], dim=-1)
        gate_input = gate_input.permute(0, 2, 1).to(dtype=cls_feat.dtype)
        fusion_gate = torch.sigmoid(self.lang_fusion_gate(gate_input))
        confidence = self._combine_confidences(
            cls_token_confidence, qwen_visual_confidence, cls_feat.shape[-1], cls_feat.dtype, base_cls_logits
        )
        if confidence is not None:
            fusion_gate = fusion_gate * confidence.permute(0, 2, 1)
        conditioned_logits = self._compute_scaled_conditioned_logits(
            conditioned_logits.permute(0, 2, 1) * fusion_gate, None, cls_feat, cls_lang_scale
        )
        candidate_mask = self._build_candidate_mask(base_cls_logits, conditioned_logits.dtype)
        if candidate_mask is not None:
            conditioned_logits = conditioned_logits * candidate_mask
        return conditioned_logits

    def _compute_expert_router_logits(
        self,
        cls_feat,
        cls_token_features,
        class_prototypes,
        cls_token_confidence=None,
        qwen_visual_features=None,
        qwen_visual_confidence=None,
        cls_lang_scale=1.0,
        base_cls_logits=None,
    ):
        experts = self._prepare_language_experts(cls_feat, cls_token_features, class_prototypes, qwen_visual_features)
        if experts is None:
            return None

        visual_logits = experts["visual_logits"]
        token_logits = experts["token_logits"] if experts["token_logits"] is not None else visual_logits.new_zeros(visual_logits.shape)
        qwen_logits = experts["qwen_logits"] if experts["qwen_logits"] is not None else visual_logits.new_zeros(visual_logits.shape)
        expert_stack = torch.stack([visual_logits, token_logits, qwen_logits], dim=1)

        router_input = torch.cat([experts["visual_embed"], experts["token_embed"], experts["qwen_embed"]], dim=-1)
        router_input = router_input.permute(0, 2, 1).to(dtype=cls_feat.dtype)
        router_logits = self.lang_expert_router(router_input)
        router_logits = router_logits.view(cls_feat.shape[0], 3, self.num_classes, cls_feat.shape[-1]).permute(0, 1, 3, 2)
        router_weights = torch.softmax(router_logits.float(), dim=1).to(dtype=cls_feat.dtype)

        conditioned_logits = (router_weights * expert_stack.to(dtype=cls_feat.dtype)).sum(dim=1)
        confidence = self._combine_confidences(
            cls_token_confidence, qwen_visual_confidence, cls_feat.shape[-1], cls_feat.dtype, base_cls_logits
        )
        conditioned_logits = self._compute_scaled_conditioned_logits(conditioned_logits, confidence, cls_feat, cls_lang_scale)
        conditioned_logits = conditioned_logits.permute(0, 2, 1)
        candidate_mask = self._build_candidate_mask(base_cls_logits, conditioned_logits.dtype)
        if candidate_mask is not None:
            conditioned_logits = conditioned_logits * candidate_mask
        return conditioned_logits

    def _compute_candidate_reranker_logits(
        self,
        cls_feat,
        cls_token_features,
        class_prototypes,
        cls_token_confidence=None,
        qwen_visual_features=None,
        qwen_visual_confidence=None,
        cls_lang_scale=1.0,
        base_cls_logits=None,
    ):
        experts = self._prepare_language_experts(cls_feat, cls_token_features, class_prototypes, qwen_visual_features)
        if experts is None:
            return None

        base_scores = experts["visual_logits"] if base_cls_logits is None else base_cls_logits.detach().permute(0, 2, 1).float()
        topk = min(max(self.lang_candidate_topk, 2), self.num_classes)
        topk_idx = torch.topk(base_scores, k=topk, dim=-1).indices

        query_terms = [experts["visual_embed"]]
        if experts["token_logits"] is not None:
            query_terms.append(experts["token_embed"])
        if experts["qwen_logits"] is not None:
            query_terms.append(experts["qwen_embed"])
        query_embed = torch.stack(query_terms, dim=0).mean(dim=0)
        query_embed = F.normalize(query_embed.float(), dim=-1)

        candidate_proto = experts["text_embed"][topk_idx]
        query_expand = query_embed.unsqueeze(2).expand(-1, -1, topk, -1)
        visual_scores = torch.gather(experts["visual_logits"], 2, topk_idx)
        token_source = (
            experts["token_logits"]
            if experts["token_logits"] is not None
            else experts["visual_logits"].new_zeros(experts["visual_logits"].shape)
        )
        qwen_source = (
            experts["qwen_logits"]
            if experts["qwen_logits"] is not None
            else experts["visual_logits"].new_zeros(experts["visual_logits"].shape)
        )
        token_scores = torch.gather(token_source, 2, topk_idx)
        qwen_scores = torch.gather(qwen_source, 2, topk_idx)
        base_topk = torch.gather(base_scores, 2, topk_idx)

        candidate_input = torch.cat(
            [
                query_expand,
                candidate_proto,
                query_expand * candidate_proto,
                query_expand - candidate_proto,
                base_topk.unsqueeze(-1),
                visual_scores.unsqueeze(-1),
                token_scores.unsqueeze(-1),
                qwen_scores.unsqueeze(-1),
            ],
            dim=-1,
        )
        rerank_delta = self.lang_candidate_mlp(candidate_input.to(dtype=cls_feat.dtype)).squeeze(-1)
        rerank_delta = torch.tanh(rerank_delta - rerank_delta.mean(dim=-1, keepdim=True))

        confidence = self._combine_confidences(
            cls_token_confidence, qwen_visual_confidence, cls_feat.shape[-1], cls_feat.dtype, base_cls_logits
        )
        if confidence is not None:
            rerank_delta = rerank_delta * confidence.squeeze(-1).unsqueeze(-1)
        scale = cls_lang_scale
        if self.lang_logit_scale is not None:
            scale = scale * torch.clamp(self.lang_logit_scale, min=0.0, max=4.0)
        rerank_delta = rerank_delta.to(dtype=cls_feat.dtype) * scale

        conditioned_logits = base_scores.new_zeros(base_scores.shape, dtype=cls_feat.dtype)
        conditioned_logits.scatter_(2, topk_idx, rerank_delta)
        return conditioned_logits.permute(0, 2, 1)

    def _compute_cross_attention_logits(
        self,
        cls_feat,
        cls_token_features,
        class_prototypes,
        cls_token_confidence=None,
        qwen_visual_features=None,
        qwen_visual_confidence=None,
        cls_lang_scale=1.0,
        base_cls_logits=None,
    ):
        if class_prototypes is None or class_prototypes.dim() != 2:
            return None

        query = self.lang_query_proj(cls_feat).permute(0, 2, 1)
        query = F.normalize(query.float(), dim=-1).to(dtype=cls_feat.dtype)

        memory_parts = [
            F.normalize(
                self.lang_text_proj(class_prototypes.to(device=cls_feat.device, dtype=cls_feat.dtype)).float(), dim=-1
            ).to(dtype=cls_feat.dtype).unsqueeze(0).expand(query.shape[0], -1, -1)
        ]

        token_features = self._resize_time_condition(cls_token_features, cls_feat.shape[-1], cls_feat.dtype)
        if token_features is not None and token_features.shape[0] == cls_feat.shape[0]:
            token_memory = F.normalize(self.lang_token_proj(token_features).float(), dim=-1).to(dtype=cls_feat.dtype)
            memory_parts.append(token_memory)

        qwen_features = qwen_visual_features
        if qwen_features is not None and qwen_features.dim() == 3 and qwen_features.shape[0] == cls_feat.shape[0]:
            qwen_memory = F.normalize(
                self.lang_qwen_proj(qwen_features.to(device=cls_feat.device, dtype=cls_feat.dtype)).float(), dim=-1
            ).to(dtype=cls_feat.dtype)
            memory_parts.append(qwen_memory)

        memory = torch.cat(memory_parts, dim=1)
        attn_output, _ = self.lang_cross_attn(query, memory, memory, need_weights=False)
        fused = self.lang_cross_norm(query + attn_output)
        conditioned_logits = self.lang_cross_cls(fused)
        gate = torch.sigmoid(self.lang_cross_gate(torch.cat([query, attn_output], dim=-1))).permute(0, 2, 1)

        confidence = self._combine_confidences(
            cls_token_confidence, qwen_visual_confidence, cls_feat.shape[-1], cls_feat.dtype, base_cls_logits
        )
        if confidence is not None:
            gate = gate * confidence.permute(0, 2, 1)

        scale = cls_lang_scale
        if self.lang_logit_scale is not None:
            scale = scale * torch.clamp(self.lang_logit_scale, min=0.0, max=4.0)
        conditioned_logits = conditioned_logits.permute(0, 2, 1).to(dtype=cls_feat.dtype) * gate * scale
        candidate_mask = self._build_candidate_mask(base_cls_logits, conditioned_logits.dtype)
        if candidate_mask is not None:
            conditioned_logits = conditioned_logits * candidate_mask
        return conditioned_logits

    def _compute_language_conditioned_logits(
        self,
        cls_feat,
        cls_token_features,
        class_prototypes,
        cls_token_confidence=None,
        qwen_visual_features=None,
        qwen_visual_confidence=None,
        cls_lang_scale=1.0,
        base_cls_logits=None,
    ):
        if not self.use_language_conditioned_classifier:
            return None
        if self.lang_mode == "cross_attention":
            return self._compute_cross_attention_logits(
                cls_feat,
                cls_token_features,
                class_prototypes,
                cls_token_confidence,
                qwen_visual_features,
                qwen_visual_confidence,
                cls_lang_scale,
                base_cls_logits,
            )
        if self.lang_mode == "expert_router":
            return self._compute_expert_router_logits(
                cls_feat,
                cls_token_features,
                class_prototypes,
                cls_token_confidence,
                qwen_visual_features,
                qwen_visual_confidence,
                cls_lang_scale,
                base_cls_logits,
            )
        if self.lang_mode == "candidate_reranker":
            return self._compute_candidate_reranker_logits(
                cls_feat,
                cls_token_features,
                class_prototypes,
                cls_token_confidence,
                qwen_visual_features,
                qwen_visual_confidence,
                cls_lang_scale,
                base_cls_logits,
            )
        return self._compute_similarity_conditioned_logits(
            cls_feat,
            cls_token_features,
            class_prototypes,
            cls_token_confidence,
            qwen_visual_features,
            qwen_visual_confidence,
            cls_lang_scale,
            base_cls_logits,
        )

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, **kwargs):
        branched = kwargs.get('branched',
                              False)  # if True, the input features are already seperated into cls and reg branches
        if branched:
            feat_list = list(feat_list)
            mask_list = list(mask_list)
        cls_logit_bias = kwargs.get('cls_logit_bias', None)
        class_prototypes = kwargs.get('class_prototypes', None)
        cls_token_features = kwargs.get('cls_token_features', None)
        cls_token_confidence = kwargs.get('cls_token_confidence', None)
        qwen_visual_features = kwargs.get('qwen_visual_features', None)
        qwen_visual_confidence = kwargs.get('qwen_visual_confidence', None)
        teacher_cls_targets = kwargs.get('teacher_cls_targets', None)
        cls_lang_scale = kwargs.get('cls_lang_scale', 1.0)
        cls_pred = []
        reg_pred = []
        distill_losses = []

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            B, C, T = feat.shape
            if branched:
                cls_feat = feat[:B // 2, :, :]
                reg_feat = feat[B // 2:, :, :]
                mask = mask[:B // 2]
                mask_list[l] = mask
                feat_list[l] = cls_feat
            else:
                cls_feat = feat
                reg_feat = feat

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)
                # prog_feat, mask = self.prog_convs[i](prog_feat, mask)

            base_cls_logits = self.cls_head(cls_feat)
            lang_logits = self._compute_language_conditioned_logits(
                cls_feat,
                cls_token_features,
                class_prototypes,
                cls_token_confidence,
                qwen_visual_features,
                qwen_visual_confidence,
                cls_lang_scale,
                base_cls_logits,
            )
            cls_pred_branch = base_cls_logits
            if lang_logits is not None:
                cls_pred_branch = cls_pred_branch + lang_logits
            bias = self._resize_cls_logit_bias(cls_logit_bias, cls_pred_branch.shape[-1], cls_pred_branch.dtype, l)
            if bias is not None:
                cls_pred_branch = cls_pred_branch + bias
            reg_pred_branch = F.relu(self.scale[l](self.reg_head(reg_feat)))

            if teacher_cls_targets is not None and self.distill_loss_weight > 0:
                teacher = self._resize_teacher_targets(teacher_cls_targets, cls_pred_branch.shape[-1], cls_pred_branch.dtype)
                if teacher is not None:
                    teacher_probs = torch.sigmoid(teacher.detach())
                    valid_mask = mask.unsqueeze(1).expand_as(cls_pred_branch).float()
                    distill_loss = F.binary_cross_entropy_with_logits(
                        cls_pred_branch, teacher_probs, reduction='none'
                    )
                    distill_losses.append((distill_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0))

            cls_pred.append(cls_pred_branch)
            reg_pred.append(reg_pred_branch)

        points = self.prior_generator(feat_list)

        losses = self.losses(cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels)
        if distill_losses:
            losses['loss_lang_distill'] = sum(distill_losses) / len(distill_losses) * self.distill_loss_weight
        return losses

    def adjust_conf_adaptive(self, scores, prog_pred):
        """
        根据 progression 动态调整分类置信度
        scores: [B,T,C] sigmoid 后的值
        prog_pred: [B,T,1] 取值 [0,1]
        phase_pref: dict[class_id -> preferred_phase]
        """
        k = 5.0
        a = 0.3
        B, T, C = scores.shape
        pref = self.phase_pref_weight.view(1, 1, C)
        # prog_pred: [B,T,1] -> expand to [B,T,C]
        prog_broadcast = prog_pred.expand(-1, -1, C)
        # gaussian weight
        gaussian_w = torch.exp(-k * (prog_broadcast - pref) ** 2)  # [B,T,C]
        # map gaussian to [-1,1] then scale
        w = 1.0 + a * (2.0 * gaussian_w - 1.0)
        # optional clipping (安全)
        clip_min = 1.0 - abs(a)
        clip_max = 1.0 + abs(a)
        w = w.clamp(min=clip_min, max=clip_max)
        scores = scores * w
        scores = scores.clamp(min=1e-6, max=1.0 - 1e-6)
        return scores

    def forward_test(self, feat_list, mask_list, **kwargs):
        branched = kwargs.get('branched',
                              False)  # if True, the input features are already seperated into cls and reg branches
        if branched:
            feat_list = list(feat_list)
            mask_list = list(mask_list)
        cls_logit_bias = kwargs.get('cls_logit_bias', None)
        class_prototypes = kwargs.get('class_prototypes', None)
        cls_token_features = kwargs.get('cls_token_features', None)
        cls_token_confidence = kwargs.get('cls_token_confidence', None)
        qwen_visual_features = kwargs.get('qwen_visual_features', None)
        qwen_visual_confidence = kwargs.get('qwen_visual_confidence', None)
        cls_lang_scale = kwargs.get('cls_lang_scale', 1.0)
        cls_pred = []
        reg_pred = []

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            B, C, T = feat.shape
            if branched:
                cls_feat = feat[:B // 2, :, :]
                reg_feat = feat[B // 2:, :, :]
                mask = mask[:1]
                mask_list[l] = mask
                feat_list[l] = cls_feat
            else:
                cls_feat = feat
                reg_feat = feat

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)
                # prog_feat, mask = self.prog_convs[i](prog_feat, mask)

            base_cls_logits = self.cls_head(cls_feat)
            lang_logits = self._compute_language_conditioned_logits(
                cls_feat,
                cls_token_features,
                class_prototypes,
                cls_token_confidence,
                qwen_visual_features,
                qwen_visual_confidence,
                cls_lang_scale,
                base_cls_logits,
            )
            cls_pred_branch = base_cls_logits
            if lang_logits is not None:
                cls_pred_branch = cls_pred_branch + lang_logits
            bias = self._resize_cls_logit_bias(cls_logit_bias, cls_pred_branch.shape[-1], cls_pred_branch.dtype, l)
            if bias is not None:
                cls_pred_branch = cls_pred_branch + bias
            reg_pred_branch = F.relu(self.scale[l](self.reg_head(reg_feat)))

            cls_pred.append(cls_pred_branch)
            reg_pred.append(reg_pred_branch)

        points = self.prior_generator(feat_list)

        # get refined proposals and scores
        proposals, scores = self.get_valid_proposals_scores(points, reg_pred, cls_pred, mask_list)  # list [T,2]
        return proposals, scores

    def get_refined_proposals(self, points, reg_pred):
        points = torch.cat(points, dim=0)  # [T,4]
        reg_pred = torch.cat(reg_pred, dim=-1).permute(0, 2, 1)  # [B,T,2]

        start = points[:, 0][None] - reg_pred[:, :, 0] * points[:, 3][None]
        end = points[:, 0][None] + reg_pred[:, :, 1] * points[:, 3][None]
        proposals = torch.stack((start, end), dim=-1)  # [B,T,2]
        return proposals

    def get_valid_proposals_scores(self, points, reg_pred, cls_pred, mask_list):
        # apply regression to get refined proposals
        proposals = self.get_refined_proposals(points, reg_pred)  # [B,T,2]
        # proposal scores
        scores = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).sigmoid()  # [B,T,num_classes]
        # mask out invalid, and return a list with batch size
        masks = torch.cat(mask_list, dim=1)  # [B,T]

        new_proposals, new_scores = [], []
        for proposal, score, mask in zip(proposals, scores, masks):
            new_proposals.append(proposal[mask])  # [T,2]
            new_scores.append(score[mask])  # [T,num_classes]
        return new_proposals, new_scores

    def losses(self, cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels):
        gt_cls, gt_reg = self.prepare_targets(points, gt_segments, gt_labels)

        # positive mask
        gt_cls = torch.stack(gt_cls)
        valid_mask = torch.cat(mask_list, dim=1)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        num_pos = pos_mask.sum().item()

        # count the frame num for each sample
        frame_num = torch.sum(valid_mask, dim=1).cpu().tolist()
        pos_num = torch.sum(pos_mask, dim=1).cpu().tolist()

        # maintain an EMA of foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        if self.training:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                    1 - self.loss_normalizer_momentum
            ) * max(num_pos, 1)
            loss_normalizer = self.loss_normalizer
        else:
            loss_normalizer = max(num_pos, 1)

        # 1. classification loss
        cls_pred = [x.permute(0, 2, 1) for x in cls_pred]
        cls_pred = torch.cat(cls_pred, dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]

        # 2. regression using IoU/GIoU/DIOU loss (defined on positive samples)
        split_size = [reg.shape[-1] for reg in reg_pred]
        gt_reg = torch.stack(gt_reg).permute(0, 2, 1).split(split_size, dim=-1)  # [B,2,T]
        pred_segments = self.get_refined_proposals(points, reg_pred)[pos_mask]
        gtgt_segments = self.get_refined_proposals(points, gt_reg)[pos_mask]

        # optional label smoothing
        gt_target *= 1 - self.label_smoothing
        gt_target += self.label_smoothing / (self.num_classes + 1)

        losses = {}
        st_cls = 0
        st_reg = 0
        for i in range(len(frame_num)):
            cls_loss = self.cls_loss(cls_pred[st_cls:st_cls + frame_num[i], :],
                                     gt_target[st_cls:st_cls + frame_num[i], :],
                                     reduction="sum")
            cls_loss /= loss_normalizer
            losses[f'cls_sample_{i}_loss'] = cls_loss
            if num_pos == 0:
                losses[f'reg_sample_{i}_loss'] = pred_segments.sum() * 0
            else:
                losses[f'reg_sample_{i}_loss'] = self.reg_loss(pred_segments[st_reg:st_reg + pos_num[i], :],
                                                               gtgt_segments[st_reg:st_reg + pos_num[i], :],
                                                               reduction="sum")

            losses[f'reg_sample_{i}_loss']/=loss_normalizer
            st_cls += frame_num[i]
            st_reg += pos_num[i]
        return losses

        #         if num_pos == 0:
        #             reg_loss = pred_segments.sum() * 0
        #         else:
        #             # giou loss defined on positive samples
        #             reg_loss = self.reg_loss(pred_segments, gtgt_segments, reduction="sum")
        #         reg_loss /= loss_normalizer
        #
        #         # # 3. progression loss
        #         # diff = [prog_pred_[:, :, 1:] - prog_pred_[:, :, :-1] for prog_pred_ in prog_pred]
        #         # prog_pred = [prog_pred_.permute(0, 2, 1) for prog_pred_ in prog_pred]
        #         # prog_pred_inside = torch.cat(prog_pred, dim=1)[inside_mask]
        #         # gt_prog = torch.stack(gt_prog)[inside_mask]
        #         # prog_pred_outside= torch.cat(prog_pred, dim=1)[~inside_mask]
        #         # # consistency loss is to encourage progression scores outside gt actions to be close to 0
        #         # consistency_loss = self.cls_loss(prog_pred_outside, torch.zeros_like(prog_pred_outside), reduction="sum")
        #         # consistency_loss /= loss_normalizer
        #         # # [increase_loss[i].sum() for i in range(len(diff))]
        #         # # [decrease_loss[i].sum() for i in range(len(diff))]
        #         # if num_pos == 0:
        #         #     prog_loss = prog_pred_inside.sum() * 0
        #         # else:
        #         #     prog_loss = F.mse_loss(prog_pred_inside, gt_prog, reduction="sum")
        #         #     prog_loss /= loss_normalizer
        #
        #         if self.loss_weight > 0:
        #             loss_weight = self.loss_weight
        #         else:
        #             loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)
        #
        # return {
        #     "cls_loss": cls_loss,
        #     "reg_loss": reg_loss * loss_weight,
        #     # "reg_loss": reg_loss * loss_weight * 2,
        #     # "prog_loss": prog_loss * 0.4,
        #     # 'consistency_loss': consistency_loss *0.01* loss_weight
        # }

    @torch.no_grad()
    def prepare_targets(self, points, gt_segments, gt_labels):
        concat_points = torch.cat(points, dim=0)
        num_pts = concat_points.shape[0]
        gt_cls, gt_reg = [], []
        # gt_prog = []
        # inside_masks = []

        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            num_gts = gt_segment.shape[0]

            # corner case where current sample does not have actions
            if num_gts == 0:
                gt_cls.append(gt_segment.new_full((num_pts, self.num_classes), 0))
                gt_reg.append(gt_segment.new_zeros((num_pts, 2)))
                # gt_prog.append(gt_segment.new_zeros((num_pts, 1)))
                continue

            # compute the lengths of all segments -> F T x N
            lens = gt_segment[:, 1] - gt_segment[:, 0]
            lens = lens[None, :].repeat(num_pts, 1)

            # compute the distance of every point to each segment boundary
            # auto broadcasting for all reg target-> F T x N x2
            gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
            left = concat_points[:, 0, None] - gt_segs[:, :, 0]
            right = gt_segs[:, :, 1] - concat_points[:, 0, None]
            reg_targets = torch.stack((left, right), dim=-1)

            # # ---------- progression 计算 ----------
            # t = concat_points[:, 0, None]  # [num_pts, 1]
            # start = gt_segs[:, :, 0]
            # end = gt_segs[:, :, 1]
            # mid = (start + end) / 2.0
            # length = (end - start).clamp(min=1e-6)
            #
            # norm_t =  (t - mid) / (length / 2.0)
            # prog_targets=tan_curve(norm_t, alpha=0.8)
            # # norm_t = (t - start) / length
            # # prog_targets = gaussian_cdf(norm_t,std=0.3)
            #
            # # mask 掉非动作区域（t不在[start, end]范围内的点）
            # inside_mask = (t >= start) & (t <= end)
            # prog_targets = prog_targets * inside_mask.float()
            # inside_mask = inside_mask.any(dim=-1, keepdim=True)  # [B, T, 1]
            #
            # # 可选：若一个点属于多个动作，取最大值或平均值
            # prog_targets, _ = prog_targets.max(dim=1, keepdim=True)
            # gt_prog.append(prog_targets)
            # inside_masks.append(inside_mask)

            if self.center_sample == "radius":
                # center of all segments F T x N
                center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
                # center sampling based on stride radius
                # compute the new boundaries:
                # concat_points[:, 3] stores the stride
                t_mins = center_pts - concat_points[:, 3, None] * self.center_sample_radius
                t_maxs = center_pts + concat_points[:, 3, None] * self.center_sample_radius
                # prevent t_mins / maxs from over-running the action boundary
                # left: torch.maximum(t_mins, gt_segs[:, :, 0])
                # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
                # F T x N (distance to the new boundary)
                cb_dist_left = concat_points[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
                cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
                # F T x N x 2
                center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
                # F T x N
                inside_gt_seg_mask = center_seg.min(-1)[0] > 0
            else:
                # inside an gt action
                inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

            # limit the regression range for each location
            max_regress_distance = reg_targets.max(-1)[0]
            # F T x N
            inside_regress_range = torch.logical_and(
                (max_regress_distance >= concat_points[:, 1, None]), (max_regress_distance <= concat_points[:, 2, None])
            )

            # if there are still more than one actions for one moment
            # pick the one with the shortest duration (easiest to regress)
            lens.masked_fill_(inside_gt_seg_mask == 0, float("inf"))
            lens.masked_fill_(inside_regress_range == 0, float("inf"))
            # F T x N -> F T
            min_len, min_len_inds = lens.min(dim=1)

            # corner case: multiple actions with very similar durations (e.g., THUMOS14)
            if self.filter_similar_gt:
                min_len_mask = torch.logical_and((lens <= (min_len[:, None] + 1e-3)), (lens < float("inf")))
            else:
                min_len_mask = lens < float("inf")
            min_len_mask = min_len_mask.to(reg_targets.dtype)

            # cls_targets: F T x C; reg_targets F T x 2
            gt_label_one_hot = F.one_hot(gt_label.long(), self.num_classes).to(reg_targets.dtype)
            cls_targets = min_len_mask @ gt_label_one_hot
            # to prevent multiple GT actions with the same label and boundaries
            cls_targets.clamp_(min=0.0, max=1.0)
            # OK to use min_len_inds
            reg_targets = reg_targets[range(num_pts), min_len_inds]
            # normalization based on stride
            reg_targets /= concat_points[:, 3, None]

            gt_cls.append(cls_targets)
            gt_reg.append(reg_targets)
        # return gt_cls, gt_reg,  torch.stack(inside_masks, dim=0)
        return gt_cls, gt_reg

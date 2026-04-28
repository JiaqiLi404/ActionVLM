import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.modules.module import T

from ..builder import DETECTORS, build_backbone, build_projection, build_head, build_language_model
from ..bricks import Scale, AffineDropPath
from .base import BaseDetector
from ..utils.post_processing import batched_nms, convert_to_seconds

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@DETECTORS.register_module()
class VLLM_Detector(BaseDetector):
    """
        Base class for single-stage detectors which should not have roi_extractors.
        """

    def __init__(self, backbone=None, projection=None, language_model=None, rpn_head=None,
                 pipeline=None, base_prompt=None, llm_proposal_order='timestamp', **kwargs):
        super(VLLM_Detector, self).__init__(**kwargs)

        if pipeline is None:
            pipeline = ['backbone', 'projection', 'language_model', 'rpn_head']
        self.pipeline = pipeline

        if backbone is not None:
            self.backbone = build_backbone(backbone, with_wrapper=backbone.get('with_wrapper', False))

        if projection is not None:
            self.projection = build_projection(projection)

        if language_model is not None:
            self.language_model = build_language_model(language_model)

        if rpn_head is not None:
            if self.with_language_model and hasattr(self.language_model, 'process_rpn_head'):
                rpn_head = self.language_model.process_rpn_head(rpn_head)
            self.rpn_head = build_head(rpn_head)

        self.llm_proposal_order = llm_proposal_order
        if self.llm_proposal_order not in ['certainty', 'timestamp']:
            raise ValueError(
                f"llm_proposal_order should be either 'certainty' or 'start', got {self.llm_proposal_order}")

        self.base_prompt = base_prompt
        self._vision_loss_sums = None
        self._vision_loss_counts = None
        self._vision_loss_avg = None
        self._vision_loss_epoch = None

    def eval(self: T) -> T:
        super().eval()
        if self.with_language_model:
            self.language_model.eval()
        return self

    @property
    def with_backbone(self):
        """bool: whether the detector has backbone"""
        return hasattr(self, "backbone") and self.backbone is not None

    @property
    def with_projection(self):
        """bool: whether the detector has projection"""
        return hasattr(self, "projection") and self.projection is not None

    @property
    def with_language_model(self):
        """bool: whether the detector has language_model"""
        """
        Some special attributes:
        - lora_enable: whether to enable lora
        - tune_llm: whether to tune llm
        - tune_mm_mlp_adapter: whether to tune mm_mlp_adapter
        - lora_bias: lora bias
        - model: peft model
        """
        return hasattr(self, "language_model") and self.language_model is not None

    @property
    def with_rpn_head(self):
        """bool: whether the detector has localization head"""
        return hasattr(self, "rpn_head") and self.rpn_head is not None

    def get_prompt(self, metas):
        actions_str = ', '.join(metas[0]['class_map'])
        if self.base_prompt is not None:
            self.base_prompt = self.base_prompt.replace('<actions>', actions_str)
            return self.base_prompt

        if self.llm_proposal_order == 'certainty':
            prompt_example = f"{metas[0]['class_map'][0]}: 0-10, {metas[0]['class_map'][-1]}: 21-36, {metas[0]['class_map'][0]}: 16-25."
        else:
            prompt_example = f"{metas[0]['class_map'][0]}: 0-10, {metas[0]['class_map'][-1]}: 16-25, {metas[0]['class_map'][0]}: 21-36."
        prompt_base = ("<video>\n " +
                       f"List all potential action boundaries in order of {self.llm_proposal_order}, following the template below: \n" +
                       "[Action1]: [Start1]-[End1], [Action2]: [Start2] - [End2], ...\n" +
                       "Here are the possible actions you can choose from: \n" + actions_str +
                       f"\nFor example: {prompt_example}\n" +
                       "If no actions are present, do not generate any text.")
        return prompt_base

    def to_cuda(self, x, masks, gt_segments=None, gt_labels=None, device='cuda'):
        x = x.float().cuda()
        masks = masks.cuda()
        if gt_segments is not None:
            gt_segments = [g.to(device) for g in gt_segments]
        if gt_labels is not None:
            gt_labels = [g.to(device) for g in gt_labels]
        return x, masks, gt_segments, gt_labels

    def forward_train(self, x, masks, metas, gt_segments, gt_labels, curr_epoch, **kwargs):
        x, masks, gt_segments, gt_labels = self.to_cuda(x, masks, gt_segments, gt_labels)
        losses = {}
        prompt_base = self.get_prompt(metas)
        self._rotate_vision_loss_stats(curr_epoch, metas, x.device)
        pass_dict = {
            'x': x,
            'masks': masks,
            'gt_segments': gt_segments,
            'gt_labels': gt_labels,
            'metas': metas,
            'curr_epoch': curr_epoch,
            'prompt_base': prompt_base,
            'llm_proposal_order': self.llm_proposal_order,
        }
        pass_dict.update(kwargs)
        del x
        torch.cuda.empty_cache()
        stage = 0
        rpn_total_cost = None

        pipeline = self.pipeline.copy()
        while len(pipeline) > 0:
            module = pipeline.pop(0)
            if module == 'backbone' and self.with_backbone:
                pass_dict.update({'stage': stage})
                pass_dict = self.backbone(**pass_dict)
                stage += 1
            elif module == 'projection' and self.with_projection:
                pass_dict.update({'stage': stage})
                pass_dict = self.projection(**pass_dict)
                stage += 1
            elif module == 'language_model' and self.with_language_model:
                self.language_model.train()
                pass_dict.update({'stage': stage})
                pass_dict = self.language_model.forward_train(**pass_dict)
                language_loss = {
                    k: v
                    for k, v in pass_dict.items()
                    if k in {'llm_loss', 'semantic_loss', 'prototype_align_loss'} and v is not None
                }
                losses.update(language_loss)
                stage += 1
            elif module == 'rpn_head' and self.with_rpn_head:
                self.rpn_head.train()
                pass_dict.update({'stage': stage, "return_all": True})
                pass_dict: dict = self.rpn_head.forward_train(**pass_dict)
                rpn_total_cost = pass_dict.get("cost", None)
                rpn_loss = {k: v for k, v in pass_dict.items() if 'loss' in k and v is not None}
                losses.update(rpn_loss)
                stage += 1
            elif module in ['backbone', 'projection', 'language_model', 'rpn_head']:
                continue
            else:
                raise ValueError(f"Unknown module {module}")

        sample_losses = self._extract_sample_losses(losses)
        if pass_dict.get("vision_only_epoch", False) and pass_dict.get("collect_vision_stats", False):
            self._update_vision_loss_stats(sample_losses, gt_labels, masks.device)
        else:
            adv_loss = self._compute_adv_loss(pass_dict, sample_losses)
            if adv_loss is not None:
                losses["adv_loss"] = adv_loss * pass_dict.get("adv_loss_weight", 1.0)

        if len(losses) > 0 or rpn_total_cost is not None:
            total_cost = rpn_total_cost if rpn_total_cost is not None else torch.tensor(0.0, device=masks.device)
            for key, value in losses.items():
                if value is None:
                    continue
                if key.startswith("cls_sample_") or key.startswith("reg_sample_"):
                    if rpn_total_cost is None:
                        total_cost = total_cost + value
                    continue
                total_cost = total_cost + value
            losses["cost"] = total_cost
        return losses

    def forward_test(self, x, masks, metas=None, infer_cfg=None, **kwargs):
        x, masks, _, _ = self.to_cuda(x, masks)
        prompt_base = self.get_prompt(metas)
        pass_dict = {
            'x': x,
            'masks': masks,
            'metas': metas,
            'prompt_base': prompt_base,
            'infer_cfg': infer_cfg,
            'llm_proposal_order': self.llm_proposal_order,
            'return_dict': True,
        }
        pass_dict.update(kwargs)
        del x
        torch.cuda.empty_cache()
        stage = 0

        pipeline = self.pipeline.copy()
        while len(pipeline) > 0:
            module = pipeline.pop(0)
            if module == 'backbone' and self.with_backbone:
                pass_dict.update({'stage': stage})
                pass_dict = self.backbone(**pass_dict)
                stage += 1
            elif module == 'projection' and self.with_projection:
                pass_dict.update({'stage': stage})
                pass_dict = self.projection(**pass_dict)
                stage += 1
            elif module == 'language_model' and self.with_language_model:
                self.language_model.eval()
                pass_dict.update({'stage': stage})
                pass_dict = self.language_model.forward_test(**pass_dict)
                stage += 1
            elif module == 'rpn_head' and self.with_rpn_head:
                self.rpn_head.eval()
                pass_dict.update({'stage': stage})
                pass_dict: dict = self.rpn_head.forward_test(**pass_dict)
                stage += 1
            elif module in ['backbone', 'projection', 'language_model', 'rpn_head']:
                continue
            else:
                raise ValueError(f"Unknown module {module}")

        scores = pass_dict.get('scores', None)
        proposals = pass_dict.get('proposals', None)
        actions = pass_dict.get('actions', None)

        predictions = scores, proposals, actions
        return predictions

    def _ensure_vision_loss_stats(self, num_classes, device):
        if self._vision_loss_sums is not None and self._vision_loss_sums.numel() == num_classes:
            return
        self._vision_loss_sums = torch.zeros(num_classes, device=device)
        self._vision_loss_counts = torch.zeros(num_classes, device=device)
        self._vision_loss_avg = torch.zeros(num_classes, device=device)

    def _rotate_vision_loss_stats(self, curr_epoch, metas, device):
        if not self.with_language_model or not hasattr(self.language_model, "is_vision_only_epoch"):
            return
        num_classes = len(metas[0]["class_map"])
        self._ensure_vision_loss_stats(num_classes, device)
        if self._vision_loss_epoch is None:
            self._vision_loss_epoch = curr_epoch
            if self.language_model.is_vision_only_epoch(curr_epoch):
                self._vision_loss_sums.zero_()
                self._vision_loss_counts.zero_()
            return
        if curr_epoch == self._vision_loss_epoch:
            return
        if self.language_model.is_vision_only_epoch(self._vision_loss_epoch):
            self._finalize_vision_loss_stats()
        if self.language_model.is_vision_only_epoch(curr_epoch):
            self._vision_loss_sums.zero_()
            self._vision_loss_counts.zero_()
        self._vision_loss_epoch = curr_epoch

    def _finalize_vision_loss_stats(self):
        if self._vision_loss_sums is None:
            return
        sums = self._vision_loss_sums.clone()
        counts = self._vision_loss_counts.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums)
            dist.all_reduce(counts)
        valid = counts > 0
        if valid.any():
            self._vision_loss_avg[valid] = sums[valid] / counts[valid]

    def _extract_sample_losses(self, losses):
        sample_losses = []
        sample_idx = 0
        while True:
            cls_key = f"cls_sample_{sample_idx}_loss"
            reg_key = f"reg_sample_{sample_idx}_loss"
            if cls_key not in losses and reg_key not in losses:
                break
            total = None
            if cls_key in losses:
                total = losses[cls_key]
            if reg_key in losses:
                total = losses[reg_key] if total is None else total + losses[reg_key]
            sample_losses.append(total)
            sample_idx += 1
        return sample_losses

    def _update_vision_loss_stats(self, sample_losses, gt_labels, device):
        if self._vision_loss_avg is None:
            return
        for sample_loss, sample_labels in zip(sample_losses, gt_labels):
            if sample_loss is None or sample_labels.numel() == 0:
                continue
            unique_labels = torch.unique(sample_labels.long())
            for label in unique_labels:
                self._vision_loss_sums[label] += sample_loss.detach()
                self._vision_loss_counts[label] += 1

    def _compute_adv_loss(self, pass_dict, sample_losses):
        adv_predictions = pass_dict.get("adv_predictions", None)
        if adv_predictions is None or self._vision_loss_avg is None:
            return None

        input_masks = pass_dict.get("input_masks", pass_dict["masks"]).bool()
        gt_segments = pass_dict["gt_segments"]
        gt_labels = pass_dict["gt_labels"]
        target = torch.zeros_like(adv_predictions)
        _, time_steps = adv_predictions.shape

        for batch_idx, sample_labels in enumerate(gt_labels):
            if batch_idx >= len(sample_losses) or sample_losses[batch_idx] is None:
                continue
            sample_loss = sample_losses[batch_idx].detach()
            if sample_labels.numel() == 0:
                continue
            for seg, label in zip(gt_segments[batch_idx], sample_labels.long()):
                class_loss = self._vision_loss_avg[label]
                advantage = class_loss - sample_loss
                start = max(int(torch.floor(seg[0]).item()), 0)
                end = min(int(torch.ceil(seg[1]).item()), time_steps)
                if end <= start:
                    end = min(start + 1, time_steps)
                target[batch_idx, start:end] = advantage

        if not input_masks.any():
            return adv_predictions.sum() * 0.0
        return torch.mean((adv_predictions[input_masks] - target[input_masks]) ** 2)

    def checkpoint_config(self):
        lora = getattr(self.language_model, "lora_enable", False)
        full_llm = not lora and getattr(self.language_model, "tune_llm", True)
        skip_module = []
        if not full_llm:
            skip_module.append('language_model')
        specific_module = []
        if getattr(self.language_model, "tune_mm_mlp_adapter", False):
            specific_module.append('mm_projector')
        return {
            "lora": lora,
            'skip_module': skip_module,
            'specific_module': specific_module,
            'lora_bias': getattr(self.language_model, "lora_bias", 'none'),
            'config': self.language_model.model.config,
            'peft_model': self.language_model.model,
            'tokenizer': self.language_model.tokenizer,
        }

    def load_checkpoint(self, **kwargs):
        peft_model = kwargs.get('peft_model', None)
        if peft_model is not None and self.with_language_model:
            self.language_model.model = peft_model
        config = kwargs.get('config', None)
        if config is not None and self.with_language_model:
            self.language_model.model.config = config
        tokenizer = kwargs.get('tokenizer', None)
        if tokenizer is not None and self.with_language_model:
            self.language_model.tokenizer = tokenizer

    @torch.no_grad()
    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        if self.with_rpn_head and hasattr(self.rpn_head, 'post_processing'):
            rpn_scores, rpn_proposals, rpn_actions = predictions
            predictions = rpn_proposals, rpn_scores
            results = self.rpn_head.post_processing(predictions, metas, post_cfg, ext_cls, **kwargs)
            return results

        rpn_scores, rpn_proposals, rpn_actions = predictions
        # rpn_proposals,  # [B,K,2]
        # rpn_scores,  # [B,K] after sigmoid
        results = {}
        for i in range(len(metas)):  # processing each video
            segments = rpn_proposals[i].detach().cpu()  # [N,2]
            scores = rpn_scores[i].detach().cpu()  # [N,class]
            video_id = metas[i]["video_name"]

            # convert segments to seconds
            segments = convert_to_seconds(segments, metas[i])

            # merge with external classifier
            if isinstance(ext_cls, list):  # own classification results
                labels = rpn_actions[i]
                ext_cls_set = set(ext_cls)
            else:
                segments, labels, scores, ext_cls_set = ext_cls(video_id, segments, scores)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                if label in ext_cls_set:
                    # convert to python scalars
                    results_per_video.append(
                        dict(
                            segment=[round(seg.item(), 2) for seg in segment],
                            label=label,
                            score=round(score.item(), 4),
                        )
                    )

            if video_id in results.keys():
                results[video_id].extend(results_per_video)
            else:
                results[video_id] = results_per_video

        return results

    def get_optim_groups(self, cfg, exclude=['backbone']):
        # separate out all parameters that with / without weight decay
        # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
        exclude = tuple(exclude)
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d)
        blacklist_weight_modules = (nn.LayerNorm, nn.GroupNorm)
        whitelist_weight_names = ("proj", "encoder", "decoder")
        blacklist_weight_names = ("norm", "token", 'embed', 'embedding', 'ls')

        # loop over all modules / params
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if not p.requires_grad or fpn.startswith(exclude):
                    continue

                has_white_list = any([wn in fpn for wn in whitelist_weight_names])
                has_black_list = any([bn in fpn for bn in blacklist_weight_names])
                has_black_module = isinstance(m, blacklist_weight_modules)
                has_white_module = isinstance(m, whitelist_weight_modules)

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and (has_white_module or (has_white_list and not has_black_list)):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                    if fpn in no_decay:
                        no_decay.remove(fpn)
                elif pn.endswith("weight") and (has_black_module or has_black_list) and fpn not in decay:
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
                    # corner case of our scale layer
                    no_decay.add(fpn)
                elif pn.endswith("rel_pe") or pn.endswith("gaussian_params"):
                    # corner case for relative position encoding
                    no_decay.add(fpn)
                elif (
                        pn.endswith("A_log")
                        or pn.endswith("D_b")
                        or pn.endswith("D")
                        or pn.endswith("A_b_log")
                        or pn.endswith("forward_embed")
                        or pn.endswith("backward_embed")
                        or pn.endswith("params")
                ):
                    # corner case for mamba
                    decay.add(fpn)
                    continue

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if not p.requires_grad or fpn.startswith(exclude):
                    continue
                if fpn not in decay and fpn not in no_decay:
                    print(f"Parameter {fpn} is not separated into either decay/no_decay set!")
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if not pn.startswith(exclude) and p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": cfg["weight_decay"],
                "lr": cfg["lr"],
            },
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0, "lr": cfg["lr"]},
        ]

        return optim_groups

import torch
from torch.nn import functional as F
import json
import math
import numpy as np
from ..builder import DETECTORS, build_backbone, build_projection, build_head, build_neck
from .base import BaseDetector
from ..utils.functions import gaussian_cdf,tan_curve
from ..utils.post_processing import batched_nms, convert_to_seconds


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """
    Base class for single-stage detectors which should not have roi_extractors.
    """

    def __init__(self, backbone=None, projection=None, neck=None, rpn_head=None):
        super(SingleStageDetector, self).__init__()

        if backbone is not None:
            self.backbone = build_backbone(backbone)

        if projection is not None:
            self.projection = build_projection(projection)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = build_head(rpn_head)

        # self.results = []
        # self.result_tensors = {}
        # self.states = {"epoch": 0}
        # gt_path = "/mnt/f/Projects/ActionReasoner/data/thumos-14/annotations/thumos_14_anno.json"
        # self.gt_data = {}
        # with open(gt_path, 'r') as f:
        #     gt_json = json.load(f)
        #     self.gt_data = gt_json['database']

    @property
    def with_backbone(self):
        """bool: whether the detector has backbone"""
        return hasattr(self, "backbone") and self.backbone is not None

    @property
    def with_projection(self):
        """bool: whether the detector has projection"""
        return hasattr(self, "projection") and self.projection is not None

    @property
    def with_neck(self):
        """bool: whether the detector has neck"""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_rpn_head(self):
        """bool: whether the detector has localization head"""
        return hasattr(self, "rpn_head") and self.rpn_head is not None

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        losses = dict()
        if self.with_backbone:
            x = self.backbone(inputs, masks)
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        if self.with_rpn_head:
            rpn_losses = self.rpn_head.forward_train(
                x,
                masks,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                **kwargs,
            )
            losses.update(rpn_losses)

        # only key has loss will be record
        losses["cost"] = sum(_value for _key, _value in losses.items())
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        if self.with_backbone:
            x = self.backbone(inputs, masks)
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        if self.with_rpn_head:
            predictions = self.rpn_head.forward_test(x, masks)
        else:
            predictions = None, None
        return predictions

    @torch.no_grad()
    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        # rpn_proposals,  # [B,K,2]
        # rpn_scores,  # [B,K,num_classes] after sigmoid
        # progression  # [B,1] optional
        rpn_proposals, rpn_scores = predictions[0], predictions[1]
        progressions = predictions[2] if len(predictions) == 3 else None
        progressions = progressions[0].squeeze(1) if progressions is not None else None

        pre_nms_thresh = getattr(post_cfg, "pre_nms_thresh", 0.001)
        pre_nms_topk = getattr(post_cfg, "pre_nms_topk", 2000)
        num_classes = rpn_scores[0].shape[-1]

        results = {}
        for i in range(len(metas)):  # processing each video
            segments = rpn_proposals[i].detach().cpu()  # [N,2]
            scores = rpn_scores[i].detach().cpu()  # [N,class]

            if num_classes == 1:
                scores = scores.squeeze(-1)
                labels = torch.zeros(scores.shape[0]).contiguous()
                pt_idxs = None
            else:
                pred_prob = scores.flatten()  # [N*class]

                # Apply filtering to make NMS faster following detectron2
                # 1. Keep seg with confidence score > a threshold
                keep_idxs1 = pred_prob > pre_nms_thresh
                pred_prob = pred_prob[keep_idxs1]
                topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

                # 2. Keep top k top scoring boxes only
                num_topk = min(pre_nms_topk, topk_idxs.size(0))
                pred_prob, idxs = pred_prob.sort(descending=True)
                pred_prob = pred_prob[:num_topk].clone()
                topk_idxs = topk_idxs[idxs[:num_topk]].clone()

                # 3. gather predicted proposals
                pt_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                cls_idxs = torch.fmod(topk_idxs, num_classes)

                segments = segments[pt_idxs]
                scores = pred_prob
                labels = cls_idxs

            # if not sliding window, do nms
            if post_cfg.sliding_window == False and post_cfg.nms is not None:
                segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            video_id = metas[i]["video_name"]

            # convert segments to seconds
            segments_ori = segments.clone()
            segments = convert_to_seconds(segments, metas[i])

            # merge with external classifier
            if isinstance(ext_cls, list):  # own classification results
                labels = [ext_cls[label.item()] for label in labels]
            else:
                segments, labels, scores = ext_cls(video_id, segments, scores)

            results_per_video = []
            for seg_i, (seg_ori, segment, label, score) in enumerate(zip(segments_ori, segments, labels, scores)):
                score = score.item()
                prog_segment=(-1.0,-1.0)
                # if progressions is not None:
                #     progression = progressions[i]
                #     start,end= float(seg_ori[0]), float(seg_ori[1])
                #     start_index, end_index = math.ceil(start), math.floor(end)
                #     length = max(end - start, 1)
                #     mid = (start + end) / 2.0
                #     prog_in_seg = progression[start_index:end_index+1]
                #     if len(prog_in_seg) == 0:
                #         continue
                #
                #     if len(prog_in_seg) > 1:
                #         # 1. 散度
                #         divergence = (prog_in_seg.max() - prog_in_seg.min()).item()
                #         # 2. 单调性
                #         diff = prog_in_seg[1:] - prog_in_seg[:-1]
                #         monotonicity = (diff > 0).float().mean().item()
                #
                #         # 使用与训练相同的参数生成理想 progression 曲线
                #         # norm_t = (torch.arange(start_index, end_index+1, device=progression.device) - start) / length
                #         # ideal_prog = gaussian_cdf(norm_t, std=0.3)
                #         norm_t = (torch.arange(start_index, end_index+1, device=progression.device) - mid) / (length / 2.0)
                #         ideal_prog = tan_curve(norm_t)
                #         # MSE 越小越好，因此我们取相似度 = 1 - MSE
                #         mse = F.mse_loss(prog_in_seg, ideal_prog, reduction='mean').item()
                #         consistency = max(0.0, 1.0 - mse)
                #     else:
                #         divergence = 0.5
                #         monotonicity = 0.5
                #         consistency = 0.5
                #
                #     # record the IoU and each scores
                #     gt = self.gt_data[video_id]['annotations']
                #     max_iou = 0.0
                #     for ann in gt:
                #         gt_start = ann['segment'][0]
                #         gt_end = ann['segment'][1]
                #         gt_cls = ann['label']
                #         if gt_cls != label:
                #             continue
                #         inter_start = max(gt_start, segment[0].item())
                #         inter_end = min(gt_end, segment[1].item())
                #         inter_len = max(0.0, inter_end - inter_start)
                #         union_len = (gt_end - gt_start) + (segment[1].item() - segment[0].item()) - inter_len
                #         iou = inter_len / union_len if union_len > 0 else 0.0
                #         if iou > max_iou:
                #             max_iou = iou
                #     self.results.append(dict(
                #         video_id=video_id,
                #         segment_original=[round(seg_ori[0].item(), 2), round(seg_ori[1].item(), 2)],
                #         segment=[round(segment[0].item(), 2), round(segment[1].item(), 2)],
                #         prog_segment=[round(prog_segment[0], 2), round(prog_segment[1], 2)],
                #         label=label,
                #         original_score=round(score, 4),
                #         divergence=round(divergence, 4),
                #         monotonicity=round(monotonicity, 4),
                #         consistency=round(consistency, 4),
                #         iou_with_gt=round(max_iou, 4),
                #         index=pt_idxs[seg_i].item() if pt_idxs is not None else -1,
                #     ))
                #     self.result_tensors[video_id] = {
                #         'rpn_scores': rpn_scores[i],
                #         'rpn_proposals': rpn_proposals[i],
                #         'progression': progressions[i] if progressions is not None else None,
                #     }

                # convert to python scalars
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=label,
                        score=round(score, 4),
                    )
                )

            if video_id in results.keys():
                results[video_id].extend(results_per_video)
            else:
                results[video_id] = results_per_video

        return results

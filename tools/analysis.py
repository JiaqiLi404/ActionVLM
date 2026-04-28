import json
import jsonlines
import copy
import numpy as np
import math
import torch
import scipy.stats as stats
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from tqdm import tqdm

from opentad.evaluations import build_evaluator


def compute_iou(gt, segment, label):
    max_iou = 0.0
    max_proposal = None
    min_close = 100
    mid = (segment[0] + segment[1]) / 2.0
    has_iou = False
    for ann in gt:
        gt_start = ann['segment'][0]
        gt_end = ann['segment'][1]
        gt_cls = ann['label']
        if gt_cls != label:
            continue
        inter_start = max(gt_start, segment[0])
        inter_end = min(gt_end, segment[1])
        inter_len = max(0.0, inter_end - inter_start)
        union_len = (gt_end - gt_start) + (segment[1] - segment[0]) - inter_len
        iou = inter_len / union_len if union_len > 0 else 0.0
        if iou > max_iou:
            max_iou = iou
            max_proposal = (gt_start, gt_end)
            has_iou = True
        if has_iou == False:
            close = abs((gt_start + gt_end) / 2.0 - mid)
            if close < min_close:
                min_close = close
                max_proposal = (gt_start, gt_end)
    return max_iou, max_proposal


def refine_proposal_by_fitting(prog_in_seg, win_start_index, win_additional_ratio, func_type='tan'):
    # predict the boundaries
    def estimate_segment_scipy(prog_pred, std=0.3):
        T = len(prog_pred)
        t = np.arange(T)
        prog_pred = prog_pred.detach().cpu().numpy()

        def model_gaussian(params):
            start, end = params
            norm_t = (t - start) / (end - start + 1e-6)
            cdf = 0.5 * (1 + torch.erf((torch.tensor(norm_t) - 0.5) / (std * (2 ** 0.5))))
            return np.clip(cdf.numpy(), 0, 1)

        def model_tan(params):
            start, end = params
            mid = (start + end) / 2.0
            length = end - start
            norm_t = (t - mid) / (length / 2.0 + 1e-6)
            cdf = (torch.tan(torch.tensor(norm_t)) / torch.tan(torch.tensor(0.8)) + 1) * 0.5
            return np.clip(cdf.numpy(), 0, 1)

        def residual(params):
            start, end = params
            if start >= end:
                return np.ones_like(prog_pred) * 1e3
            if func_type == 'gaussian':
                return prog_pred - model_gaussian(params)
            return prog_pred - model_tan(params)

        result = least_squares(residual, x0=[0, T], bounds=([-0.1 * T, 0.9 * T], [0.1 * T, 1.1 * T]))
        return result.x

    assert win_additional_ratio == 0, "Only support no context now."
    x = estimate_segment_scipy(prog_in_seg, std=0.3)
    prog_segment = [max(0, x[0] + win_start_index), x[1] + win_start_index]
    return prog_segment


def refine_proposal_by_constraints(prog_in_seg, win_start_index, ori_segment, win_additional_ratio):
    if len(prog_in_seg) <= 2 or len(prog_in_seg) <30:
        return ori_segment
    mid = round((len(prog_in_seg)) / 2.0)
    diff = prog_in_seg[1:] - prog_in_seg[:-1]
    # prog_smooth = torch.nn.functional.avg_pool1d(
    #     prog_in_seg[None, None, :], kernel_size=4, stride=1, padding=1
    # )[0, 0]
    max_idx = torch.argmax(prog_in_seg[mid:]).item() + mid
    max_val = max(prog_in_seg[mid:])
    # end index is the first index after max_idx where prog_smooth >0.5
    end_idx = max_idx
    for i in range(max_idx, len(prog_in_seg)):
        if prog_in_seg[i] >= max_val * 0.7:
            end_idx = i
        else:
            break
    start_idx = int(torch.argmax(diff[:end_idx]) + 1)
    # return [win_start_index + start_idx, ori_segment[1]]
    if win_start_index + end_idx <= ori_segment[0]:
        return [win_start_index + start_idx, win_start_index + end_idx]
    return [ori_segment[0], win_start_index + end_idx]


# set cwd
# os.chdir("../")
gt_path = "/mnt/f/Projects/ActionReasoner/data/thumos-14/annotations/thumos_14_anno.json"
analysis_path = "analysis_155691/predictions_epoch030.json"
tensor_path = analysis_path.replace("predictions", "tensors").replace(".json", ".pth")

with open(gt_path, 'r') as f:
    gt_data = json.load(f)['database']
# read analysis data (jsonlines)
pred_data = []
with jsonlines.open(analysis_path, mode='r') as reader:
    for obj in reader:
        pred_data.append(obj)
tensors = torch.load(tensor_path)
score_list, div_score_list, mono_score_list, consistency_score_list, iou_list = [], [], [], [], []
results, results_ori = {}, {}
imp_num, dec_num = 0, 0
for pred in tqdm(pred_data):
    score, div_score, mono_score, consistency_score, iou = (
        pred['original_score'], pred['divergence'], pred['monotonicity'],
        pred['consistency'], pred['iou_with_gt'])
    if score < 0.1:
        continue
    score_list.append(score)
    div_score_list.append(div_score)
    mono_score_list.append(mono_score)
    consistency_score_list.append(consistency_score)
    iou_list.append(iou)
    ori_score = score

    vid = pred['video_id']
    if vid not in results_ori:
        results_ori[vid] = []
    results_ori[vid].append(dict(segment=pred['segment'], label=pred['label'], score=ori_score))

    prog_in_seg = tensors[vid]['progression']
    start, end = float(max(0, pred['segment_original'][0])), float(pred['segment_original'][1])
    start_index, end_index = math.ceil(start), math.floor(end)
    length = max(end - start, 1)
    mid = (start + end) / 2.0

    win_additional_ratio = 0.3
    win_start_index = max(0, start_index - max(round(win_additional_ratio * length), 5))
    win_end_index = end_index + max(round(win_additional_ratio * length), 5)
    prog_in_seg = prog_in_seg[win_start_index:win_end_index]

    # prog_segment = refine_proposal_by_fitting(prog_in_seg, win_start_index, win_additional_ratio, func_type='tan')
    prog_segment = refine_proposal_by_constraints(prog_in_seg, win_start_index, (start, end), win_additional_ratio)

    # compute iou and record results
    fps = gt_data[vid]['frame'] / gt_data[vid]['duration']
    prog_segment_sec = [(p * 4 + 8) / fps for p in prog_segment]  # convert to seconds
    prog_iou, max_proposal = compute_iou(gt_data[vid]['annotations'], prog_segment_sec, pred['label'])
    delta = prog_iou - iou
    if iou != 0:
        # print(f"Iou Improvement: {delta:.4f}")
        if delta > 0:
            imp_num += 1
        else:
            if abs(delta) > 0.15:
                print("Segment Before:", pred['segment_original'], "After:", prog_segment, "GT:",
                      [(p * fps - 8) / 4 for p in max_proposal])
                print(prog_in_seg)
                print("win_start:", win_start_index, " win_end:", win_end_index, " score:", score, " delta iou:", delta)
                print("video:", vid, " sec segment before:", pred['segment'], " after:", prog_segment_sec, " gt:",
                      max_proposal, " iou:", prog_iou)
                print("-----------------------------------")
                a = 1
            dec_num += 1
    if vid not in results:
        results[vid] = []
    results[vid].append(dict(segment=prog_segment_sec, label=pred['label'], score=ori_score))
print(f"Total improved: {imp_num}, decreased: {dec_num}")


def rank_normalize(x):
    """将一维浮点分数映射为 [0,1] 排名得分（越大越靠前）"""
    x = np.array(x)
    ranks = np.argsort(np.argsort(-x))  # 大值排前
    norm_rank = 1 - ranks / (len(x) - 1)
    return norm_rank


r_score, r_div, r_mono, r_cons = (rank_normalize(score_list), rank_normalize(div_score_list),
                                  rank_normalize(mono_score_list), rank_normalize(consistency_score_list))

# 按 ori_score 排序
sorted_idx = np.argsort(score_list)[::-1]
top_k = 0
top_k_idx = sorted_idx[top_k:-1]

# 取出原始score和progression score
top_ori, top_cons, top_div, top_mono = (np.array(score_list)[top_k_idx], np.array(consistency_score_list)[top_k_idx],
                                        np.array(div_score_list)[top_k_idx], np.array(mono_score_list)[top_k_idx])
max_score = top_ori.max()
min_score = max(0.1, top_ori.min())

# Rank normalize progression scores
r_top_ori, r_top_cons, r_top_div, r_top_mono = (rank_normalize(top_ori), rank_normalize(top_cons),
                                                rank_normalize(top_div), rank_normalize(top_mono))
# 局部微调：只在 top_k 内做加权
# alpha = 0.2  # 微调幅度
# top_fused = r_score * (1 - alpha) + alpha * r_cons[top_k_idx]  # 或者指数融合
top_fused = top_ori * (0.5 + top_cons * 0.5)
top_fused = (top_fused - top_fused.min()) / (top_fused.max() - top_fused.min()) * (max_score - min_score) + min_score


# 替换原始 score
# fused_score = np.array(score_list)
# fused_score[top_k_idx] = top_fused
# results = copy.deepcopy(results_ori)
# idx = 0
# for vid in results:
#     for i in range(len(results[vid])):
#         results[vid][i]['score'] = float(fused_score[idx])
#         idx += 1


def calculate_topk_score_mean_iou():
    def topk_mean_iou(scores, iou_list, to_k=100, fr_k=0):
        idx = np.argsort(scores)[::-1][fr_k:to_k]
        return float(np.mean([iou_list[i] for i in idx]))

    ks = [0, 50, 100, 200, 500, 1000, 2000, 3000, 6000, 7000, 8000, 12000]
    for i in range(len(ks) - 1):
        fr_k = ks[i]
        to_k = ks[i + 1]
        print(f"Top {fr_k}-{to_k} mean IoU - ori: {topk_mean_iou(r_score, iou_list, to_k, fr_k):.4f}, "
              f"cons: {topk_mean_iou(r_cons, iou_list, to_k, fr_k):.4f}, "
              f"div: {topk_mean_iou(r_div, iou_list, to_k, fr_k):.4f}, "
              f"mono: {topk_mean_iou(r_mono, iou_list, to_k, fr_k):.4f}")
        print(f"Top {to_k} mean IoU - ori: {topk_mean_iou(r_score, iou_list, to_k):.4f}, "
              f"cons: {topk_mean_iou(r_cons, iou_list, to_k):.4f}, "
              f"div: {topk_mean_iou(r_div, iou_list, to_k):.4f}, "
              f"mono: {topk_mean_iou(r_mono, iou_list, to_k):.4f}")

    def topk_overlap(a_scores, b_scores, k=100):
        a_idx = set(np.argsort(a_scores)[::-1][:k])
        b_idx = set(np.argsort(b_scores)[::-1][:k])
        return len(a_idx & b_idx) / float(k)

    for k in [50, 100, 200]:
        print("Overlap ori-cons:", k, topk_overlap(r_score, r_cons, k))
        print("Overlap ori-div:", k, topk_overlap(r_score, r_div, k))


def calculate_pr_auc():
    y_true = (np.array(iou_list) >= 0.5).astype(int)
    # ======== 计算 ROC 曲线 ========
    p_ori, recall_ori, _ = precision_recall_curve(y_true, r_score)
    p_div, recall_div, _ = precision_recall_curve(y_true, r_div)
    p_mono, recall_mono, _ = precision_recall_curve(y_true, r_mono)
    p_cons, recall_cons, _ = precision_recall_curve(y_true, r_cons)
    auc_ori, auc_div, auc_mono, auc_cons = (auc(recall_ori, p_ori), auc(recall_div, p_div),
                                            auc(recall_mono, p_mono), auc(recall_cons, p_cons))
    # ======== 绘图 ========
    plt.figure(figsize=(6, 6))
    plt.plot(recall_ori, p_ori, label=f"Original (AUC={auc_ori:.3f})", lw=2)
    plt.plot(recall_div, p_div, label=f"Divergence (AUC={auc_div:.3f})", lw=2)
    plt.plot(recall_mono, p_mono, label=f"Monotonicity (AUC={auc_mono:.3f})", lw=2)
    plt.plot(recall_cons, p_cons, label=f"Consistency (AUC={auc_cons:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR-AUC Comparison (Positive: IoU ≥ 0.5)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_ap():
    def compute_ap(scores, iou_list, thresh=0.5):
        y = (np.array(iou_list) >= thresh).astype(int)
        return average_precision_score(y, scores)

    print("AP(ori/div/mono/cons) @ IoU>=0.5:", compute_ap(r_score, iou_list), compute_ap(r_div, iou_list),
          compute_ap(r_mono, iou_list), compute_ap(r_cons, iou_list))


def calculate_correlation():
    # compute the correlation between scores and iou
    pearson_corr = stats.pearsonr(list(r_score), iou_list)
    spearman_corr = stats.spearmanr(list(r_score), iou_list)
    print(f"Original Score - IoU: Pearson Correlation: {pearson_corr[0]:.4f}, p-value: {pearson_corr[1]:.4e}")
    print(f"Original Score - IoU: Spearman Correlation: {spearman_corr[0]:.4f}, p-value: {spearman_corr[1]:.4e}")
    pearson_corr = stats.pearsonr(list(r_div), iou_list)
    spearman_corr = stats.spearmanr(list(r_div), iou_list)
    print(f"Divergence Score - IoU: Pearson Correlation: {pearson_corr[0]:.4f}, p-value: {pearson_corr[1]:.4e}")
    print(f"Divergence Score - IoU: Spearman Correlation: {spearman_corr[0]:.4f}, p-value: {spearman_corr[1]:.4e}")
    pearson_corr = stats.pearsonr(list(r_mono), iou_list)
    spearman_corr = stats.spearmanr(list(r_mono), iou_list)
    print(f"Monotonicity Score - IoU: Pearson Correlation: {pearson_corr[0]:.4f}, p-value: {pearson_corr[1]:.4e}")
    print(f"Monotonicity Score - IoU: Spearman Correlation: {spearman_corr[0]:.4f}, p-value: {spearman_corr[1]:.4e}")
    pearson_corr = stats.pearsonr(list(r_cons), iou_list)
    spearman_corr = stats.spearmanr(list(r_cons), iou_list)
    print(f"Consistency Score - IoU: Pearson Correlation: {pearson_corr[0]:.4f}, p-value: {pearson_corr[1]:.4e}")
    print(f"Consistency Score - IoU: Spearman Correlation: {spearman_corr[0]:.4f}, p-value: {spearman_corr[1]:.4e}")
    new_score = [min(0.75 * div + 0.25 * mono, 1) for div, mono in zip(div_score_list, mono_score_list)]
    pearson_corr = stats.pearsonr(new_score, iou_list)
    spearman_corr = stats.spearmanr(new_score, iou_list)
    print(f"New Score - IoU: Pearson Correlation: {pearson_corr[0]:.4f}, p-value: {pearson_corr[1]:.4e}")
    print(f"New Score - IoU: Spearman Correlation: {spearman_corr[0]:.4f}, p-value: {spearman_corr[1]:.4e}")
    new_score2 = [min(ori_sco * (1.5 * div + 0.5 * mono) / 2, 1) for div, mono, consistency, ori_sco in
                  zip(div_score_list, mono_score_list, consistency_score_list, score_list)]
    pearson_corr = stats.pearsonr(new_score2, iou_list)
    spearman_corr = stats.spearmanr(new_score2, iou_list)
    print(f"New Score 2 - IoU: Pearson Correlation: {pearson_corr[0]:.4f}, p-value: {pearson_corr[1]:.4e}")
    print(f"New Score 2 - IoU: Spearman Correlation: {spearman_corr[0]:.4f}, p-value: {spearman_corr[1]:.4e}")


def evaluate_map(_results):
    result_eval = dict(results=_results)
    evaluator = build_evaluator(dict(prediction_filename=result_eval,
                                     type="mAP",
                                     subset="validation",
                                     tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
                                     ground_truth_filename="data/thumos-14/annotations/thumos_14_anno.json", ))
    # evaluate and output
    metrics_dict = evaluator.evaluate()
    average_mAP = metrics_dict["average_mAP"]
    tiou_all = metrics_dict["tiou_all"]
    class_index = evaluator.activity_index
    ap_per_label = metrics_dict["ap_per_label"]
    valid_prediction_ids = metrics_dict["valid_prediction_ids"]
    evaluator.logging()
    return average_mAP


calculate_topk_score_mean_iou()
calculate_pr_auc()
calculate_ap()
calculate_correlation()
average_mAP = evaluate_map(results_ori)
print(f"Average mAP before adjustment: {average_mAP:.4f}")
average_mAP = evaluate_map(results)
print(f"Average mAP after adjustment: {average_mAP:.4f}")

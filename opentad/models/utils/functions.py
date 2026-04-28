import torch


def gaussian_cdf(x, mean=0.5, std=1.0):
    """
    高斯累积分布函数 (CDF)
    x ∈ [0,1]
    mean: 控制中心点
    std: 控制“平滑度”，越小则两边越陡、中间越慢
    输出范围 [0,1]
    """
    x = torch.clamp(x, 0.0, 1.0)
    # erf 是误差函数，erf(z) = 2/sqrt(pi) * ∫ e^{-t^2} dt
    cdf = 0.5 * (1 + torch.erf((x - mean) / (std * (2 ** 0.5))))
    return torch.clamp(cdf, 0.0, 1.0)


def tan_curve(x, alpha=0.8):
    norm_t = alpha * x
    prog_targets = (torch.tan(norm_t) / torch.tan(torch.tensor(alpha)) + 1) * 0.5
    return torch.clamp(prog_targets, 0.0, 1.0)

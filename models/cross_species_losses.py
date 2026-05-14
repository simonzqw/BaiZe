import torch


def weighted_delta_mse(pred_delta, true_delta, weight=None):
    err = (pred_delta - true_delta) ** 2
    return (err if weight is None else err * weight).mean()


def topk_corr_loss(pred_delta, true_delta, k=20, eps=1e-8):
    k = min(k, true_delta.shape[1])
    idx = torch.topk(true_delta.abs(), k=k, dim=1).indices
    p = torch.gather(pred_delta, 1, idx)
    t = torch.gather(true_delta, 1, idx)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    corr = (p * t).sum(1) / (torch.sqrt((p**2).sum(1) + eps) * torch.sqrt((t**2).sum(1) + eps))
    return 1.0 - corr.mean()


def sign_consistency_loss(pred_delta, true_delta):
    return torch.relu(-(pred_delta * true_delta)).mean()


def delta_norm_loss(pred_delta, true_delta, eps=1e-8):
    pn = torch.sqrt((pred_delta**2).sum(1) + eps)
    tn = torch.sqrt((true_delta**2).sum(1) + eps)
    return ((pn - tn) ** 2).mean()


def residual_l2_loss(correction):
    return (correction**2).mean()

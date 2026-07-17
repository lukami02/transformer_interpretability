import torch
import torch.nn.functional as F
import numpy as np

from ....scripts.config import ModelConfig

def normalize_saliency(saliency: torch.Tensor) -> torch.Tensor:
    """
    Min-max normalisation saliency map to [0, 1].
    """
    s = saliency.float()
    s_min = s.min()
    s_max = s.max()
    if (s_max - s_min).abs() < 1e-8:
        return torch.zeros_like(s)
    return (s - s_min) / (s_max - s_min)


def upsample_to_image_size(
    saliency: torch.Tensor,
    model_cfg: ModelConfig,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Upsample saliency map to model_cfg.image_size x model_cfg.image_size if it isn't already.
    """
    size = model_cfg.image_size
    if saliency.shape[-2:] == (size, size):
        return saliency

    added_channel = False
    if saliency.dim() == 2:            
        saliency = saliency.unsqueeze(0)
        added_channel = True

    s = saliency.unsqueeze(0).float() 
    s = F.interpolate(
        s,
        size=(size, size),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    s = s.squeeze(0)               
    if added_channel:
        s = s.squeeze(0)    
    return s        


def saliency_to_patch_scores(
    saliency: torch.Tensor,
    model_cfg: ModelConfig,
    aggregation: str = "mean",
) -> np.ndarray:
    """
    Returns a 1D numpy array of saliency scores for each patch.
    """
    n = model_cfg.num_patches_per_dim
    p = model_cfg.patch_size

    sal = normalize_saliency(saliency)

    if sal.shape[-2:] == (n, n):
        return sal.squeeze(0).flatten().detach().cpu().numpy()

    sal = upsample_to_image_size(sal, model_cfg)
    sal = sal.squeeze(0) 

    agg_fn = {
        "mean": lambda x: x.mean().item(),
        "max":  lambda x: x.max().item(),
        "sum":  lambda x: x.sum().item(),
    }.get(aggregation)

    if agg_fn is None:
        raise ValueError(
            f"Unknown aggregation: '{aggregation}'. Use 'mean', 'max' or 'sum'."
        )

    scores = [
        agg_fn(sal[pi * p : (pi + 1) * p, pj * p : (pj + 1) * p])
        for pi in range(n)
        for pj in range(n)
    ]

    return np.array(scores)


def saliency_to_pixel_scores(
    saliency: torch.Tensor,
    model_cfg: ModelConfig,
) -> np.ndarray:
    """
    Returns a 1D numpy array of pixel-level saliency values.
    """
    sal = normalize_saliency(saliency)
    sal = upsample_to_image_size(sal, model_cfg)
    return sal.squeeze(0).flatten().detach().cpu().numpy()


def get_sorted_patches(
    saliency: torch.Tensor,
    model_cfg: ModelConfig,
    descending: bool = True,
) -> list:
    """
    Returns a list of (patch_row, patch_col) tuples sorted by saliency score.
    """
    n = model_cfg.num_patches_per_dim
    scores = saliency_to_patch_scores(saliency, model_cfg)

    indexed = [
        ((pi, pj), scores[pi * n + pj])
        for pi in range(n)
        for pj in range(n)
    ]
    indexed.sort(key=lambda x: x[1], reverse=descending)
    return [patch for patch, _ in indexed]


def get_sorted_patch_indices(
    saliency: torch.Tensor,
    model_cfg: ModelConfig,
    descending: bool = True,
) -> np.ndarray:
    scores = saliency_to_patch_scores(saliency, model_cfg)
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    return order
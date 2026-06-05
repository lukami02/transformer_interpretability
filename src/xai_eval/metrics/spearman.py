import numpy as np
import torch
from itertools import combinations
from scipy.stats import spearmanr

from ....scripts.config import ModelConfig
from ..utils.saliency import saliency_to_patch_scores, saliency_to_pixel_scores


def spearman_single(
    saliency_a: torch.Tensor,
    saliency_b: torch.Tensor,
    model_cfg: ModelConfig,
    level: str = "patch",
) -> float:
    """
    Spearman correlation between two saliency maps for a single image.
    """
    if level == "patch":
        arr_a = saliency_to_patch_scores(saliency_a, model_cfg)
        arr_b = saliency_to_patch_scores(saliency_b, model_cfg)
    elif level == "pixel":
        arr_a = saliency_to_pixel_scores(saliency_a, model_cfg)
        arr_b = saliency_to_pixel_scores(saliency_b, model_cfg)
    else:
        raise ValueError(f"Unknown level: '{level}'. Use 'patch' or 'pixel'.")

    if arr_a.std() < 1e-8 or arr_b.std() < 1e-8:
        return float("nan")

    rho, _ = spearmanr(arr_a, arr_b)
    return float(rho)


def compute_spearman(
    saliencies_a: list,
    saliencies_b: list,
    model_cfg: ModelConfig,
    level: str = "patch",
) -> dict:
    """
    Computes the Spearman rho averaged across a set of images.
    """
    assert len(saliencies_a) == len(saliencies_b), (
        f"Different number of saliency maps: {len(saliencies_a)} vs {len(saliencies_b)}"
    )

    rhos = [spearman_single(sa, sb, model_cfg, level) for sa, sb in zip(saliencies_a, saliencies_b)]
    valid_rhos = [r for r in rhos if not np.isnan(r)]

    return {
        "mean_rho":      float(np.mean(valid_rhos)) if valid_rhos else float("nan"),
        "std_rho":       float(np.std(valid_rhos))  if valid_rhos else float("nan"),
        "per_image_rho": rhos,
        "valid_count":   len(valid_rhos),
    }


def compute_spearman_matrix(
    named_saliencies: dict,
    model_cfg: ModelConfig,
    level: str = "patch",
) -> dict:
    """
    Computes the Spearman rho for all pairs of methods.
    """
    results = {}
    for name_a, name_b in combinations(named_saliencies.keys(), 2):
        results[f"{name_a} vs {name_b}"] = compute_spearman(
            named_saliencies[name_a],
            named_saliencies[name_b],
            model_cfg=model_cfg,
            level=level,
        )
    return results
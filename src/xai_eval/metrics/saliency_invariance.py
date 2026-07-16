import torch
import numpy as np
from typing import Callable, Optional
from scipy.stats import spearmanr

from ....scripts.config import ModelConfig
from ..utils.saliency import saliency_to_patch_scores
from ..utils.perturbation import get_baseline, apply_patch_perturbation


def _shuffle_image_patches(
    image: torch.Tensor,
    model_cfg: ModelConfig,
    perm: np.ndarray,
) -> torch.Tensor:
    """
    Shuffles the patches of an image according to a given permutation.
    """
    n = model_cfg.num_patches_per_dim
    p = model_cfg.patch_size

    shuffled = image.clone()
    for new_idx, old_idx in enumerate(perm):
        pi_new, pj_new = divmod(new_idx, n)
        pi_old, pj_old = divmod(int(old_idx), n)

        r0_new, c0_new = pi_new * p, pj_new * p
        r0_old, c0_old = pi_old * p, pj_old * p

        shuffled[:, r0_new:r0_new + p, c0_new:c0_new + p] = \
            image[:, r0_old:r0_old + p, c0_old:c0_old + p]

    return shuffled


def patch_shuffle_test(
    image: torch.Tensor,
    saliency: torch.Tensor,
    target_class: int,
    xai_fn: Callable,
    model_cfg: ModelConfig,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> dict:
    """
    Performs a patch shuffle test for a single image. 
    It shuffles the patches of the image, computes the saliency map for
    the shuffled image and then compares it to the original saliency map
    using Spearman correlation.
    """
    n = model_cfg.num_patches_per_dim
    num_patches = n * n

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_patches) 

    image_cpu = image.detach().cpu()
    shuffled_img = _shuffle_image_patches(image_cpu, model_cfg, perm)

    sal_shuffled = xai_fn(shuffled_img.to(device), target_class).cpu()

    scores_shuffled = saliency_to_patch_scores(sal_shuffled, model_cfg)
    scores_depermuted = np.empty_like(scores_shuffled)
    for new_idx, old_idx in enumerate(perm):
        scores_depermuted[old_idx] = scores_shuffled[new_idx]

    scores_original = saliency_to_patch_scores(saliency, model_cfg)

    rho, _ = spearmanr(scores_original, scores_depermuted)

    return {
        "rho": rho,
        "perm": perm,
        "saliency_shuffled_depermuted": scores_depermuted,
    }


def single_patch_perturbation_test(
    image: torch.Tensor,
    saliency: torch.Tensor,
    target_class: int,
    xai_fn: Callable,
    model_cfg: ModelConfig,
    baseline_strategy: str = "mean",
    device: str = "cpu",
    patch_idx: Optional[tuple] = None,
    seed: Optional[int] = None,
) -> dict:
    """ 
    Performs a single patch perturbation test for a single image.
    It perturbs a single patch of the image, computes the saliency map for 
    the perturbed image and then compares it to the original saliency map
    using Spearman correlation.
    """
    n = model_cfg.num_patches_per_dim

    if patch_idx is None:
        rng = np.random.default_rng(seed)
        pi, pj = rng.integers(0, n), rng.integers(0, n)
        patch_idx = (int(pi), int(pj))

    image_cpu = image.detach().cpu()
    baseline = get_baseline(image_cpu, model_cfg=model_cfg, strategy=baseline_strategy)
    perturbed_img = apply_patch_perturbation(
        image_cpu.clone(), [patch_idx], baseline, model_cfg,
    )

    sal_perturbed = xai_fn(perturbed_img.to(device), target_class).cpu()

    scores_original = saliency_to_patch_scores(saliency, model_cfg)
    scores_perturbed = saliency_to_patch_scores(sal_perturbed, model_cfg)

    rho, _ = spearmanr(scores_original, scores_perturbed)

    return {
        "rho": rho,
        "patch_idx": patch_idx,
        "saliency_perturbed": scores_perturbed,
    }
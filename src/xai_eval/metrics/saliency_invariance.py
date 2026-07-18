import torch
import numpy as np
from typing import Callable, Literal, Optional
from scipy.stats import spearmanr

from ....scripts.config import ModelConfig
from ..utils.saliency import saliency_to_patch_scores
from ..utils.perturbation import get_baseline, apply_patch_perturbation


PatchSelection = Literal["random", "most_salient", "least_salient"]


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


def _select_patch_indices(
    scores_original: np.ndarray,
    n: int,
    selection: PatchSelection,
    rng: np.random.Generator,
) -> tuple:
    """
    Selects a single (pi, pj) patch index according to the given strategy.
    """
    if selection == "random":
        pi, pj = rng.integers(0, n), rng.integers(0, n)
        return int(pi), int(pj)
    elif selection == "most_salient":
        flat_idx = int(np.argmax(scores_original))
        return divmod(flat_idx, n)
    elif selection == "least_salient":
        flat_idx = int(np.argmin(scores_original))
        return divmod(flat_idx, n)
    else:
        raise ValueError(f"Unknown patch_selection: {selection}")
    

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
    patch_selection: PatchSelection = "random",
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
        patch_idx = _select_patch_indices(
            saliency_to_patch_scores(saliency, model_cfg), n, patch_selection, rng
        )
 
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

def run_full_perturbation_suite(
    image: torch.Tensor,
    saliency: torch.Tensor,
    target_class: int,
    xai_fn: Callable,
    model_cfg: ModelConfig,
    baseline_strategy: str = "mean",
    device: str = "cpu",
    n_random: int = 10,
    base_seed: int = 0,
) -> dict:
    """
    Runs the full robustness battery for a single image:
      - single_patch_perturbation_test for most_salient / least_salient
      - single_patch_perturbation_test for n_random random patches
      - patch_shuffle_test for most_salient / least_salient
      - patch_shuffle_test for n_random random shuffles
    """
    results = {"single_patch": {}, "shuffle": {}}
 
    # single patch: most/least salient -> one deterministic rho each
    for selection in ("most_salient", "least_salient"):
        out = single_patch_perturbation_test(
            image, saliency, target_class, xai_fn, model_cfg,
            baseline_strategy=baseline_strategy, device=device,
            patch_selection=selection, seed=base_seed,
        )
        results["single_patch"][selection] = {"rho": out["rho"]}
 
    # single patch
    random_single_rhos = np.array([
        single_patch_perturbation_test(
            image, saliency, target_class, xai_fn, model_cfg,
            baseline_strategy=baseline_strategy, device=device,
            patch_selection="random", seed=base_seed + i,
        )["rho"]
        for i in range(n_random)
    ])
    results["single_patch"]["random"] = {
        "rho_mean": float(np.nanmean(random_single_rhos)),
        "rho_std": float(np.nanstd(random_single_rhos)),
    }
 
    # shuffle
    random_shuffle_rhos = np.array([
        patch_shuffle_test(
            image, saliency, target_class, xai_fn, model_cfg,
            device=device, seed=base_seed + i,
        )["rho"]
        for i in range(n_random)
    ])
    results["shuffle"]["random"] = {
        "rho_mean": float(np.nanmean(random_shuffle_rhos)),
        "rho_std": float(np.nanstd(random_shuffle_rhos)),
    }
 
    return results
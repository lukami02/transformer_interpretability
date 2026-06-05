import torch
from ....scripts.config import ModelConfig
from .saliency import get_sorted_patches


def get_baseline(image: torch.Tensor, model_cfg: ModelConfig, strategy: str = "mean") -> torch.Tensor:
    """
    Returns a baseline tensor based on the specified strategy.
    """
    if strategy == "mean":
        return torch.tensor(model_cfg.mean, dtype=torch.float32, device=image.device).view(3, 1, 1) 
    elif strategy == "zero":
        return torch.zeros(3, 1, 1, device=image.device)
    elif strategy == "noise":
        mean = torch.tensor(model_cfg.mean, dtype=torch.float32, device=image.device).view(3, 1, 1)
        noise = torch.randn(3, 1, 1, device=image.device) * 0.1
        return (mean + noise).clamp(0, 1)
    else:
        raise ValueError(
            f"Unknown strategy: '{strategy}'. Use 'mean', 'zero' or 'noise'."
        )


def apply_patch_perturbation(
    image: torch.Tensor,
    patches_to_remove: list,
    baseline: torch.Tensor,
    model_cfg: ModelConfig,
) -> torch.Tensor:
    """
    Removes specified patches from the image by replacing them with the baseline.
    """
    p = model_cfg.patch_size
    for pi, pj in patches_to_remove:
        r0, c0 = pi * p, pj * p
        image[:, r0 : r0 + p, c0 : c0 + p] = baseline
    return image


def perturbation_steps(
    image: torch.Tensor,
    saliency: torch.Tensor,
    model_cfg: ModelConfig,
    n_steps: int,
    order: str,
    baseline_strategy: str = "mean",
):
    """
    Generator that yields perturbed images at each step of the perturbation process.
    """
    sorted_patches = get_sorted_patches(saliency, model_cfg, descending=(order == "morf"))
    total_patches = len(sorted_patches)

    baseline = get_baseline(image, model_cfg=model_cfg, strategy=baseline_strategy)

    current = image.clone()
    yield current.clone() 
    patches_per_step = max(1, total_patches // n_steps)
    ptr = 0
    for step in range(n_steps):
        if step == n_steps - 1:
            end = total_patches
        else:
            end = min(ptr + patches_per_step, total_patches)
            
        if ptr >= total_patches:
            break
            
        current = apply_patch_perturbation(current, sorted_patches[ptr:end], baseline, model_cfg)
        yield current.clone()
        ptr = end
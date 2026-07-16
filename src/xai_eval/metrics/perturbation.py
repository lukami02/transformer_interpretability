import torch
import numpy as np
from typing import Optional

from models import model_base

from ....scripts.config import ModelConfig
from ..utils.perturbation import perturbation_steps, perturbation_mask_steps


@torch.no_grad()
def _compute_curve_slow(
    image: torch.Tensor,
    saliency: torch.Tensor,
    model: torch.nn.Module,
    target_class: int,
    model_cfg: ModelConfig,
    n_steps: int,
    order: str,
    baseline_strategy: str,
    device: str,
) -> np.ndarray:
    """
    Computes the MoRF or LeRF curve for a single image one by one.
    """
    model    = model.to(device).eval()
    image    = image.to(device)
    saliency = saliency.to(device)

    probs = []
    for perturbed_img in perturbation_steps(
        image, saliency, model_cfg,
        n_steps=n_steps,
        order=order,
        baseline_strategy=baseline_strategy,
    ):
        logits = model(perturbed_img.unsqueeze(0).to(device))
        prob = torch.softmax(logits, dim=-1)[0, target_class].item()
        probs.append(prob)

        if len(probs) > n_steps + 1:
            break

    return np.array(probs)

@torch.no_grad()
def _compute_curve_batched(
    image: torch.Tensor,
    saliency: torch.Tensor,
    model_base: torch.nn.Module,
    target_class: int,
    model_cfg: ModelConfig,
    n_steps: int, 
    order: str,
    device: str,
) -> np.ndarray:
    """
    Computes the MoRF or LeRF curve for a single image as one batch.
    """
    model_base = model_base.to(device).eval()
    image      = image.to(device)
    saliency   = saliency.to(device)

    mask_stack = perturbation_mask_steps(
        saliency, model_cfg, n_steps=n_steps, order=order,
    ).to(device)  # [steps, num_patches] (CLS added later by the model)
 
    n_rows = mask_stack.shape[0]
    imgs_batch = image.unsqueeze(0).expand(n_rows, *image.shape)  # [steps, C, H, W]
 
    logits = model_base(imgs_batch, mask=mask_stack)
    probs = torch.softmax(logits, dim=-1)[:, target_class]
 
    return probs.detach().cpu().numpy()

def _compute_curve(
    image: torch.Tensor,
    saliency: torch.Tensor,
    model: torch.nn.Module,
    target_class: int,
    model_cfg: ModelConfig,
    n_steps: int,
    order: str,
    baseline_strategy: str,
    device: str,
    model_base: Optional[torch.nn.Module] = None,
) -> np.ndarray:
    """
    Dispatches to the fast batched path when available, otherwise
    falls back to the slow image-by-image path.
    """
    if model_base is not None:
        return _compute_curve_batched(
            image, saliency, model_base, target_class,
            model_cfg=model_cfg, n_steps=n_steps, order=order, device=device,
        )
    return _compute_curve_slow(
        image, saliency, model, target_class,
        model_cfg=model_cfg, n_steps=n_steps, order=order,
        baseline_strategy=baseline_strategy, device=device,
    )
 

def _auc(curve: np.ndarray, limit_ratio: Optional[float] = None) -> float:
    """Computes the area under the curve using the trapezoidal rule."""
    if limit_ratio is not None:
        n_elements = int(np.ceil(len(curve) * limit_ratio))
        n_elements = max(min(n_elements, len(curve)), 2)
        curve_segment = curve[:n_elements]
    else:
        curve_segment = curve

    return float(np.trapz(curve_segment) / max(len(curve_segment) - 1, 1))

def compute_morf(
    image: torch.Tensor,
    saliency: torch.Tensor,
    model: torch.nn.Module,
    target_class: int,
    model_cfg: ModelConfig,
    n_steps: int = 100,
    baseline_strategy: str = "mean",
    device: str = "cpu",
    model_base: Optional[torch.nn.Module] = None,
) -> dict:
    """
    MoRF (Most Relevant First) for a single image.
    """
    curve = _compute_curve(
        image, saliency, model, target_class,
        model_cfg=model_cfg, n_steps=n_steps, order="morf",
        baseline_strategy=baseline_strategy, device=device,
        model_base=model_base,
    )
    return {
        "auc": _auc(curve), 
        "auc_30": _auc(curve, limit_ratio=0.3),
        "curve": curve
    }


def compute_lerf(
    image: torch.Tensor,
    saliency: torch.Tensor,
    model: torch.nn.Module,
    target_class: int,
    model_cfg: ModelConfig,
    n_steps: int = 100,
    baseline_strategy: str = "mean",
    device: str = "cpu",
    model_base: Optional[torch.nn.Module] = None,
) -> dict:
    """
    LeRF (Least Relevant First) for a single image.
    """
    curve = _compute_curve(
        image, saliency, model, target_class,
        model_cfg=model_cfg, n_steps=n_steps, order="lerf",
        baseline_strategy=baseline_strategy, device=device,
        model_base=model_base,
    )
    return {
        "auc": _auc(curve), 
        "auc_30": _auc(curve, limit_ratio=0.3),
        "curve": curve
    }

def aggregate_perturbation_results(per_image_results: list) -> dict:
    """
    Aggregates perturbation results across a set of images.
    """
    aucs    = [r["auc"] for r in per_image_results]
    aucs_30 = [r["auc_30"] for r in per_image_results]
    curves  = [r["curve"] for r in per_image_results]
 
    max_len = max(len(c) for c in curves)
    padded  = [np.pad(c, (0, max_len - len(c)), mode="edge") for c in curves]
 
    return {
        "mean_auc":       float(np.mean(aucs)),
        "std_auc":        float(np.std(aucs)),
        "mean_auc_30":    float(np.mean(aucs_30)),
        "std_auc_30":     float(np.std(aucs_30)),
        "mean_curve":     np.mean(padded, axis=0),
        "per_image_auc":  aucs,
        "per_image_auc_30": aucs_30,
    }
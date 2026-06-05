import torch

from ....scripts.config import ModelConfig
from ..utils.saliency import normalize_saliency, upsample_to_image_size

def scale_bbox(
    bbox: tuple,
    original_size: tuple,
    model_cfg: ModelConfig,
) -> tuple:
    """
    Scales a bounding box from original image size to model_cfg.image_size.
    """
    orig_w, orig_h = original_size
    target = model_cfg.image_size
    sx     = target / orig_w
    sy     = target / orig_h

    x_min, y_min, x_max, y_max = bbox
    return (
        int(x_min * sx),
        int(y_min * sy),
        int(x_max * sx),
        int(y_max * sy),
    )


def _argmax_position(
    saliency: torch.Tensor,
    model_cfg: ModelConfig,
) -> tuple[int, int]:
    """
    Returns the (row, col) position of the maximum in the saliency map
    after upsampling to image_size.
    """
    sal = normalize_saliency(saliency)
    sal = upsample_to_image_size(sal, model_cfg).squeeze(0) 
    flat_idx = sal.argmax().item()
    return flat_idx // model_cfg.image_size, flat_idx % model_cfg.image_size


def _bbox_with_tolerance(bbox: tuple, tolerance: int, image_size: int) -> tuple:
    """Expands a bounding box by a given tolerance, clamped to the image boundaries."""
    x_min, y_min, x_max, y_max = bbox
    lim = image_size - 1
    return (
        max(0,   x_min - tolerance),
        max(0,   y_min - tolerance),
        min(lim, x_max + tolerance),
        min(lim, y_max + tolerance),
    )


def pointing_game_single(
    saliency: torch.Tensor,
    bbox: tuple,
    model_cfg: ModelConfig,
    tolerance: int = 0,
) -> bool:
    """
    Pointing Game for a single image with a single bounding box.
    """
    max_row, max_col = _argmax_position(saliency, model_cfg)
    x_min, y_min, x_max, y_max = _bbox_with_tolerance(bbox, tolerance, model_cfg.image_size)
    return (x_min <= max_col <= x_max) and (y_min <= max_row <= y_max)


def pointing_game_multi_bbox(
    saliency: torch.Tensor,
    bboxes: list,
    model_cfg: ModelConfig,
    tolerance: int = 0,
) -> bool:
    """
    Pointing Game for a single image with multiple bounding boxes.

    Useful for COCO where a single image may have multiple instances of the same class.
    """
    max_row, max_col = _argmax_position(saliency, model_cfg)
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = _bbox_with_tolerance(bbox, tolerance, model_cfg.image_size)
        if (x_min <= max_col <= x_max) and (y_min <= max_row <= y_max):
            return True
    return False


def compute_pointing_game(
    saliencies: list,
    bboxes: list,
    model_cfg: ModelConfig,
    tolerance: int = 0,
    multi_bbox: bool = False,
) -> dict:
    """
    Computes the Pointing Game metric for a set of images.
    """
    assert len(saliencies) == len(bboxes), (
        f"Different number of saliency maps ({len(saliencies)}) and bboxes ({len(bboxes)})"
    )

    fn = pointing_game_multi_bbox if multi_bbox else pointing_game_single
    per_image_hit = [
        fn(sal, bbox, model_cfg, tolerance)
        for sal, bbox in zip(saliencies, bboxes)
    ]

    hits  = sum(per_image_hit)
    total = len(per_image_hit)

    return {
        "accuracy":      hits / total if total > 0 else 0.0,
        "hits":          hits,
        "total":         total,
        "per_image_hit": per_image_hit,
    }
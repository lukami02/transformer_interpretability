from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import torch.nn as nn
from typing import List, Tuple


@dataclass(frozen=True)
class ModelConfig:
    """
    Architectural and preprocessing configuration of the model.
    """
    patch_size: int
    image_size: int
    mean: Tuple[float, ...] = (0.5, 0.5, 0.5)
    std: Tuple[float, ...] = (0.5, 0.5, 0.5)

    @property
    def num_patches_per_dim(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.num_patches_per_dim ** 2

    def __str__(self) -> str:
        return (
            f"ModelConfig("
            f"patch_size={self.patch_size}, "
            f"image_size={self.image_size}, "
            f"num_patches={self.num_patches}, "
            f"mean={self.mean}, "
            f"std={self.std}"
            f")"
        )

    @classmethod
    def from_vit(cls, model: nn.Module) -> ModelConfig:
        """
        Factory method to create ModelConfig from a VisionTransformer instance.
        """
        patch_embed = getattr(model, "patch_embed", None)
        if patch_embed is None:
            raise AttributeError("Model does not have a 'patch_embed' layer. ")

        patch_size = getattr(patch_embed, "patch_size", None)
        img_size   = getattr(patch_embed, "img_size",   None)

        if patch_size is None:
            raise AttributeError("patch_embed does not have a 'patch_size' attribute.")
        if img_size is None:
            raise AttributeError("patch_embed does not have a 'img_size' attribute.")

        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        if isinstance(img_size, (tuple, list)):
            img_size = img_size[0]

        cfg = getattr(model, "default_cfg", {})
        kwargs = {}
        if "mean" in cfg:
            kwargs["mean"] = tuple(cfg["mean"])
        if "std" in cfg:
            kwargs["std"] = tuple(cfg["std"])

        return cls(
            patch_size=int(patch_size), 
            image_size=int(img_size),
            **kwargs
        )


@dataclass
class EvalConfig:
    """
    Configuration for the evaluation run.
    """
    model_cfg:           ModelConfig
    n_steps:             int  = 49
    baseline_strategy:   str  = "mean"    # "mean" | "zero" | "noise"
    spearman_level:      str  = "patch"   # "patch" | "pixel"
    pointing_tolerance:  int  = 0
    pointing_multi_bbox: bool = False
    device:              str  = "cpu"
    verbose:             bool = True
    verbose_step:        int  = 10
    seed:                int  = 42


@dataclass
class RunConfig:
    # Paths 
    data_dir: Path = Path(__file__).resolve().parents[1] / "data" / "imagenet_xai_subset"
    results_dir: Path = Path(__file__).resolve().parents[1] / "results"

    # Data 
    num_samples: int = 50
    seed: int = 42
    batch_size: int = 1          
    num_workers: int = 4
    confidence_threshold: float = 0.5

    # Model 
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True

    # XAI methods 
    methods: List[str] = field(default_factory=lambda: [
        "Vanilla Gradient", "Gradient X Input", 
        #"Integrated Gradients", "Smooth Gradient", 
        "GradCAM", "Attention Rollout", 
        "Transformer Attribution", "GMAR", "LRP",
        "RISE", "SHAP"
    ])
    
    # EvalConfig hyperparameters 
    n_steps: int = 50
    baseline_strategy: str = "mean"   # "mean" | "zero" | "noise"
    spearman_level: str = "patch"     # "patch" | "pixel"
    pointing_tolerance: int = 0
    pointing_multi_bbox: bool = False

    # Black-box XAI hyperparameters
    black_box_batch: int = 192
    rise_mask: int = 8000
    rise_mask_prob: float = 0.5
    kernel_shap_mask: int = 2000
    kernel_shap_ridge_alpha: float = 1e-3

    # Runtime 
    device: str = "auto"              # "auto" | "cuda" | "cpu"
    verbose: bool = True
    save_results: bool = True
    verbose_step: int = 10

    def resolved_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def to_eval_config(self, model_cfg: ModelConfig) -> EvalConfig:
        return EvalConfig(
            model_cfg=model_cfg, 
            n_steps=self.n_steps,
            baseline_strategy=self.baseline_strategy,
            spearman_level=self.spearman_level,
            pointing_tolerance=self.pointing_tolerance,
            pointing_multi_bbox=self.pointing_multi_bbox,
            device=self.resolved_device(),
            verbose=self.verbose,
            verbose_step=self.verbose_step,
            seed=self.seed
        )


def parse_args(argv: list[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(
        description="XAI Evaluation pipeline for ViT models on ImageNet subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    g = parser.add_argument_group("Paths")
    g.add_argument("--data-dir",    type=Path, default=RunConfig.data_dir,    help="Root dir of the cached ImageNet subset")
    g.add_argument("--results-dir",  type=Path, default=RunConfig.results_dir,  help="Where to save result JSONs")

    # Data
    g = parser.add_argument_group("Data")
    g.add_argument("--num-samples",           type=int,   default=RunConfig.num_samples)
    g.add_argument("--seed",                  type=int,   default=RunConfig.seed)
    g.add_argument("--batch-size",            type=int,   default=RunConfig.batch_size)
    g.add_argument("--num-workers",           type=int,   default=RunConfig.num_workers)
    g.add_argument("--confidence-threshold",  type=float, default=RunConfig.confidence_threshold)

    # Model
    g = parser.add_argument_group("Model")
    g.add_argument("--model-name",  type=str,  default=RunConfig.model_name)
    g.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights")

    # XAI methods
    g = parser.add_argument_group("XAI Methods")
    g.add_argument(
        "--methods",
        nargs="+",
        default=RunConfig().methods,
        help="Which XAI methods to evaluate",
    )

    # EvalConfig hyperparams
    g = parser.add_argument_group("Evaluator Hyperparameters")
    g.add_argument("--n-steps",              type=int,   default=RunConfig.n_steps)
    g.add_argument("--baseline-strategy",    type=str,   default=RunConfig.baseline_strategy, choices=["mean", "zero", "noise"])
    g.add_argument("--spearman-level",       type=str,   default=RunConfig.spearman_level,    choices=["patch", "pixel"])
    g.add_argument("--pointing-tolerance",   type=int,   default=RunConfig.pointing_tolerance)
    g.add_argument("--pointing-multi-bbox",  action="store_true", help="Allow hit if peak falls in any valid bbox")

    # Black-box hyperparams
    g = parser.add_argument_group("Black-box (RISE / KernelSHAP) Hyperparameters")
    g.add_argument("--black-box-batch",         type=int,   default=RunConfig.black_box_batch)
    g.add_argument("--rise-mask",               type=int,   default=RunConfig.rise_mask)
    g.add_argument("--rise-mask-prob",          type=float, default=RunConfig.rise_mask_prob)
    g.add_argument("--kernel-shap-mask",        type=int,   default=RunConfig.kernel_shap_mask)
    g.add_argument("--kernel-shap-ridge-alpha", type=float, default=RunConfig.kernel_shap_ridge_alpha)

    # Runtime
    g = parser.add_argument_group("Runtime")
    g.add_argument("--device",        type=str,  default=RunConfig.device, choices=["auto", "cuda", "cpu"])
    g.add_argument("--no-verbose",    action="store_true")
    g.add_argument("--save-results", action=argparse.BooleanOptionalAction, 
                   default=RunConfig.save_results, help="Save results to --results-dir as JSON")
    g.add_argument("--verbose-step",  type=int, default=RunConfig.verbose_step)

    args = parser.parse_args(argv)

    return RunConfig(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        confidence_threshold=args.confidence_threshold,
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        methods=args.methods,
        n_steps=args.n_steps,
        baseline_strategy=args.baseline_strategy,
        spearman_level=args.spearman_level,
        pointing_tolerance=args.pointing_tolerance,
        pointing_multi_bbox=args.pointing_multi_bbox,
        black_box_batch=args.black_box_batch,
        rise_mask=args.rise_mask,
        rise_mask_prob=args.rise_mask_prob,
        kernel_shap_mask=args.kernel_shap_mask,
        kernel_shap_ridge_alpha=args.kernel_shap_ridge_alpha,
        device=args.device,
        verbose=not args.no_verbose,
        verbose_step=args.verbose_step,
        save_results=args.save_results
    )
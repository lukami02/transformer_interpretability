from __future__ import annotations
import json
import torch

from .config import parse_args, ModelConfig
from ..src.xai_eval.evaluate import XAIEvaluator
from ..src.loader.model_loader import create_model
from ..src.loader.data_loader import get_xai_dataloader
from ..src.xai import *

ALL_METHODS = {
    "Vanilla Gradient": VanillaGradient,
    "Gradient X Input": GradientXInput,
    #"Integrated Gradients": IntegratedGradients,
    #"Smooth Gradient": SmoothGrad,   
    "GradCAM": GradCAM,
    "Attention Rollout": AttentionRollout,
    "Transformer Attribution": TransformerAttribution,
    "GMAR": GMAR,
    "LRP": LRP,
    "RISE": RISE,
    "SHAP": KernelSHAP,
}

if __name__ == "__main__":
    cfg = parse_args()
    device = cfg.resolved_device()
    print(f"Device: {device}")
    print(f"Config: {cfg}")

    # Model
    model = create_model(model_name=cfg.model_name, pretrained=cfg.pretrained).to(device)
    model_base = create_model(model_name=cfg.model_name, pretrained=cfg.pretrained, basic=True).to(device)
    model_cfg = ModelConfig.from_vit(model)

    # Data
    dataloader = get_xai_dataloader(
        output_dir=cfg.data_dir,
        transform=model.default_cfg["eval_transform"],
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        num_samples=cfg.num_samples,
        seed=cfg.seed,
    )

    # Evaluator
    eval_config = cfg.to_eval_config(model_cfg=model_cfg)

    method_kwargs = {
        "RISE": {"model_base": model_base, "n_masks": cfg.rise_mask, "mask_prob": cfg.rise_mask_prob},
        "SHAP": {"model_base": model_base, "n_masks": cfg.kernel_shap_mask, "ridge_alpha": cfg.kernel_shap_ridge_alpha},
    }

    evaluator = XAIEvaluator(
        methods=[(name, ALL_METHODS[name](model, **method_kwargs.get(name, {}))) for name in cfg.methods],
        model=model,
        model_base=model_base,
        config=eval_config,
    )

    results = evaluator.run(dataloader)
    XAIEvaluator.print_summary(results)

    if cfg.save_results:
        cfg.results_dir.mkdir(parents=True, exist_ok=True)

        def _stat(d, key, subkey):
            return d[key][subkey] if key in d else None

        summary = {
            name: {
                "morf": {
                    "mean_auc": _stat(data, "morf", "mean_auc"),
                    "std_auc": _stat(data, "morf", "std_auc"),
                    "mean_auc_30": _stat(data, "morf", "mean_auc_30"),
                    "std_auc_30": _stat(data, "morf", "std_auc_30"),
                },
                "lerf": {
                    "mean_auc": _stat(data, "lerf", "mean_auc"),
                    "std_auc": _stat(data, "lerf", "std_auc"),
                    "mean_auc_30": _stat(data, "lerf", "mean_auc_30"),
                    "std_auc_30": _stat(data, "lerf", "std_auc_30"),
                },
                "patch_shuffle": {
                    "mean_rho": _stat(data, "patch_shuffle", "mean_rho"),
                    "std_rho": _stat(data, "patch_shuffle", "std_rho"),
                },
                "patch_perturb_most_salient": {
                    "mean_rho": _stat(data, "patch_perturb_most_salient", "mean_rho"),
                    "std_rho": _stat(data, "patch_perturb_most_salient", "std_rho"),
                },
                "patch_perturb_least_salient": {
                    "mean_rho": _stat(data, "patch_perturb_least_salient", "mean_rho"),
                    "std_rho": _stat(data, "patch_perturb_least_salient", "std_rho"),
                },
                "patch_perturb_random": {
                    "mean_rho": _stat(data, "patch_perturb_random", "mean_rho"),
                    "std_rho": _stat(data, "patch_perturb_random", "std_rho"),
                },
                "pointing_game": {
                    "accuracy": _stat(data, "pointing_game", "accuracy"),
                    "hits": _stat(data, "pointing_game", "hits"),
                    "total": _stat(data, "pointing_game", "total"),
                },
            }
            for name, data in results["methods"].items()
        }

        summary["spearman"] = {
            pair: {
                "mean_rho": sp["mean_rho"],
                "std_rho": sp["std_rho"],
                "valid_count": sp["valid_count"],
            }
            for pair, sp in results["spearman"].items()
        }

        out_path = cfg.results_dir / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {out_path}")

        saliencies_stacked = {
            method_name: torch.stack(sal_list, dim=0)
            for method_name, sal_list in results["saliencies"].items()
        }
        saliency_path = cfg.results_dir / "saliencies.pt"
        torch.save(saliencies_stacked, saliency_path)
        print(f"Saliency maps saved to {saliency_path}")
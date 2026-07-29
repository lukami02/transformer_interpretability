from __future__ import annotations
import json
import torch
import numpy as np
from ..src.xai_eval.evaluate2 import XAIEvaluator2

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
    "GMARAttribution": GMARAttribution,
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

    saliences = torch.load('/kaggle/input/datasets/luxeljobless/slike-xai/saliencies_all.pt', map_location='cpu')


    evaluator = XAIEvaluator2(
        methods=[(name, ALL_METHODS[name](model, **method_kwargs.get(name, {}))) for name in saliences.keys()],
        model=model,
        model_base=model_base,
        config=eval_config,
        saliencies=saliences
    )

    evaluator = XAIEvaluator2(
        methods=[(name, ALL_METHODS[name](model, **method_kwargs.get(name, {}))) for name in saliences.keys() if name == "RISE"],
        model=model,
        model_base=model_base,
        config=eval_config,
        saliencies=saliences
    )

    results = evaluator.run(dataloader)

    if cfg.save_results:
        cfg.results_dir.mkdir(parents=True, exist_ok=True)

        def _stat(d, key, subkey):
            return d[key][subkey] if key in d else None
        
        def _stat(d, key, subkey):
            val = d[key][subkey] if key in d and subkey in d[key] else None
            if isinstance(val, np.ndarray):
                val = val.tolist()
            return val

        summary = {
            name: {
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
            }
            for name, data in results["methods"].items()
        }

        out_path = cfg.results_dir / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {out_path}")
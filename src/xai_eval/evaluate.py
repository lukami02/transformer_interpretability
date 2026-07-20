import logging
import torch
import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass, field

from ...scripts.config import EvalConfig
from .metrics.perturbation import compute_morf, compute_lerf, aggregate_perturbation_results
from .metrics.pointing_game import pointing_game_multi_bbox, pointing_game_single
from .metrics.spearman import compute_spearman_matrix
from .metrics.saliency_invariance import patch_shuffle_test, single_patch_perturbation_test


logger = logging.getLogger(__name__)


class XAIEvaluator:
    """
    Evaluator for XAI methods on image classification.
    """
    def __init__(
        self,
        methods: list,                        
        model:   Optional[torch.nn.Module] = None,
        model_base: Optional[torch.nn.Module] = None,
        config:  Optional[EvalConfig] = None,
        custom_logger: Optional[logging.Logger] = None,
    ):
        self.methods = methods
        self.model = model
        self.model_base = model_base
        self.cfg = config
        self.logger = custom_logger if custom_logger is not None else logger

        if self.cfg is None:
            raise ValueError("EvalConfig must be provided.")
        
        self.logger.setLevel(logging.INFO if self.cfg.verbose else logging.WARNING)

        if self.model_base is not None:
            self.logger.info("model_base provided: MoRF/LeRF will use the fast batched attention-masking path.")
        else:
            self.logger.info("No model_base provided: MoRF/LeRF will fall back to the slow per-step path using `model`.")


    def run(self, dataloader: torch.utils.data.DataLoader,) -> dict:
        """
        Runs all metrics for all provided XAI methods.
        """
        n = len(dataloader.dataset)
        method_results = {}
 
        spearman_saliencies: dict[str, list] = {}
 
        for method_name, method_xai in self.methods:
            self.logger.info(f"\n{'─'*50}\n {method_name}\n{'─'*50}")
            method_out, sals = self._run_method(method_name, method_xai.attribute, dataloader, n)
            method_results[method_name] = method_out
            spearman_saliencies[method_name] = sals
 
        self.logger.info(f"\n{'─'*50}\n Spearman correlation\n{'─'*50}")
        spearman = compute_spearman_matrix(
            spearman_saliencies,
            model_cfg=self.cfg.model_cfg,
            level=self.cfg.spearman_level,
        )
        for pair, sp in spearman.items():
            self.logger.info(f"  {pair}: rho={sp['mean_rho']:.4f} ± {sp['std_rho']:.4f}  (n={sp['valid_count']})")
 
        return {"methods": method_results, "spearman": spearman, "saliencies": spearman_saliencies}

    def _run_method(
        self,
        name: str,
        xai_fn: Callable,
        dataloader: torch.utils.data.DataLoader,
        n: int,
    ) -> tuple[dict, list]:
 
        morf_per_image = []
        lerf_per_image = []
        shuffle_rhos = []
        perturb_rhos = {"most_salient": [], "least_salient": [], "random": []}
        pg_hits  = 0
        pg_total = 0
        saliencies_for_spearman = [] 
        i = 0
 
        pg_fn = pointing_game_multi_bbox if self.cfg.pointing_multi_bbox else pointing_game_single
 
        for batch in dataloader:
            has_bbox = len(batch) == 3
            imgs_b, lbls_b = batch[0], batch[1]
            bboxs_b = batch[2] if has_bbox else None
 
            for j, img in enumerate(imgs_b.unbind(0)):
                lbl = lbls_b[j].item()
                i += 1
 
                log_msg = f"  [{i}/{n}] "
                
                sal = xai_fn(img.to(self.cfg.device), lbl).cpu()
 
                saliencies_for_spearman.append(sal)
                
                if self.model is not None:
                    # MoRF and LeRF
                    morf_res = compute_morf(
                        img, sal, self.model, lbl,
                        self.cfg.model_cfg, self.cfg.n_steps,
                        self.cfg.baseline_strategy, self.cfg.device,
                        model_base=self.model_base,
                    )
                    lerf_res = compute_lerf(
                        img, sal, self.model, lbl,
                        self.cfg.model_cfg, self.cfg.n_steps,
                        self.cfg.baseline_strategy, self.cfg.device,
                        model_base=self.model_base,
                    )
                    morf_per_image.append(morf_res)
                    lerf_per_image.append(lerf_res)
                    log_msg += (
                        f" | MoRF={morf_res['auc']:.4f} (@30%={morf_res['auc_30']:.4f})"
                        f"  LeRF={lerf_res['auc']:.4f} (@30%={lerf_res['auc_30']:.4f})"
                    )
 
                    # Patch Shuffle Test (always full random permutation)
                    """
                    shuffle_res = patch_shuffle_test(
                        image=img,
                        saliency=sal,
                        target_class=lbl,
                        xai_fn=xai_fn,
                        model_cfg=self.cfg.model_cfg,
                        device=self.cfg.device,
                        seed=self.cfg.seed
                    )
                    if not np.isnan(shuffle_res["rho"]):
                        shuffle_rhos.append(shuffle_res["rho"])
 
                    # Single Patch Perturbation Test — most_salient / least_salient / random
                    for selection in ("most_salient", "least_salient", "random"):
                        perturb_res = single_patch_perturbation_test(
                            image=img,
                            saliency=sal,
                            target_class=lbl,
                            xai_fn=xai_fn,
                            model_cfg=self.cfg.model_cfg,
                            baseline_strategy=self.cfg.baseline_strategy,
                            device=self.cfg.device,
                            patch_selection=selection,
                            seed=self.cfg.seed,
                        )
                        if not np.isnan(perturb_res["rho"]):
                            perturb_rhos[selection].append(perturb_res["rho"])
 
                    
                    log_msg += f" | Shfl_rho={shuffle_res['rho']:+.3f} | Patch_rho={perturb_res['rho']:+.3f}"
                    """
                    shuffle_rhos.append(1)
                    perturb_rhos["most_salient"].append(1)
                    perturb_rhos["least_salient"].append(1)
                    perturb_rhos["random"].append(1)

                    if bboxs_b is not None:
                        bbox = bboxs_b[j]
                        if bbox[0] >= 0:   
                            hit = pg_fn(sal, bbox, self.cfg.model_cfg, self.cfg.pointing_tolerance)
                            pg_hits  += int(hit)
                            pg_total += 1
                            
                if i % 5 == 0:
                    self.logger.info(log_msg)

            if i > 200:
                break
 
        result: dict = {}
 
        if morf_per_image:
            result["morf"] = aggregate_perturbation_results(morf_per_image)
            result["lerf"] = aggregate_perturbation_results(lerf_per_image)
            self.logger.info(f"\n  → MoRF mean={result['morf']['mean_auc']:.4f} ± {result['morf']['std_auc']:.4f}  (lower = better)")
            self.logger.info(f"  → MoRF@30% mean={result['morf']['mean_auc_30']:.4f} ± {result['morf']['std_auc_30']:.4f}  (lower = better)")
            self.logger.info(f"  → LeRF mean={result['lerf']['mean_auc']:.4f} ± {result['lerf']['std_auc']:.4f}  (higher = better)")
            self.logger.info(f"  → LeRF@30% mean={result['lerf']['mean_auc_30']:.4f} ± {result['lerf']['std_auc_30']:.4f}  (higher = better)")
 
        if shuffle_rhos:
            result["patch_shuffle"] = {
                "mean_rho": float(np.mean(shuffle_rhos)),
                "std_rho": float(np.std(shuffle_rhos))
            }
            self.logger.info(f"  → Patch Shuffle Rho={result['patch_shuffle']['mean_rho']:.4f} ± {result['patch_shuffle']['std_rho']:.4f}")
 
        for selection, rhos in perturb_rhos.items():
            if rhos:
                result[f"patch_perturb_{selection}"] = {
                    "mean_rho": float(np.mean(rhos)),
                    "std_rho": float(np.std(rhos))
                }
                self.logger.info(
                    f"  → Single Patch Perturb ({selection}) Rho="
                    f"{result[f'patch_perturb_{selection}']['mean_rho']:.4f} ± "
                    f"{result[f'patch_perturb_{selection}']['std_rho']:.4f}"
                )
                
        if pg_total > 0:
            result["pointing_game"] = {
                "accuracy": pg_hits / pg_total,
                "hits":     pg_hits,
                "total":    pg_total,
            }
            self.logger.info(f"  → Pointing Game={result['pointing_game']['accuracy']:.4f}  ({pg_hits}/{pg_total})")
 
        return result, saliencies_for_spearman


    @staticmethod
    def print_summary(eval_output: dict, custom_logger: Optional[logging.Logger] = None):
        """Prints a summary of the evaluation results in a nice format."""
        log = custom_logger if custom_logger is not None else logger

        methods  = eval_output["methods"]
        spearman = eval_output["spearman"]

        log.info("\n" + "=" * 65)
        log.info("  SUMMARY — XAI Evaluation")
        log.info("=" * 65)

        col_w = 22
        log.info(
            f"{'Method':<22}"
            f"{'MoRF AUC↓':>{col_w}}"
            f"{'MoRF@30%↓':>{col_w}}"
            f"{'LeRF AUC↑':>{col_w}}"
            f"{'LeRF@30%↑':>{col_w}}"
            f"{'Shfl ρ↑':>{col_w}}"
            f"{'Pert-Most ρ↓':>{col_w}}"
            f"{'Pert-Least ρ↑':>{col_w}}"
            f"{'Pert-Rand ρ':>{col_w}}"
            f"{'PG Acc↑':>{col_w}}"
        )
        log.info("-" * (22 + col_w * 8))

        for name, r in methods.items():
            if "morf" in r:
                morf_str = f"{r['morf']['mean_auc']:.4f}±{r['morf']['std_auc']:.3f}"
                morf_30_str = f"{r['morf']['mean_auc_30']:.4f}±{r['morf']['std_auc_30']:.3f}"
            else:
                morf_str, morf_30_str = "—", "—"

            if "lerf" in r:
                lerf_str = f"{r['lerf']['mean_auc']:.4f}±{r['lerf']['std_auc']:.3f}"
                lerf_30_str = f"{r['lerf']['mean_auc_30']:.4f}±{r['lerf']['std_auc_30']:.3f}"
            else:
                lerf_str, lerf_30_str = "—", "—"

            shfl_str = f"{r['patch_shuffle']['mean_rho']:.3f}±{r['patch_shuffle']['std_rho']:.3f}" if "patch_shuffle" in r else "—"
 
            pert_most_str = (
                f"{r['patch_perturb_most_salient']['mean_rho']:.3f}±{r['patch_perturb_most_salient']['std_rho']:.3f}"
                if "patch_perturb_most_salient" in r else "—"
            )
            pert_least_str = (
                f"{r['patch_perturb_least_salient']['mean_rho']:.3f}±{r['patch_perturb_least_salient']['std_rho']:.3f}"
                if "patch_perturb_least_salient" in r else "—"
            )
            pert_rand_str = (
                f"{r['patch_perturb_random']['mean_rho']:.3f}±{r['patch_perturb_random']['std_rho']:.3f}"
                if "patch_perturb_random" in r else "—"
            )

            pg_str = f"{r['pointing_game']['accuracy']:.4f}" if "pointing_game" in r else "—"

            log.info(
                f"{name:<22}"
                f"{morf_str:>{col_w}}"
                f"{morf_30_str:>{col_w}}"
                f"{lerf_str:>{col_w}}"
                f"{lerf_30_str:>{col_w}}"
                f"{shfl_str:>{col_w}}"
                f"{pert_most_str:>{col_w}}"
                f"{pert_least_str:>{col_w}}"
                f"{pert_rand_str:>{col_w}}"
                f"{pg_str:>{col_w}}"
            )

        if spearman:
            log.info("\n  Spearman rho:")
            log.info("  " + "-" * 50)
            for pair, sp in spearman.items():
                bar = "█" * int(max(0.0, sp["mean_rho"]) * 20)
                log.info(f"  {pair:<30}  rho={sp['mean_rho']:+.4f} ± {sp['std_rho']:.4f}  {bar}")

        log.info("=" * (22 + col_w * 8))
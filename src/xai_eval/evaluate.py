import logging
import torch
import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass, field

from ...scripts.config import EvalConfig
from .metrics.perturbation import compute_morf, compute_lerf, aggregate_perturbation_results
from .metrics.pointing_game import pointing_game_multi_bbox, pointing_game_single
from .metrics.spearman import compute_spearman_matrix


logger = logging.getLogger(__name__)


class XAIEvaluator:
    """
    Evaluator for XAI methods on image classification.
    """
    def __init__(
        self,
        methods: list,                        
        model:   Optional[torch.nn.Module] = None,
        config:  Optional[EvalConfig] = None,
        custom_logger: Optional[logging.Logger] = None,
    ):
        self.methods = methods
        self.model   = model
        self.cfg     = config
        self.logger = custom_logger if custom_logger is not None else logger

        if self.cfg is None:
            raise ValueError("EvalConfig must be provided.")
        
        self.logger.setLevel(logging.INFO if self.cfg.verbose else logging.WARNING)

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

        return {"methods": method_results, "spearman": spearman}


    def _run_method(
        self,
        name: str,
        xai_fn: Callable,
        dataloader: torch.utils.data.DataLoader,
        n: int,
    ) -> tuple[dict, list]:

        morf_per_image = []
        lerf_per_image = []
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
                i  += 1

                log_msg = f"  [{i}/{n}] "
                
                with torch.no_grad():
                    sal = xai_fn(img.to(self.cfg.device)).cpu()

                saliencies_for_spearman.append(sal)
                
                if self.model is not None:
                    morf_res = compute_morf(img, sal, self.model, lbl, 
                        self.cfg.model_cfg, self.cfg.n_steps, 
                        self.cfg.baseline_strategy, self.cfg.device
                    )
                    lerf_res = compute_lerf(img, sal, self.model, lbl, 
                        self.cfg.model_cfg, self.cfg.n_steps, 
                        self.cfg.baseline_strategy, self.cfg.device
                    )
                    morf_per_image.append(morf_res)
                    lerf_per_image.append(lerf_res)
                    log_msg += f" | MoRF={morf_res['auc']:.4f}  LeRF={lerf_res['auc']:.4f}"

                    if bboxs_b is not None:
                        bbox = bboxs_b[j]
                        if bbox[0] >= 0:   
                            hit = pg_fn(sal, bbox, self.cfg.model_cfg, self.cfg.pointing_tolerance)
                            pg_hits  += int(hit)
                            pg_total += 1
                            
                if i % 100 == 0:
                    self.logger.info(log_msg)

        result: dict = {}

        if morf_per_image:
            result["morf"] = aggregate_perturbation_results(morf_per_image)
            result["lerf"] = aggregate_perturbation_results(lerf_per_image)
            self.logger.info(f"\n  → MoRF mean={result['morf']['mean_auc']:.4f} ± {result['morf']['std_auc']:.4f}  (lower = better)")
            self.logger.info(f"  → LeRF mean={result['lerf']['mean_auc']:.4f} ± {result['lerf']['std_auc']:.4f}  (higher = better)")

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

        col_w = 16
        log.info(f"{'Metoda':<22}{'MoRF AUC↓':>{col_w}}{'LeRF AUC↑':>{col_w}}{'PG Acc↑':>{col_w}}")
        log.info("-" * (22 + col_w * 3))

        for name, r in methods.items():
            morf_str = f"{r['morf']['mean_auc']:.4f} ±{r['morf']['std_auc']:.3f}" if "morf" in r else "—"
            lerf_str = f"{r['lerf']['mean_auc']:.4f} ±{r['lerf']['std_auc']:.3f}" if "lerf" in r else "—"
            pg_str   = f"{r['pointing_game']['accuracy']:.4f}"                      if "pointing_game" in r else "—"
            log.info(f"{name:<22}{morf_str:>{col_w}}{lerf_str:>{col_w}}{pg_str:>{col_w}}")

        if spearman:
            log.info("\n  Spearman rho (saglasnost metoda):")
            log.info("  " + "-" * 50)
            for pair, sp in spearman.items():
                bar = "█" * int(max(0.0, sp["mean_rho"]) * 20)
                log.info(f"  {pair:<30}  rho={sp['mean_rho']:+.4f} ± {sp['std_rho']:.4f}  {bar}")

        log.info("=" * 65)
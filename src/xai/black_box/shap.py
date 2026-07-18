from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from ..base import XAIMethod

class KernelSHAP(XAIMethod):
    """
    KernelSHAP at patch level.
    """

    def __init__(
        self,
        model: nn.Module,
        model_base: Optional[nn.Module] = None,
        n_masks: int = 2000,
        batch_size: int = 1,
        ridge_alpha: float = 1e-3,
    ):
        super().__init__(model)
        self.model_base = model_base
        if model_base is not None:
            self.model_base.eval()

        self.n_masks = n_masks
        self.batch_size = batch_size
        self.num_patches = self.model.patch_embed.num_patches
        self.ridge_alpha = ridge_alpha

        self._masks: Optional[torch.Tensor] = None 
        self._weights: Optional[torch.Tensor] = None

        self.model.set_mode("shap")


    def _generate_masks(self) -> torch.Tensor:
        masks = torch.zeros(self.n_masks, self.num_patches)
        sizes = torch.randint(1, self.num_patches, (self.n_masks,))
        for i in range(2, self.n_masks):
            idx = torch.randperm(self.num_patches)[: sizes[i]]
            masks[i, idx] = 1.0

        masks[0] = 0.0
        masks[1] = 1.0
        return masks 


    def _kernel_weights(self) -> torch.Tensor:
        """
        Standard Shapley kernel weight:
            w(z) = (M - 1) / ( C(M, |z|) * |z| * (M - |z|) )
        """
        M = self._masks.shape[1]
        weight_table = torch.zeros(M + 1, dtype=torch.float32, device=self._masks.device)

        valid_sizes = torch.arange(1, M, dtype=torch.float32, device=self._masks.device)
        log_binom = (
            torch.lgamma(torch.tensor(float(M), device=self._masks.device) + 1)
            - torch.lgamma(valid_sizes + 1)
            - torch.lgamma(torch.tensor(float(M), device=self._masks.device) - valid_sizes + 1)
        )
        binom = torch.exp(log_binom)
        
        weight_table[1:M] = (M - 1) / (binom * valid_sizes * (M - valid_sizes))

        big = weight_table[1:M].max() * 1e6 if valid_sizes.any() else torch.tensor(1e6, device=self._masks.device)
        weight_table[0] = big
        weight_table[M] = big

        sizes = self._masks.sum(dim=1).long() 
        weights = weight_table[sizes]

        return weights


    def _solve_shapley_values(
        self,
        scores: torch.Tensor,
        f_full: float,
        f_empty: float,
    ) -> torch.Tensor:
        """
        Solves the weighted linear system to compute Shapley values.
        """
        M = self._masks.shape[1]
        y = (scores - f_empty).view(-1)  # (N,)

        sqrt_w = torch.sqrt(self._weights).view(-1)
        A = self._masks * sqrt_w.unsqueeze(1)              
        b = y * sqrt_w

        alpha = self.ridge_alpha
        AtA = A.T @ A + alpha * torch.eye(M, device=A.device)
        Atb = A.T @ b

        phi = torch.linalg.solve(AtA, Atb)  
        return phi


    def attribute(self, x: torch.Tensor, target: Optional[int] = None, **kwargs) -> torch.Tensor:
        if self.model._method_name != "shap":
            self.model.set_mode("shap")
            
        if self._masks is None:
            self._masks = self._generate_masks()
            self._weights = self._kernel_weights()
        X = self._prepare_input(x)

        logits = self._forward(x)
        if target is None:
            target = logits.argmax(dim=1).item()

        model = self.model_base if self.model_base is not None else self.model
        scores = []

        with torch.no_grad():
            for i in range(0, self._masks.shape[0], self.batch_size):
                masks = self._masks[i:i + self.batch_size].to(self.device)
                logits = model(X.expand(masks.shape[0], -1, -1, -1), masks)
                scores.append(logits[:, target].detach().cpu())

        scores = torch.cat(scores, dim=0).view(-1)

        f_empty = scores[0].item() 
        f_full = scores[1].item()

        saliency = self._solve_shapley_values(scores, f_full, f_empty)

        patch_embed = self.model.patch_embed
        grid_h = patch_embed.img_size[0] // patch_embed.patch_size[0]
        grid_w = patch_embed.img_size[1] // patch_embed.patch_size[1]

        saliency = saliency.view(1, grid_h, grid_w)
        return saliency
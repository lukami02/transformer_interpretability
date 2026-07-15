from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from base import XAIMethod

class RISE(XAIMethod):
    """
    RISE (Randomized Input Sampling for Explanation) at patch level.
    """
    def __init__(
        self,
        model: nn.Module,
        model_base: Optional[nn.Module] = None,
        n_masks: int = 2000,
        batch_size: int = 1,
        mask_prob: float = 0.5,
    ):
        super().__init__(model)
        self.model_base = model_base
        if model_base is not None:
            self.model_base.eval()

        self.n_masks = n_masks
        self.batch_size = batch_size
        self.num_patches = self.model.patch_embed.num_patches
        self.mask_prob = mask_prob

        self._masks: Optional[torch.Tensor] = None
        self.model.set_mode("rise")


    def _generate_masks(self) -> torch.Tensor:
        masks = (torch.rand(self.n_masks, self.num_patches) < self.mask_prob).float()
        return masks
    
    def attribute(self, x: torch.Tensor, target: Optional[int] = None, **kwargs) -> torch.Tensor:
        if self.model._method_name != "rise":
            self.model.set_mode("rise")

        if self._masks is None:
            self._masks = self._generate_masks()
        X = self._prepare_input(x)

        logits = self._forward(X)
        if target is None:
            target = logits.argmax(dim=1).item()

        model = self.model_base if self.model_base is not None else self.model
        scores = []

        with torch.no_grad():
            for i in range(0, self._masks.shape[0], self.batch_size):
                masks = self._masks[i:i + self.batch_size].to(self.device)
                logits = model(X.expand(masks.shape[0], -1, -1, -1), masks)
                scores.append(logits[:, target].detach().cpu())

        scores = torch.cat(scores, dim=0)

        weights = scores.view(-1, 1)
        saliency = (weights * self._masks).sum(dim=0) / (self._masks.sum(dim=0) + 1e-8)

        patch_embed = self.model.patch_embed
        grid_h = patch_embed.img_size[0] // patch_embed.patch_size[0]
        grid_w = patch_embed.img_size[1] // patch_embed.patch_size[1]

        saliency = saliency.view(1, grid_h, grid_w)
        return saliency
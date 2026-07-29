from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import XAIMethod

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


    def _generate_masks(self, grid_h: int, grid_w: int, low_res: int = 7) -> torch.Tensor:
        cell_h = grid_h // low_res
        cell_w = grid_w // low_res

        low_masks = (torch.rand(self.n_masks, 1, low_res + 1, low_res + 1) < self.mask_prob).float()

        up_h = (low_res + 1) * cell_h
        up_w = (low_res + 1) * cell_w
        up_masks = F.interpolate(low_masks, size=(up_h, up_w), mode='nearest')

        masks = torch.empty(self.n_masks, grid_h, grid_w, dtype=torch.bool)

        for i in range(self.n_masks):
            top = torch.randint(0, cell_h, (1,)).item()  
            left = torch.randint(0, cell_w, (1,)).item() 
            
            masks[i] = up_masks[i, 0, top:top + grid_h, left:left + grid_w].bool()
    
        return masks.view(self.n_masks, -1)

    
    def attribute(self, x: torch.Tensor, target: Optional[int] = None, **kwargs) -> torch.Tensor:
        if self.model._method_name != "rise":
            self.model.set_mode("rise")

        patch_embed = self.model.patch_embed
        grid_h = patch_embed.img_size[0] // patch_embed.patch_size[0]
        grid_w = patch_embed.img_size[1] // patch_embed.patch_size[1]
        
        if self._masks is None:
            self._masks = self._generate_masks(grid_h, grid_w, low_res=7)
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
                probs = torch.softmax(logits, dim=1)
                
                scores.append(probs[:, target].detach().cpu())

        scores = torch.cat(scores, dim=0)

        weights = scores.view(-1, 1)
        saliency = (weights * self._masks).sum(dim=0) / (self._masks.sum(dim=0) + 1e-8)

        saliency = saliency.view(1, grid_h, grid_w)
        return saliency
from typing import Optional
import torch
from ..base import XAIMethod

class LRP(XAIMethod):
    """
    Layer-wise Relevance Propagation (LRP).

    Backpropagates the target class score through the network using modified
    rules to produce a relevance map that highlights input features contributing
    to the prediction.
    """

    def __init__(self, model, alpha: float = 1.0):
        super().__init__(model)
        self.alpha = alpha
        self.model.set_mode("lrp")

    def attribute(self, x: torch.Tensor, target: Optional[int], patch_level: bool = True, **kwargs) -> torch.Tensor:
        if self.model._method_name != "lrp":
            self.model.set_mode("lrp")

        x = self._prepare_input(x)

        logits = self._forward(x)
        if target is None:
            target = logits.argmax(dim=1).item()
            
        R = torch.zeros_like(logits)
        R[0, target] = 1.0

        saliency = self.model.relprop(R, alpha=self.alpha, patch_level=patch_level)
        for block in self.model.blocks:
            block.attn._attn_relevance.clear()

        return saliency
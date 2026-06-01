from typing import Optional
import torch
from ..base import XAIMethod


class VanillaGradient(XAIMethod):
    """
    Vanilla Gradient (Saliency Map).
    Computes the gradient of the target class score with respect to the input.
    """
    def __init__(self, model):
        super().__init__(model)
        self.model.set_mode("grad")

    def attribute(self, x: torch.Tensor, target: Optional[int], **kwargs) -> torch.Tensor:
        if self.model._method_name != "grad":
            self.model.set_mode("grad")

        x = self._prepare_input(x, requires_grad=True)

        self._forward_backward(x, target)

        saliency = x.grad.abs().sum(dim=1)
        return saliency
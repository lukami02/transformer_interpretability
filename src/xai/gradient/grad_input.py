from typing import Optional
import torch
from ..base import XAIMethod


class GradientXInput(XAIMethod):
    """
    Gradient × Input.

    Element-wise product of the input and its gradient with respect to the
    target class score.
    """

    def __init__(self, model):
        super().__init__(model)
        self.model.set_mode("grad_input")

    def attribute(self, x: torch.Tensor, target: Optional[int], **kwargs) -> torch.Tensor:
        if self.model._method_name != "grad_input":
            self.model.set_mode("grad_input")

        x = self._prepare_input(x, requires_grad=True)

        self._forward_backward(x, target)
 
        saliency = (x.grad * x.detach()).abs().sum(dim=1)
        return saliency
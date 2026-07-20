from typing import Optional
import torch
from ..base import XAIMethod


class VanillaGradient(XAIMethod):
    """
    Vanilla Gradient (Saliency Map).
    Computes the gradient of the target class score with respect to the input.
    """
    def __init__(self, model, patch_level: bool = True, target_layer: int = 0):
        super().__init__(model)
        self.patch_level = patch_level
        self.target_layer = target_layer
        self.model.set_mode("grad")

    def attribute(self, x: torch.Tensor, target: Optional[int], **kwargs) -> torch.Tensor:
        if self.model._method_name != "grad":
            self.model.set_mode("grad")

        x = self._prepare_input(x, requires_grad=True)

        self._forward_backward(x, target)

        if self.patch_level:
            gradients = self.model.get_attn_gradients()[self.target_layer][0][0]
            gradients = gradients[:, 0, 1:]

            saliency = gradients.abs().mean(dim=0).clamp(min=0).unsqueeze(0)

            grid = int(saliency.shape[1] ** 0.5)
            saliency = saliency.reshape(grid, grid)
        else:
            saliency = x.grad.abs().mean(dim=1)

        return saliency
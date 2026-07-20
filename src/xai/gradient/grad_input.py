from typing import Optional
import torch
from ..base import XAIMethod


class GradientXInput(XAIMethod):
    """
    Gradient × Input.

    Element-wise product of the input and its gradient with respect to the
    target class score.
    """

    def __init__(self, model, patch_level: bool = True, target_layer: int = 0):
        super().__init__(model)
        self.patch_level = patch_level
        self.target_layer = target_layer
        self.model.set_mode("grad_input")

    def attribute(self, x: torch.Tensor, target: Optional[int], **kwargs) -> torch.Tensor:
        if self.model._method_name != "grad_input":
            self.model.set_mode("grad_input")

        x = self._prepare_input(x, requires_grad=True)

        self._forward_backward(x, target)
    
        if self.patch_level:
            activations = self.model.get_attn_weights()[self.target_layer][0][0]
            activations = activations[:, 0, 1:]

            gradients = self.model.get_attn_gradients()[self.target_layer][0][0]
            gradients = gradients[:, 0, 1:]

            saliency = (activations * gradients).abs().mean(0).clamp(0).unsqueeze(0)

            grid = int(saliency.shape[1] ** 0.5)
            saliency = saliency.reshape(grid, grid)
        else:
            saliency = (x.grad * x.detach()).abs().mean(dim=1)

        return saliency
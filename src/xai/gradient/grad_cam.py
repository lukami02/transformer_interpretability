from typing import Optional
import torch
from ..base import XAIMethod

class GradCAM(XAIMethod):
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping).

    Computes the gradient of the target class score with respect to the feature
    maps of a specified convolutional layer, and weights the feature maps by
    these gradients to produce a coarse localization map highlighting important
    regions in the input.
    """

    def __init__(self, model, target_layer: int = -1):
        super().__init__(model)
        self.target_layer = target_layer
        self.model.set_mode("grad_cam")

    def attribute(self, x: torch.Tensor, target: Optional[int], **kwargs) -> torch.Tensor:
        if self.model._method_name != "grad_cam":
            self.model.set_mode("grad_cam")

        x = self._prepare_input(x)
        self._forward_backward(x)

        activations = self.model.get_attn_weights()[self.target_layer][0][0]
        activations = activations[:, 0, 1:]

        gradients = self.model.get_attn_gradients()[self.target_layer][0][0]
        gradients = gradients[:, 0, 1:]
        weight = gradients.mean(dim=[1], keepdim=True)    

        cam = (activations * gradients).mean(0).clamp(0).unsqueeze(0)    

        grid = int(cam.shape[1] ** 0.5)
        saliency = cam.reshape(grid, grid)                 

        return saliency
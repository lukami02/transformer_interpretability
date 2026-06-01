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

    def __init__(self, model, target_layer: str = None):
        super().__init__(model)
        self.target_layer = target_layer
        self.model.set_mode("grad_cam")

    def attribute(self, x: torch.Tensor, target: Optional[int], **kwargs) -> torch.Tensor:
        if self.model._method_name != "grad_cam":
            self.model.set_mode("grad_cam")

        x = self._prepare_input(x)

        activations = self.model.get_layer_activations(self.target_layer)
        activations = activations[self.target_layer] 
        activations.requires_grad_(True)
    
        gradients = self._compute_gradient_wrt(x, target, wrt=activations, retain_graph=False)
        activations = activations[:, 1:, :]
        gradients = gradients[:, 1:, :]

        weights = gradients.mean(dim=1, keepdim=True)    # [B, 1, C]
        cam = (weights * activations).sum(dim=2)         # [B, N]
        cam = torch.relu(cam)                            

        B = cam.shape[0]
        grid = int(cam.shape[1] ** 0.5)
        saliency = cam.reshape(B, grid, grid)                     

        return saliency
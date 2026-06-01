from typing import Optional
import torch
from ..base import XAIMethod

class TransformerAttribution(XAIMethod):  
    """
    Transformer Attribution.
    "Transformer Interpretability Beyond Attention Visualization"

    Algorithm:
    1. Full forward pass — model stores attention weights
    2. Full LRP backward pass
    3. For each block l, compute:
           cam_l = mean_heads( relu( grad(score, A_l) * R_l ) )
    4. Rollout with residual blending starting from start_layer:
           C  =  I                          
           for l = start_layer … L:
               Ã_l = 0.5 * cam_l + 0.5 * I
               C   = Ã_l @ C
    """

    def __init__(self, model, start_layer: int = 0):
        super().__init__(model)
        self.start_layer = start_layer
        self.model.set_mode("transformer_attribution")
    
    def attribute(
        self, 
        x: torch.Tensor, 
        target: Optional[int] = None,
        alpha: float = 1.0, 
        **kwargs
    ) -> torch.Tensor:
        if self.model._method_name != "transformer_attribution":
            self.model.set_mode("transformer_attribution")

        X = self._prepare_input(x, requires_grad=True)

        logits = self._forward_backward(X, target, retain_graph=True)
        if target is None:
            target = logits.argmax(dim=1).item()

        one_hot = torch.zeros_like(logits)
        one_hot[0, target] = 1.0
        self.model.relprop(one_hot, alpha=alpha)

        num_patches = self.model.patch_embed.num_patches + 1

        attn_weights = self.model.get_attn_weights()
        attn_relevance = self.model.get_attn_relevance()
        attn_gradients = self.model.get_attn_gradients()

        cams = []
        for weight, gradients, relevances in zip(attn_weights, attn_gradients, attn_relevance):
            weight = weight[0][0]  
            gradients = gradients[0][0]  
            relevances = relevances[0][0]  

            cam = (gradients * relevances).clamp(min=0).mean(dim=0)
            cams.append(cam)

        C = torch.eye(num_patches, device=self.device)
        for cam in cams[self.start_layer:]:
            I = torch.eye(cam.size(-1), device=self.device)
            cam_blended = 0.5 * cam + 0.5 * I
            cam_blended = cam_blended / cam_blended.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            C = cam_blended @ C

        patch_embed = self.model.patch_embed
        grid_h = patch_embed.img_size[0] // patch_embed.patch_size[0]
        grid_w = patch_embed.img_size[1] // patch_embed.patch_size[1]

        saliency = C[0, 1:].reshape(1, grid_h, grid_w)

        return saliency




from typing import Optional
import torch
from ..base import XAIMethod

class GMARAttribution(XAIMethod):  
    """
    GMAR Attribution.
    "Transformer Interpretability Beyond Attention Visualization"

    Algorithm:
    1. Full forward pass — model stores attention weights
    2. Full LRP backward pass
    3. For each block l, compute per-head gradient weights (GMAR-style):
           GR_h = mean(|G_h|)          [l1]
           GR_h = sqrt(mean(G_h**2))   [l2]
           w_h  = GR_h / sum(GR_h)
       then combine heads with those weights instead of a plain mean:
           cam_l = sum_heads( w_h * relu( G_h * R_h ) )
    4. Rollout with residual blending starting from start_layer:
           C  =  I                          
           for l = start_layer … L:
               Ã_l = 0.5 * cam_l + 0.5 * I
               C   = Ã_l @ C
    """

    def __init__(
        self, 
        model, 
        start_layer: int = 0,
        norm: str = "l1",
    ):
        super().__init__(model)
        assert norm in ("l1", "l2"), f"Invalid normalization method: {norm}"
        self.start_layer = start_layer
        self.norm = norm
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
            weight = weight[0][0]        # [H, N, N]
            gradients = gradients[0][0]  # [H, N, N]
            relevances = relevances[0][0]  # [H, N, N]

            # GMAR-style per-head weighting
            if self.norm == "l1":
                GR = gradients.abs().mean(dim=(1, 2))
            else:
                GR = torch.sqrt((gradients**2).mean(dim=(1, 2)))
            w_h = GR / (GR.sum() + 1e-6)
            w_h = w_h.reshape(-1, 1, 1)  # [H, 1, 1]

            head_cams = (gradients * relevances).clamp(min=0)  # [H, N, N]
            cam = (head_cams * w_h).sum(dim=0)  # weighted combination instead of plain mean

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
from typing import Optional
import torch
from ..base import XAIMethod

class GMAR(XAIMethod):
    """
    Gradient-Driven Multi-head Attention Rollout (GMAR).

    Combines attention rollout with gradient weighting to produce sharper
    attribution maps that reflect both attention flow and gradient importance.
    
    Algorithm:
    1. Full forward pass — model stores attention weights
    2. Full backward pass
    3. Compute head weights:
        G_h = grad(score, A_h)
        L1: GR_h = mean(|G_h|)  per head 
        L2: GR_h = sqrt(mean(G_h**2)) per head
        w = GR_h / sum(GR_h) 
    4. Weighted attention rollout:
        A_weighted = (A_l @ W).sum(dim=0)
        A_rollout = A_rollout @ (A_weighted + alpha * I)
    """

    def __init__(
        self, 
        model, 
        start_layer: int = 0,
        norm: str = "l1",
        alpha: float = 0.5, 
    ):
        super().__init__(model)
        assert norm in ("l1", "l2"), f"Invalid normalization method: {norm}"
        self.start_layer = start_layer
        self.norm = norm
        self.alpha = alpha
        self.model.set_mode("gmar")

    def attribute(
        self, 
        x: torch.Tensor, 
        target: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        if self.model._method_name != "gmar":
            self.model.set_mode("gmar")

        X = self._prepare_input(x, requires_grad=True)

        self._forward_backward(X, target, retain_graph=False)
        
        attn_weights = self.model.get_attn_weights()
        attn_gradients = torch.stack(
            [grads[0].squeeze(0) for grads in self.model.get_attn_gradients()],
            dim=0
        ) # [L, H, N, N]

        if self.norm == "l1":
            GR = attn_gradients.abs().mean(dim=(2, 3))
        else:
            GR = torch.sqrt((attn_gradients**2).mean(dim=(2, 3)))

        w = GR / (GR.sum(dim=1, keepdim=True) + 1e-6)

        num_patches = self.model.patch_embed.num_patches + 1

        A_rollout = torch.eye(num_patches, device=self.device)

        for attn_weight, head_weights in zip(attn_weights[self.start_layer:], w[self.start_layer:]):
            attn_weight = attn_weight[0][0]  
            W = head_weights.reshape(-1, 1, 1)

            A_weighted = (attn_weight * W).sum(dim=0)
            A_weighted = A_weighted + self.alpha * torch.eye(num_patches, device=self.device)
            A_weighted = A_weighted / A_weighted.sum(dim=-1, keepdim=True).clamp(min=1e-9)

            A_rollout =  A_weighted @ A_rollout

        patch_embed = self.model.patch_embed
        grid_h = patch_embed.img_size[0] // patch_embed.patch_size[0]
        grid_w = patch_embed.img_size[1] // patch_embed.patch_size[1]

        saliency = A_rollout[0, 1:].reshape(1, grid_h, grid_w)

        return saliency

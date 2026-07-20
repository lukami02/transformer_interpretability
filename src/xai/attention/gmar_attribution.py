from typing import Optional
import torch
from ..base import XAIMethod


class GMARAttribution(XAIMethod):
    """
    Gradient-weighted Multi-head Attention Rollout with LRP relevance.

    Combines GMAR's gradient-based head weighting with LRP relevance
    propagation for the rollout step (instead of raw attention weights).

    Algorithm:
    1. Full forward pass — model stores attention weights
    2. Full LRP backward pass (gives per-head relevance R_l)
    3. Compute gradient-based head weights (as in GMAR):
        G_h = grad(score, A_h)
        L1: GR_h = mean(|G_h|)  per head
        L2: GR_h = sqrt(mean(G_h**2)) per head
        w = GR_h / sum(GR_h)
    4. Weighted rollout using LRP relevance instead of attention:
        R_weighted = (R_l @ W).sum(dim=0)
        R_weighted = relu(R_weighted) + alpha * I   (optional relu, see note)
        C = C @ (R_weighted / R_weighted.sum(-1))
    """

    def __init__(
        self,
        model,
        start_layer: int = 0,
        norm: str = "l1",
        alpha: float = 0.5,
        apply_relu: bool = True,
    ):
        super().__init__(model)
        assert norm in ("l1", "l2"), f"Invalid normalization method: {norm}"
        self.start_layer = start_layer
        self.norm = norm
        self.alpha = alpha
        self.apply_relu = apply_relu
        self.model.set_mode("gmar_attribution")

    def attribute(
        self,
        x: torch.Tensor,
        target: Optional[int] = None,
        alpha: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        if self.model._method_name != "gmar_attribution":
            self.model.set_mode("gmar_attribution")

        X = self._prepare_input(x, requires_grad=True)

        logits = self._forward_backward(X, target, retain_graph=True)
        if target is None:
            target = logits.argmax(dim=1).item()

        one_hot = torch.zeros_like(logits)
        one_hot[0, target] = 1.0
        self.model.relprop(one_hot, alpha=alpha)

        num_patches = self.model.patch_embed.num_patches + 1

        attn_relevance = self.model.get_attn_relevance()  
        attn_gradients_raw = self.model.get_attn_gradients()  

        attn_gradients = torch.stack(
            [grads[0].squeeze(0).detach() for grads in attn_gradients_raw],
            dim=0
        )

        if self.norm == "l1":
            GR = attn_gradients.abs().mean(dim=(2, 3))
        else:
            GR = torch.sqrt((attn_gradients ** 2).mean(dim=(2, 3)))

        w = GR / (GR.sum(dim=1, keepdim=True) + 1e-6)  # [L, H]

        C = torch.eye(num_patches, device=self.device)

        for layer_relevance, head_weights in zip(attn_relevance[self.start_layer:], w[self.start_layer:]):
            layer_relevance = layer_relevance[0][0].detach()
            W = head_weights.reshape(-1, 1, 1)

            R_weighted = (layer_relevance * W).sum(dim=0)

            if self.apply_relu:
                R_weighted = R_weighted.clamp(min=0)

            R_weighted = R_weighted + self.alpha * torch.eye(num_patches, device=self.device)
            R_weighted = R_weighted / R_weighted.sum(dim=-1, keepdim=True).clamp(min=1e-9)

            C = R_weighted @ C

        patch_embed = self.model.patch_embed
        grid_h = patch_embed.img_size[0] // patch_embed.patch_size[0]
        grid_w = patch_embed.img_size[1] // patch_embed.patch_size[1]

        saliency = C[0, 1:].reshape(1, grid_h, grid_w)

        return saliency
from typing import Optional
import torch
import torch.nn.functional as F
from ..base import XAIMethod


class AttentionRollout(XAIMethod):
    """
    Attention Rollout.

    Recursively multiplies attention maps across all transformer layers to
    account for residual connections and multi-layer information mixing.
    At each layer, the effective attention is:

        A_eff = 0.5 * A + 0.5 * I  

    Then rolled out as a product across layers:

        R_l = A_eff_l @ R_{l-1}
    """

    def __init__(self, model, discard_ratio: float = 0.9, head_fusion: str = "mean"):
        super().__init__(model)
        assert head_fusion in ("mean", "max", "min"), \
            f"head_fusion must be 'mean', 'max', or 'min', got '{head_fusion}'"
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion
        self.model.set_mode("attention_rollout")

    def attribute(self, x: torch.Tensor, target: Optional[int], **kwargs) -> torch.Tensor:
        if self.model._method_name != "attention_rollout":
            self.model.set_mode("attention_rollout")

        x = self._prepare_input(x)

        with torch.no_grad():
            self._forward(x)

        attn_weights = self.model.get_attn_weights()

        num_patches = self.model.patch_embed.num_patches + 1

        R = torch.eye(num_patches, device=self.device)  # [N, N]

        for attn in attn_weights:                                  
            attn = attn[0][0]                                  

            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=0)
            elif self.head_fusion == "max":
                attn_fused = attn.max(dim=0).values
            else:
                attn_fused = attn.min(dim=0).values

            # Discard lowest attention weights
            flat = attn_fused.view(-1)
            threshold = flat.kthvalue(int(flat.numel() * self.discard_ratio)).values
            attn_fused = attn_fused * (attn_fused > threshold).float()

            # Residual blending
            I = torch.eye(attn_fused.size(-1), device=self.device)
            a = (attn_fused + I) / 2
            a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-9)

            R = a @ R

        
        patch_embed = self.model.patch_embed
        grid_h = patch_embed.img_size[0] // patch_embed.patch_size[0]
        grid_w = patch_embed.img_size[1] // patch_embed.patch_size[1]

        saliency = R[0, 1:].reshape(1, grid_h, grid_w)                     

        return saliency      
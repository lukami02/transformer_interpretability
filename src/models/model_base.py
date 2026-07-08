import torch
import torch.nn as nn
from typing import Dict, Set, Tuple, Union, Optional, List


class PatchEmbed(nn.Module):
    """
    Patch Embedding layer that divides the input image into patches and projects them to a specified dimension.
    """
    def __init__(
        self, 
        img_size: Union[int, Tuple[int, int]] = 224, 
        patch_size: Union[int, Tuple[int, int]] = 16, 
        in_channels: int = 3, 
        dim: int = 768
    ):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=self.patch_size, stride=self.patch_size)    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}x{W}) does not match model expectations ({self.img_size[0]}x{self.img_size[1]})."
        
        # Output shape: [B, num_patches, dim]
        return self.proj(x).flatten(2).transpose(1, 2)
    

class Attention(nn.Module):
    """
    Standard Multi-Head Self-Attention module.
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        qkv_bias: bool = False, 
        attn_drop: float = 0.0, 
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        #Project to Q, K, V and reshape for multi-head processing: [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calculate scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            key_mask = mask[:, None, None, :]
            attn = attn.masked_fill(~key_mask, float("-inf"))

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Concatenate heads and project back to original dim
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    Multilayer Perceptron (Feed-Forward Network) inside Transformer block.
    """
    def __init__(
        self, 
        in_features: int, 
        hidden_features: Optional[int] = None, 
        out_features: Optional[int] = None, 
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class Block(nn.Module):
    """
    Transformer Encoder Block with explicit LRP tracking layers.
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0, 
        qkv_bias: bool = True, 
        drop_rate: float = 0.0, 
        attn_drop_rate: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop_rate, drop_rate)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, drop_rate)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1, x2 = x.clone(), x.clone()
        x = x1 + self.attn(self.norm1(x2), mask=mask)  # Attention block
        x1, x2 = x.clone(), x.clone()
        x = x1 + self.mlp(self.norm2(x2))   # MLP block
        return x

class VisionTransformerBase(nn.Module):
    """
    Vision Transformer model for image classification.
    """
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            mlp_head: bool = False,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Special class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate) 
            for _ in range(depth)
        ])

        # Normalization and classification head
        self.norm = nn.LayerNorm(embed_dim)
        if mlp_head:
            self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
        else:
            self.head = nn.Linear(embed_dim, num_classes)

        # Init weights
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        
        x = self.patch_embed(x)  # [B, num_patches, dim]

        # Expand class tokens across batch and prepend to sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, dim]

        # Inject positional embeddings
        x = x + self.pos_embed 

        full_mask = None
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            cls_visible = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_visible, mask], dim=1)  # [B, num_patches + 1]

        for block in self.blocks:
            x = block(x, mask=full_mask)
        
        x = self.norm(x)

        # Select the classification token
        x = x[:, 0, :]

        # Pass representation through final classification head
        x = self.head(x)
        return x

    @property
    def no_weight_decay(self) -> Set[str]:
        """Specifies parameters that should bypass weight decay regularization."""
        return {'pos_embed', 'cls_token'}
    
    def _init_weights(self, m: nn.Module) -> None:
        """Applies truncated normal weight initialization according to standard ViT specifications."""
        if m is self:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
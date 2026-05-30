import torch
import torch.nn as nn
from typing import Set, Tuple, Union, Optional
from .modules import *

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
        self.proj = Conv2d(in_channels, dim, kernel_size=self.patch_size, stride=self.patch_size)    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}x{W}) does not match model expectations ({self.img_size[0]}x{self.img_size[1]})."
        
        # Output shape: [B, num_patches, dim]
        return self.proj(x).flatten(2).transpose(1, 2)
    
    def relprop(self, R, **kwargs):
        R = R.transpose(1, 2)
        R = R.reshape(R.shape[0], 
                      R.shape[1], 
                      (self.img_size[0] // self.patch_size[0]), 
                      (self.img_size[1] // self.patch_size[1]))
        return self.proj.relprop(R, **kwargs)
    
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

        self.matmul1 = MatMul() 
        self.matmul2 = MatMul()

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        #Project to Q, K, V and reshape for multi-head processing: [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calculate scaled dot-product attention
        attn = self.matmul1([q, k.transpose(-2, -1)]) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Concatenate heads and project back to original dim
        x = self.matmul2([attn, v]).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def relprop(self, R, **kwargs):
        B, N, C = R.shape
        R = self.proj_drop.relprop(R, **kwargs)
        R = self.proj.relprop(R, **kwargs)
        R = R.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # x = A @ V
        (R_A, R_V) = self.matmul2.relprop(R, **kwargs)

        R_A = self.attn_drop.relprop(R_A, **kwargs)
        R_A = self.softmax.relprop(R_A, **kwargs)

        # A = Q @ K.T
        (R_Q, R_K) = self.matmul1.relprop(R_A, **kwargs)
        R_K = R_K.transpose(-2, -1) / 2

        R = torch.stack([R_Q, R_K, R_V], dim = 0).permute(1, 3, 0, 2, 4)
        R = torch.flatten(R, start_dim = 2)

        return self.qkv.relprop(R, **kwargs)


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

        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def relprop(self, R, **kwargs):
        R = self.drop.relprop(R, **kwargs)
        R = self.fc2.relprop(R, **kwargs)
        R = self.act.relprop(R, **kwargs)
        R = self.fc1.relprop(R, **kwargs)
        return R
    
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
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop_rate, drop_rate)
        self.norm2 = LayerNorm(dim, eps=1e-6)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, drop_rate)

        # Custom layers preserved for backward-compatibility and LRP propagation graph integrity
        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:       [B, num_patches + 1, dim] - input tensor (including class token)
        returns: [B, num_patches + 1, dim] - output tensor after transformer blocks
        """
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2))])  # Attention block
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])   # MLP block
        return x
    
    def relprop(self, R, **kwargs):
        (R_resid, R_mlp) = self.add2.relprop(R, **kwargs)
        R_mlp = self.mlp.relprop(R_mlp, **kwargs)
        R_mlp = self.norm2.relprop(R_mlp, **kwargs)
        R = self.clone2.relprop((R_resid, R_mlp), **kwargs)

        (R_resid, R_mlp) = self.add1.relprop(R, **kwargs)
        R_mlp = self.attn.relprop(R_mlp, **kwargs)
        R_mlp = self.norm1.relprop(R_mlp, **kwargs)
        R = self.clone1.relprop((R_resid, R_mlp), **kwargs) 
        return R

class VisionTransformer(nn.Module):
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
        
        # Core custom LRP pooling and routing layers
        self.add = Add()
        self.pool = IndexSelect()

        # Normalization and classification head
        self.norm = LayerNorm(embed_dim)
        if mlp_head:
            self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
        else:
            self.head = Linear(embed_dim, num_classes)

        # Init weights
        self.apply(self._init_weights)

        # Placeholder for input gradients 
        self.inp_grad = None

    @property
    def no_weight_decay(self) -> Set[str]:
        """Specifies parameters that should bypass weight decay regularization."""
        return {'pos_embed', 'cls_token'}
    
    def _save_inp_grad(self, grad: torch.Tensor) -> None:
        """Internal callback hooked to capture intermediate tensor gradients."""
        self.inp_grad = grad

    def get_input_grad(self) -> Optional[torch.Tensor]:
        """Retrieves stored input gradients after backward execution."""
        return self.inp_grad
    
    def _init_weights(self, m: nn.Module) -> None:
        """Applies truncated normal weight initialization according to standard ViT specifications."""
        if m is self:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        elif isinstance(m, Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, num_patches, dim]

        # Expand class tokens across batch and prepend to sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, dim]

        # Inject positional embeddings
        x = self.add([x, self.pos_embed]) 

        # Dynamic hook registration on the execution graph
        if x.requires_grad:
            x.register_hook(self._save_inp_grad) 

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)

        # Select the classification token
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))  

        # Pass representation through final classification head
        x = x.squeeze(dim=1)  
        x = self.head(x)
        return x

    def relprop(self, R, **kwargs):
        R = self.head.relprop(R, **kwargs)
        R = R.unsqueeze(dim=1)  
        R = self.pool.relprop(R, **kwargs)

        R = self.norm.relprop(R, **kwargs)

        for block in reversed(self.blocks):
            R = block.relprop(R, **kwargs)

        (R, _) = self.add.relprop(R, **kwargs)
        R = R[:, 1:]  # Remove the class token
        R = self.patch_embed.relprop(R, **kwargs)
        R = R.sum(dim=1)
        return R
    
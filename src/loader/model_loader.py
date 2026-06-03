import math
import logging
import torch
from torchvision import transforms
import torch.nn as nn
from typing import Callable, Dict, Any, Optional
from ..model import VisionTransformer

logger = logging.getLogger("WeightLoader")

_LRP_LAYER_SUFFIXES = ("add", "pool", "add1", "add2", "clone1", "clone2")

def _conv_filter(state_dict: Dict[str, torch.Tensor], patch_size: int = 16) -> Dict[str, torch.Tensor]:
    """
    Converts patch embedding weights from a flattened linear projection 
    to a 2D convolutional filter format.
    """
    out_dict = {}
    for key, value in state_dict.items():
        if 'patch_embed.proj.weight' in key:
            # Reshape from [embed_dim, in_chans * patch_H * patch_W] to [embed_dim, in_chans, patch_H, patch_W]
            value = value.reshape((value.shape[0], 3, patch_size, patch_size))
        out_dict[key] = value
    return out_dict

def load_pretrained_vit(
    model: nn.Module, 
    num_classes: int = 1000, 
    in_chans: int = 3, 
    strict: bool = False
) -> nn.Module:
    """
    Loads pretrained weights into the custom VisionTransformer model.
    """
    cfg = getattr(model, 'default_cfg', None)
    if cfg is None or not cfg['url']:
        logger.warning("Pretrained model URL is invalid or missing. Using random initialization.")
        return model

    logger.info(f"Loading weights from: {cfg['url']}")
    state_dict = torch.hub.load_state_dict_from_url(cfg['url'], map_location='cpu')
    state_dict = _conv_filter(state_dict, patch_size=model.patch_embed.patch_size[0])

    # Handle Input Channel Mismatches 
    conv1_weight_key = f"{cfg.get('first_conv', 'patch_embed.proj')}.weight"
    if conv1_weight_key in state_dict:
        conv1_weight = state_dict[conv1_weight_key]
        src_chans = conv1_weight.shape[1]

        if in_chans != src_chans:
            if in_chans == 1:
                logger.info("Converting first conv from %d channels to 1 (sum).", src_chans)
                state_dict[conv1_weight_key] = conv1_weight.sum(dim=1, keepdim=True)
            else:
                logger.info("Adapting first conv from %d to %d channels.", src_chans, in_chans)
                repeat = math.ceil(in_chans / src_chans)
                adapted = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans]
                adapted *= src_chans / float(in_chans)
                state_dict[conv1_weight_key] = adapted

    # Handle Classification Head Mismatches
    classifier = cfg.get('classifier', 'head')
    if num_classes != cfg.get('num_classes', 1000):
        logger.warning("Class count mismatch. Discarding pretrained head.")
        state_dict.pop(f"{classifier}.weight", None)
        state_dict.pop(f"{classifier}.bias", None)
        strict = False

    # Filter out incompatible keys 
    model_state = model.state_dict()
    state_dict = {
        k: v for k, v in state_dict.items()
        if k in model_state and v.shape == model_state[k].shape
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    structural_missing = [k for k in missing if not any(s in k for s in _LRP_LAYER_SUFFIXES)]
    if structural_missing:
        logger.warning("Missing structural keys: %s", structural_missing)
    if unexpected:
        logger.warning("Unexpected keys in checkpoint: %s", unexpected)

    logger.info("Pretrained weights loaded successfully.")
    return model


def vit_base_patch16_224(pretrained: bool = False, **kwargs: Any) -> VisionTransformer:
    """
    Instantiates a ViT-Base architecture with patch size 16 and resolution 224x224.
    """
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        **kwargs
    )

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    eval_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    model.default_cfg = {
        'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        'num_classes': 1000, 
        'input_size': (3, 224, 224),
        'first_conv': 'patch_embed.proj', 
        'classifier': 'head',
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'eval_transform': eval_transform,
    }
    
    if pretrained:
        model = load_pretrained_vit(
            model,
            num_classes=kwargs.get('num_classes', 1000),
            in_chans=kwargs.get('in_chans', 3),
        )
        
    return model

_MODEL_REGISTRY: Dict[str, Callable[..., VisionTransformer]] = {
    "vit_base_patch16_224": vit_base_patch16_224,
}

def create_model(model_name: str, pretrained: bool = True, **kwargs) -> VisionTransformer:
    """
    Factory function to create a model by name.
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry. Available models: {list(_MODEL_REGISTRY.keys())}")
    
    model_fn = _MODEL_REGISTRY[model_name]
    return model_fn(pretrained=pretrained, **kwargs)
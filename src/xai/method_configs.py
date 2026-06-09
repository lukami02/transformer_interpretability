from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class MethodConfig:
    """
    Describes exactly what the model needs to do during a forward pass
    to support a given XAI method.

    requires_grad
        Allow gradient flow for input tensors.

    lrp_hooks
        Attach RelProp forward hooks so that every LRP-capable module
        stores its input/output activations for the relprop() pass.

    cache_attn_weights
        Save post-softmax attention tensors inside each
        Attention module after every forward call.

    cache_attn_gradients
        Register a backward hook on the softmax attention tensors so that
        their gradients are stored alongside the weights.

    cache_layer_activations
        List of named modules  whose
        output tensors should be saved via forward hooks.
    """

    requires_grad: bool = False
    lrp_hooks: bool = False
    cache_attn_weights: bool = False
    cache_attn_gradients: bool = False
    cache_layer_activations: List[str] = field(default_factory=list)


METHOD_CONFIGS: dict[str, MethodConfig] = {

    # ------------------------------------------------------------------
    # Gradient-only methods
    # ------------------------------------------------------------------

    "grad": MethodConfig(
        requires_grad=True,
    ),

    "grad_input": MethodConfig(
        requires_grad=True,
    ),

    "integrated_grad": MethodConfig(
        requires_grad=True,
    ),

    "smooth_grad": MethodConfig(
        requires_grad=True,
    ),

    "grad_cam": MethodConfig(
        cache_attn_weights=True,
        cache_attn_gradients=True
    ),

    # ------------------------------------------------------------------
    # Attention Rollout
    # ------------------------------------------------------------------

    "attention_rollout": MethodConfig(
        cache_attn_weights=True,
    ),

    "transformer_attribution": MethodConfig(
        lrp_hooks=True,
        cache_attn_weights=True,
        cache_attn_gradients=True,
    ),
    
    "gmar": MethodConfig(
        cache_attn_weights=True,
        cache_attn_gradients=True,
    ),


    # ------------------------------------------------------------------
    # LRP
    # ------------------------------------------------------------------

    "lrp": MethodConfig(
        lrp_hooks=True,
    ),
}
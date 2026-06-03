from typing import List, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def safe_divide(a: Tensor, b: Tensor, eps: float = 1e-9) -> Tensor:
    """
    Stabilized division avoiding division-by-zero errors.
    """
    denom = torch.where(b >= 0, b.clamp(min=eps), b.clamp(max=-eps))
    return (a / denom) * (b != 0).to(a.dtype)


TensorOrList = Union[Tensor, List[Tensor], Tuple[Tensor, ...]]


# ---------------------------------------------------------------------------
# Base Classes
# ---------------------------------------------------------------------------

class RelProp(nn.Module):
    """
    Base class for Layer-wise Relevance Propagation (LRP).
    Registers a forward hook to capture inputs and outputs for use during
    the backward relevance propagation pass.
    """
    _lrp_enabled: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.X: TensorOrList = None
        self.Y: Tensor = None
        self._hook_handle = None
        if RelProp._lrp_enabled:
            self._attach_hook()

    @classmethod
    def set_lrp_mode(cls, enabled: bool) -> None:
        cls._lrp_enabled = enabled

    def _attach_hook(self) -> None:
        if self._hook_handle is None:
            self._hook_handle = self.register_forward_hook(self._forward_hook_fn)

    def _detach_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def apply_mode(self) -> None:
        """Sync hook registration with the current class-level flag."""
        if RelProp._lrp_enabled:
            self._attach_hook()
        else:
            self._detach_hook()

    def _forward_hook_fn(
        self,
        module: nn.Module,
        inputs: Tuple[TensorOrList, ...],
        output: Tensor,
    ) -> None:
        x = inputs[0]
        if isinstance(x, (list, tuple)):
            self.X = [i.detach().requires_grad_(True) for i in x]
        else:
            self.X = x.detach().requires_grad_(True)
        self.Y = output

    @staticmethod
    def gradprop(
        Z: TensorOrList,
        X: TensorOrList,
        S: TensorOrList,
    ) -> Tuple[Tensor, ...]:
        """Computes the localized vector-Jacobian product for relevance routing."""
        inputs = list(X) if isinstance(X, (list, tuple)) else [X]
        return torch.autograd.grad(Z, inputs, S, retain_graph=True)

    def relprop(self, R: Tensor, alpha: float) -> TensorOrList:
        return R

    def cleanup(self) -> None:
        self.X = None
        self.Y = None


class RelPropSimple(RelProp):
    """
    Generic LRP propagation for standard functional layers.
    Uses the basic LRP rule: R_i = x_i * (dZ/dx_i) * (R / Z)
    """

    def relprop(self, R: Tensor, alpha: float) -> TensorOrList:
        if self.X is None:
            raise RuntimeError("Forward pass must be executed before calling relprop.")

        Z = self.forward(self.X) if isinstance(self.X, Tensor) else self.forward(*self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if isinstance(self.X, Tensor):
            return self.X * C[0]

        return [x * c for x, c in zip(self.X, C)]


# ---------------------------------------------------------------------------
# Pass-through & Activation Wrappers
# ---------------------------------------------------------------------------

class ReLU(nn.ReLU, RelProp):
    pass


class GELU(nn.GELU, RelProp):
    pass


class Softmax(nn.Softmax, RelProp):
    pass


class LayerNorm(nn.LayerNorm, RelProp):
    pass


class Dropout(nn.Dropout, RelProp):
    pass


# ---------------------------------------------------------------------------
# Structural and Tensor Manipulation Modules
# ---------------------------------------------------------------------------

class Add(RelPropSimple):
    """
    LRP for element-wise addition.
    """

    def forward(self, inputs: List[Tensor]) -> Tensor:
        return torch.add(*inputs)

    def relprop(self, R: Tensor, alpha: float) -> List[Tensor]:
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        branches = [x * c for x, c in zip(self.X, C)]

        sums = torch.stack([b.sum() for b in branches])
        abs_sums = sums.abs()
        
        total_abs = abs_sums.sum()
        R_sum = R.sum()

        scales = safe_divide(abs_sums * R_sum, total_abs * sums)

        return [b * scale for b, scale in zip(branches, scales)]


class IndexSelect(RelProp):
    """
    LRP for torch.index_select.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dim: int = 0
        self.indices: Tensor = None

    def forward(self, inputs: Tensor, dim: int, indices: Tensor) -> Tensor:
        self.dim, self.indices = dim, indices
        return torch.index_select(inputs, dim, indices)

    def relprop(self, R: Tensor, alpha: float) -> TensorOrList:
        Z = self.forward(self.X, self.dim, self.indices)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if isinstance(self.X, Tensor):
            return self.X * C[0]
        return [x * c for x, c in zip(self.X, C)]


class Clone(RelProp):
    """
    LRP for tensor cloning.
    """

    def __init__(self) -> None:
        super().__init__()
        self.num: int = 1

    def forward(self, input: Tensor, num: int) -> List[Tensor]:
        self.num = num
        return [input] * num

    def relprop(self, R: List[Tensor], alpha: float) -> Tensor:
        return sum(R)


class Cat(RelProp):
    """
    LRP for torch.cat.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dim: int = 0

    def forward(self, inputs: List[Tensor], dim: int) -> Tensor:
        self.dim = dim
        return torch.cat(inputs, dim)

    def relprop(self, R: Tensor, alpha: float) -> List[Tensor]:
        Z = self.forward(self.X, self.dim)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)
        return [x * c for x, c in zip(self.X, C)]


class Sequential(nn.Sequential):
    """
    Sequential container with LRP support.
    Propagates relevance backwards through all child modules.
    """

    def relprop(self, R: Tensor, alpha: float) -> Tensor:
        for m in reversed(self._modules.values()):
            if hasattr(m, "relprop"):
                R = m.relprop(R, alpha)
        return R


# ---------------------------------------------------------------------------
# Parametric Modules — Alpha-Beta LRP Rule
# ---------------------------------------------------------------------------

class Linear(nn.Linear, RelProp):
    """
    LRP for fully-connected layers using the alpha-beta rule.

    The rule decomposes weight and activation into positive/negative parts:
        z_k^+ = conv(x^+, w^+) + conv(x^-, w^-)   (activator term)
        z_k^- = conv(x^+, w^-) + conv(x^-, w^+)   (inhibitor term)
    """

    def relprop(self, R: Tensor, alpha: float) -> Tensor:
        beta = alpha - 1.0
        pw = torch.clamp(self.weight, min=0.0)
        nw = torch.clamp(self.weight, max=0.0)
        px = torch.clamp(self.X, min=0.0)
        nx = torch.clamp(self.X, max=0.0)

        def f(w1: Tensor, w2: Tensor, x1: Tensor, x2: Tensor) -> Tensor:
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            # Shared denominator: total contribution of this sign
            S = safe_divide(R, Z1 + Z2)
            C1 = x1 * torch.autograd.grad(Z1, x1, S, retain_graph=True)[0]
            C2 = x2 * torch.autograd.grad(Z2, x2, S, retain_graph=True)[0]
            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)
        return alpha * activator_relevances - beta * inhibitor_relevances


class MatMul(RelPropSimple):
    """
    LRP for matrix multiplication using LRP.
    """

    def forward(self, *inputs):
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = inputs[0]
        return torch.matmul(inputs[0], inputs[1])
    
    def relprop(self, R: Tensor, alpha: float) -> List[Tensor]:
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        branches = [x * c for x, c in zip(self.X, C)]

        sum_branches = [b.sum() for b in branches]
        abs_sums = torch.stack([s.abs() for s in sum_branches])
        total_abs = abs_sums.sum()
        R_sum = R.sum()

        scales = [
            safe_divide(abs_s, total_abs) * safe_divide(R_sum, s)
            for abs_s, s in zip(abs_sums, sum_branches)
        ]

        return [b * scale for b, scale in zip(branches, scales)]


class Conv2d(nn.Conv2d, RelProp):
    """
    LRP for 2D convolution.

    Two rules are applied depending on the layer's position in the network:

    1) First layer: Deep Taylor Decomposition (DTD / zB rule).
       Uses pixel-wise lower (L) and upper (H) bounds to handle unbounded input space.

    2. Hidden layers: Alpha-beta rule.
       Same as Linear — positive/negative decomposition with shared denominator
       per sign group to ensure relevance conservation.
    """

    def _gradprop2(self, DY: Tensor, weight: Tensor) -> Tensor:
        """
        Transposed convolution used to propagate relevance signals back through
        the convolutional layer.
        """

        Z = self.forward(self.X)
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        kh, kw = self.kernel_size

        out_pad_h = self.X.size(2) - ((Z.size(2) - 1) * stride_h - 2 * pad_h + kh)
        out_pad_w = self.X.size(3) - ((Z.size(3) - 1) * stride_w - 2 * pad_w + kw)

        return F.conv_transpose2d(
            DY, weight, stride=self.stride, padding=self.padding, output_padding=(out_pad_h, out_pad_w)
        )


    def relprop(self, R: Tensor, alpha: float) -> Tensor:
        if self.X.shape[1] == 3:
            pw = torch.clamp(self.weight, min=0.0)
            nw = torch.clamp(self.weight, max=0.0)

            b = self.X.size(0)
            L = (
                self.X.view(b, -1).min(dim=1).values
                .view(b, 1, 1, 1)
                .expand_as(self.X)
            )
            H = (
                self.X.view(b, -1).max(dim=1).values
                .view(b, 1, 1, 1)
                .expand_as(self.X)
            )

            Za = (
                F.conv2d(self.X, self.weight, bias=None, stride=self.stride, padding=self.padding)
                - F.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding)
                - F.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding)
                + 1e-9
            )

            S = R / Za
            return (
                self.X * self._gradprop2(S, self.weight)
                - L * self._gradprop2(S, pw)
                - H * self._gradprop2(S, nw)
            )

        else:
            # Alpha-beta rule for hidden convolutional layers 
            beta = alpha - 1.0
            pw = torch.clamp(self.weight, min=0.0)
            nw = torch.clamp(self.weight, max=0.0)
            px = torch.clamp(self.X, min=0.0)
            nx = torch.clamp(self.X, max=0.0)

            def f(w1: Tensor, w2: Tensor, x1: Tensor, x2: Tensor) -> Tensor:
                Z1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding)
                Z2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding)
                
                S = safe_divide(R, Z1 + Z2)
                C1 = x1 * self.gradprop(Z1, x1, S)[0]
                C2 = x2 * self.gradprop(Z2, x2, S)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)
            return alpha * activator_relevances - beta * inhibitor_relevances
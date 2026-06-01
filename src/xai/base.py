from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn


class XAIMethod(ABC):
    """
    Base class for all XAI attribution methods.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.eval()
        
        self.device = next(model.parameters()).device

    @abstractmethod
    def attribute(self, x: torch.Tensor, target: Optional[int], **kwargs) -> torch.Tensor:
        pass

    def _prepare_input(self, x: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
        """
        Helper to ensure input tensor is on the correct device and has the right shape.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)  
        
        x = x.to(self.device)
        if requires_grad:
            x = x.clone().detach().requires_grad_(True)
        return x
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _forward_backward(
        self, 
        x: torch.Tensor, 
        target: Optional[int],
        retain_graph: bool = False
    ) -> torch.Tensor:
        logits = self.model(x)
        score = logits[0, target if target is not None else logits.argmax(dim=1).item()]
        score.backward(retain_graph=retain_graph)
        return logits

    def _compute_gradient_wrt(
        self, 
        x: torch.Tensor, 
        target: Optional[int],
        wrt: torch.Tensor,
        retain_graph: bool = False
    ) -> torch.Tensor:
        logits = self._forward(x)
        score = logits[0, target if target is not None else logits.argmax(dim=1).item()]
        (grad, ) = torch.autograd.grad(score, wrt, retain_graph=retain_graph)
        return grad
    
    @staticmethod
    def visualise(
        saliency: torch.Tensor, 
        upsample_size: tuple = None
    ) -> torch.Tensor:
        saliency = saliency.detach().cpu()
        
        if saliency.dim() == 4:
            saliency = torch.sum(torch.abs(saliency), dim=1, keepdim=True)
        elif saliency.dim() == 3:
            saliency = saliency.unsqueeze(1)
        else:
            saliency = saliency.unsqueeze(0).unsqueeze(0)
        
        saliency = torch.nn.functional.interpolate(
            saliency,  size=upsample_size, mode='bilinear', align_corners=False)

        saliency = saliency.squeeze(0)

        # Normalize to [0, 1]
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            
        return saliency

    def cleanup(self) -> None:
        if hasattr(self.model, "cleanup"):
            self.model.cleanup()
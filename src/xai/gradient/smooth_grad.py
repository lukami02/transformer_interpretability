from typing import Optional
import torch
from ..base import XAIMethod

class SmoothGrad(XAIMethod):
    """
    Smooth Grad

    Averages gradients over multiple noisy versions of the input to reduce noise in the attribution map.

    SG(x) = 1/N * sum_i [ grad f(x + noise_i) ]
    """ 

    def __init__(self, model, noise_lvl: float = 0.1, samples: int = 50):
        super().__init__(model)
        self.noise_lvl = noise_lvl
        self.samples = samples
        self.model.set_mode("smooth_grad")

    def attribute(self, x: torch.Tensor, target: Optional[int], **kwargs) -> torch.Tensor:
        if self.model._method_name != "smooth_grad":
            self.model.set_mode("smooth_grad")

        x = self._prepare_input(x)
        std = self.noise_lvl * (x.max() - x.min()).item()

        if target is None:
            with torch.no_grad():
                target = self._forward(x).argmax(dim=1).item()

        total_grad = torch.zeros_like(x)

        for _ in range(self.samples):
            noise = torch.randn_like(x) * std
            
            noisy_input = (x + noise).clone().detach()
            noisy_input.requires_grad_(True)

            logits = self.model(noisy_input)
            score = logits[0, target]

            self.model.zero_grad()
            score.backward()

            total_grad += noisy_input.grad.data

        avg_grad = total_grad / self.samples

        saliency = avg_grad.abs().sum(dim=1)
        return saliency

from typing import Optional
import torch
from ..base import XAIMethod


class IntegratedGradients(XAIMethod):
    """
    Integrated Gradients (IG).

    Attributes each input feature by integrating the gradient of the model
    output along a straight-line path from a baseline to the actual input.
    """

    def __init__(self, model, steps: int = 50):
        super().__init__(model)
        self.steps = steps
        self.model.set_mode("integrated_grad")

    def attribute(
        self,
        x: torch.Tensor,
        target: Optional[int],
        baseline: torch.Tensor | None = None,
        steps: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        if self.model._method_name != "integrated_grad":
            self.model.set_mode("integrated_grad")

        x = self._prepare_input(x)

        if target is None:
            with torch.no_grad():
                target = self._forward(x).argmax(dim=1).item()

        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = self._prepare_input(baseline)

        # Build interpolated inputs
        alphas = torch.linspace(0.0, 1.0, steps, device=self.device)
        delta = x - baseline                                           
        interpolated = baseline + alphas.view(-1, 1, 1, 1) * delta
        interpolated = interpolated.requires_grad_(True)

        # Forward pass over the whole batch of interpolated inputs
        logits = self.model(interpolated)            
        scores = logits[:, target].sum()             

        self.model.zero_grad()
        scores.backward()

        grads = interpolated.grad                   

        avg_grads = grads.mean(dim=0, keepdim=True) 
        saliency = (delta * avg_grads).abs().sum(dim=1)                   

        return saliency 
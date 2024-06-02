from typing import List, Optional
import torch
from darts.utils.likelihood_models import QuantileRegression


class PositiveQuantileRegression(QuantileRegression):
    """Qunatile Regession model that ensures all samples are positive by applying ReLU to the samples."""
    def __init__(self, quantiles: Optional[List[float]] = None):
        super().__init__(quantiles)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        # Call the parent class's sample method
        samples = super().sample(model_output)

        # Apply ReLU to ensure all samples are positive
        return torch.nn.functional.relu(samples)

    def simplified_name(self) -> str:
        return "positive_quantile"
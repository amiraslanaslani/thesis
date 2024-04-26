import torch
from torch.distributions.distribution import Distribution

class ConstantDistribution(Distribution):
    def __init__(
        self, 
        constant: float,
        validate_args=False
    ):
        super().__init__(validate_args=validate_args)
        self.constant_value = torch.tensor(constant)

    def rsample(self, sample_shape=torch.Size()):
        return torch.full(sample_shape, self.constant_value)


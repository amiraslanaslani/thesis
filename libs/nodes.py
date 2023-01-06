
from typing import Iterable, Optional, Tuple, Union
import torch
from bindsnet.network.nodes import LIFNodes


class NoisyLIFNode(LIFNodes):
    def __init__(
        self, 
        n: Optional[int] = None, 
        shape: Optional[Iterable[int]] = None, 
        traces: bool = False, 
        traces_additive: bool = False, 
        tc_trace: Union[float, torch.Tensor] = 20, 
        trace_scale: Union[float, torch.Tensor] = 1, 
        sum_input: bool = False, 
        thresh: Union[float, torch.Tensor] = -52,
        rest: Union[float, torch.Tensor] = -65, 
        reset: Union[float, torch.Tensor] = -65, 
        refrac: Union[int, torch.Tensor] = 5, 
        tc_decay: Union[float, torch.Tensor] = 100, 
        lbound: float = None, 
        noise: Tuple[float, float] = (0, 0),
        thresh_noise: float = 0,
        **kwargs
    ) -> None:
        if not isinstance(noise, tuple) and not isinstance(noise, list):
            noise = (noise, noise)
        self.noise = noise
        shape = (n, )
        threshold_noise_tensor = torch.rand(shape) * thresh_noise * 2 - thresh_noise
        threshold = torch.ones(shape) * thresh + threshold_noise_tensor
        super().__init__(
            n, 
            shape, 
            traces, 
            traces_additive, 
            tc_trace, 
            trace_scale, 
            sum_input, 
            threshold, 
            rest, 
            reset, 
            refrac, 
            tc_decay, 
            lbound, 
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> None:
        noise_1, noise_2 = [self.noise[0], self.noise[1]]
        if callable(noise_1):
            noise_1 = noise_1()
        if callable(noise_2):
            noise_2 = noise_2()
        self.v += torch.rand_like(self.v) * (noise_2 - noise_1) + noise_1
        return super().forward(x)

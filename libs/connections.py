from typing import Union, Optional, Sequence

import torch
from torch.nn import Parameter
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import Nodes


class ConnectionWithConvergence(Connection):
    def __init__(self, source: Nodes, target: Nodes, nu: Optional[Union[float, Sequence[float]]] = None, reduction: Optional[callable] = None, weight_decay: float = 0, **kwargs) -> None:
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)
        self.converge = 0.0 #Parameter(torch.tensor(0.0), requires_grad=False)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        self.converge = ((self.wmax - self.w) * (self.w - self.wmin)).sum()
        return super().compute(s)

class RandomConnection(Connection):
    def __init__(
        self, 
        source: Nodes, 
        target: Nodes, 
        nu: Optional[Union[float, Sequence[float]]] = None, 
        reduction: Optional[callable] = None, 
        weight_decay: float = 0.0, 
        connection_probability: float = 0.5,
        second_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> None:
        super().__init__(source, target, nu=nu, reduction=reduction, weight_decay=weight_decay, **kwargs)
        random_mask = (torch.rand(source.n, target.n) > connection_probability) 
        if second_mask is not None:
            random_mask |= second_mask
        self.random_mask = Parameter(random_mask, requires_grad=False)
        self.connection_number = (self.source.n * self.target.n) - random_mask.sum()
        # self.changes = torch.tensor(0)

        self.converge = torch.tensor(0)
        self.set_zero_connections()

    def update_converge(self):
        self.converge = ((self.wmin - self.w) * (self.wmax - self.w)).sum() / self.connection_number

    def update(self, **kwargs) -> None:
        super().update(**kwargs)
        self.set_zero_connections()
        self.update_converge()

    def set_zero_connections(self):
        self.w.masked_fill_(self.random_mask, 0)


class RandomConnectionWithInhibitory(RandomConnection):
    def __init__(
        self, 
        source: Nodes, 
        target: Nodes, 
        nu: Optional[Union[float, Sequence[float]]] = None, 
        reduction: Optional[callable] = None, 
        weight_decay: float = 0, 
        connection_probability: float = 0.5, 
        inhibitory: float = 0., # of source
        **kwargs
    ) -> None:
        super().__init__(source, target, nu=nu, reduction=reduction, weight_decay=weight_decay, connection_probability=connection_probability, **kwargs)
        self.inhibitory_n = int(source.n * inhibitory)
        self.w[:self.inhibitory_n,:] *= -1

    def update(self, **kwargs) -> None:
        super().update(**kwargs)
        # torch.clamp_(self.w[:self.inhibitory_n,:], min=self.wmin, max=0.)
        # torch.clamp_(self.w[self.inhibitory_n:,:], min=0., max=self.wmax)


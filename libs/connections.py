from math import floor
from typing import Tuple, Union, Optional, Sequence

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from bindsnet.network.topology import Connection, AbstractConnection
from bindsnet.network.nodes import Nodes


class ConnectionWithConvergence(Connection):
    def __init__(self, source: Nodes, target: Nodes, nu: Optional[Union[float, Sequence[float]]] = None, reduction: Optional[callable] = None, weight_decay: float = 0, **kwargs) -> None:
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)
        self.converge = 0.0 #Parameter(torch.tensor(0.0), requires_grad=False)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        self.converge = ((self.wmax - self.w) * (self.w - self.wmin)).sum()
        return super().compute(s)

class RandomConnection(ConnectionWithConvergence):
    def __init__(
        self, 
        source: Nodes, 
        target: Nodes, 
        nu: Optional[Union[float, Sequence[float]]] = None, 
        reduction: Optional[callable] = None, 
        weight_decay: float = 0.0, 
        probability: float = 1.0,
        second_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> None:
        super().__init__(source, target, nu=nu, reduction=reduction, weight_decay=weight_decay, **kwargs)
        random_mask = (torch.rand(source.n, target.n) > probability) 
        if second_mask is not None:
            random_mask |= second_mask
        self.random_mask = Parameter(random_mask, requires_grad=False)
        self.connection_number = (self.source.n * self.target.n) - random_mask.sum()
        self.set_zero_connections()

    def update(self, **kwargs) -> None:
        super().update(**kwargs)
        self.set_zero_connections()

    def set_zero_connections(self):
        self.w.masked_fill_(self.random_mask, 0)


# class RandomConnectionWithInhibitory(RandomConnection):
#     def __init__(
#         self, 
#         source: Nodes, 
#         target: Nodes, 
#         nu: Optional[Union[float, Sequence[float]]] = None, 
#         reduction: Optional[callable] = None, 
#         weight_decay: float = 0, 
#         connection_probability: float = 0.5, 
#         inhibitory: float = 0., # of source
#         **kwargs
#     ) -> None:
#         super().__init__(source, target, nu=nu, reduction=reduction, weight_decay=weight_decay, connection_probability=connection_probability, **kwargs)
#         self.inhibitory_n = int(source.n * inhibitory)
#         self.w[:self.inhibitory_n,:] *= -1

#     def update(self, **kwargs) -> None:
#         super().update(**kwargs)
#         # torch.clamp_(self.w[:self.inhibitory_n,:], min=self.wmin, max=0.)
#         # torch.clamp_(self.w[self.inhibitory_n:,:], min=0., max=self.wmax)


def get_output_size_maxpool1d(
    source: Union[int, Nodes],
    kernel_size: Union[int, Tuple[int]],
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    **kwargs
):
    source_size = source if isinstance(source, int) else source.n
    return floor((source_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


class MaxPool1dConnection(AbstractConnection):
    # language=rst
    """
    Specifies max-pooling synapses between one or two populations of neurons by keeping
    online estimates of maximally firing neurons.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        weight: int = 14.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a ``MaxPool1dConnection`` object.
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: the size of 1-D convolutional kernel.
        :param stride: stride for convolution.
        :param padding: padding for convolution.
        :param dilation: dilation for convolution.
        Keyword arguments:
        :param decay: Decay rate of online estimates of average firing activity.
        """
        super().__init__(source, target, None, None, 0.0, **kwargs)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.converge = torch.tensor([0.0])
        self.weight = weight

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute max-pool pre-activations given spikes using online firing rate
        estimates.
        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """


        x = F.max_pool1d(
            self.source.s.unsqueeze(0).float(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

        return x.view((self.target.n,)).float() * self.weight

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        # language=rst
        """
        No weights -> no normalization.
        """

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()

        self.firing_rates = torch.zeros(
            self.source.batch_size, *(self.source.s.shape[1:])
        )

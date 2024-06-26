from math import floor
from typing import Tuple, Union, Optional, Sequence

import torch
from torch.nn import Parameter
from torch.distributions.distribution import Distribution
from torch.distributions.uniform import Uniform
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
        normal_distr: float = 0.0, # true if greater than zero and the value is using to specify std.
        **kwargs
    ) -> None:
        if normal_distr > 0:
            std = normal_distr
            wmin = kwargs.get("wmin", None)
            wmax = kwargs.get("wmax", None)
            if wmin is None or wmax is None:
                mean = 0
            else:
                mean = (wmin + wmax) / 2
            kwargs["w"] = torch.normal(mean, std, size=(source.n, target.n))
            
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

class BackwardConnections(AbstractConnection):
    def __init__(
        self, 
        source: Nodes, 
        target: Nodes, 
        nu: Optional[Union[float, Sequence[float]]] = None, 
        reduction: Optional[callable] = None, 
        weight_decay: float = 0, 
        # potential_percent="random", # static:float | "random" | "random", min:float, max:float
        potential_percent: Union[Distribution, float] = Uniform(0, 1),
        connection_rate: float = 1.0,
        before_computation_delay: int = 0,
        after_computation_delay: int = 0,
        behavior: str = "exc", # exc, inh
        direct_voltage_manipulation: bool = False,
        enable: bool = True,
        **kwargs
    ) -> None:
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)
        self.connection_rate = connection_rate
        self.after_computation_delay = after_computation_delay
        self.before_computation_delay = before_computation_delay
        self.direct_voltage_manipulation = direct_voltage_manipulation
        self.enable = enable
        self.behavior = behavior

        if isinstance(potential_percent, (float, int)):
            from libs.utils import ConstantDistribution
            potential_percent = ConstantDistribution(constant=potential_percent)
        self.potential_percent = potential_percent

        if behavior == "exc":
            self.get_target_needed_potential = self._compute_exc_target_needed_potential
            self.weights_cooef = 1
        elif behavior == "inh":
            self.get_target_needed_potential = self._compute_inh_target_needed_potential
            self.weights_cooef = -1
        else:
            raise NotImplementedError(
                f"This type of behavior ({behavior}) is not valid/implemented."
            )

        self.w = torch.ones(source.n, target.n)
        self.mask = torch.rand(source.n, target.n) <= connection_rate
        self.w *= self.mask
        self.w *= self.weights_cooef * self.potential_percent.sample(self.w.size())

        if self.after_computation_delay > 0:
            self.after_computation_delay_window = torch.zeros((self.after_computation_delay + 2, self.target.n,))
        if self.before_computation_delay > 0:
            self.before_computation_delay_window = torch.zeros((self.before_computation_delay + 2, self.source.n,))

    def set_enable(self):
        self.enable = True

    def set_disable(self):
        self.enable = False

    def direct_manipulation(self, output_current):
        if self.direct_voltage_manipulation:
            self.target.v += (self.target.refrac_count == 0).float() * output_current
            return torch.zeros_like(output_current)
        else:
            return output_current

    def _compute_after_computation_delay_window(self, current_output) -> torch.Tensor:
        if self.after_computation_delay == 0:
            return current_output
        will_return  = self.after_computation_delay_window[-1].clone().detach()
        self.after_computation_delay_window[-1] = current_output
        self.after_computation_delay_window = torch.roll(self.after_computation_delay_window, shifts=1, dims=0)
        return will_return

    def _compute_before_computation_delay_window(self, spikes) -> torch.Tensor:
        if self.before_computation_delay == 0:
            return spikes
        will_return  = self.before_computation_delay_window[-1].clone().detach()
        self.before_computation_delay_window[-1] = spikes
        self.before_computation_delay_window = torch.roll(self.before_computation_delay_window, shifts=1, dims=0)
        return will_return

    def _get_commulative_percentage_picking(self, percentages, spikes):
        increase_percentage = 1 - torch.prod(1 - self.weights_cooef * self.w[spikes.bool().flatten()], axis=0)
        potentiations = percentages * increase_percentage
        return potentiations.view((self.target.n,))
    
    def _compute_exc_target_needed_potential(self) -> torch.Tensor:
        return self.target.thresh - self.target.v
    
    def _compute_inh_target_needed_potential(self) -> torch.Tensor:
        return self.target.rest - self.target.v

    def get_target_needed_potential(self):
        """
        At the begining, in __init__ function, this function will be setted to one of these functions:
        -    _compute_exc_target_needed_potential
        -    _compute_inh_target_needed_potential
        """
        pass
        
    def compute(self, s: torch.Tensor) -> None:
        if not self.enable:
            return torch.zeros((1, self.target.n))
        super().compute(s)
        spikes = self._compute_before_computation_delay_window(s)
        target_needed_potential = self.get_target_needed_potential()
        result = self._get_commulative_percentage_picking(target_needed_potential, spikes)
        result = self._compute_after_computation_delay_window(result)
        return self.direct_manipulation(result)

    def reset_state_variables(self) -> None:
        if self.after_computation_delay > 0:
            self.after_computation_delay_window = torch.zeros((self.after_computation_delay, self.target.n,))
        if self.before_computation_delay > 0:
            self.before_computation_delay_window = torch.zeros((self.before_computation_delay, self.target.n,))
        return super().reset_state_variables()

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        if not self.enable:
            return
        super().update(**kwargs)
        self.w *= self.mask

    def normalize(self) -> None:
        # language=rst
        """
        No weights -> no normalization.
        """

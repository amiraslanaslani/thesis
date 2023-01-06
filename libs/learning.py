from abc import ABC
from typing import Union, Optional, Sequence

import torch
from bindsnet.learning import LearningRule, MSTDPET, MSTDP
from bindsnet.network.topology import (
    AbstractConnection
)


class SoftlyBoundable():
    def alpha(self, soft=False):
        if soft:
            learning_rule: LearningRule = self
            return (learning_rule.connection.wmax - learning_rule.connection.w) * (learning_rule.connection.w - learning_rule.connection.wmin)
        return 1


class PostPreInh(LearningRule, SoftlyBoundable):

    def __init__(
        self, 
        connection: AbstractConnection, 
        nu: Optional[Union[float, Sequence[float]]] = None, 
        reduction: Optional[callable] = None, 
        weight_decay: float = 0, 
        windows_size: int = 30,
        windows_std: float = 4,
        soft_bound: bool = False,
        **kwargs
    ) -> None:

        super().__init__(
            connection, 
            nu, 
            reduction, 
            weight_decay, 
            **kwargs
        )
        self.soft_bound = soft_bound
        self.is_cuda = False
        self.windows_size = windows_size

        self.windows_positive = torch.zeros(
            self.source.n,
            self.windows_size
        )

        self.windows_negative = torch.zeros(
            self.target.n,
            self.windows_size
        )

        def g(x, c, m):   
            return torch.exp(-((x - m)*(x - m)) / (2*c*c))

        self.p_window = self.nu[1] * g(
            torch.arange(0, self.windows_size),
            windows_std,
            self.windows_size / 2
        )

        self.n_window = self.nu[0] * g(
            torch.arange(0, self.windows_size),
            windows_std,
            self.windows_size / 2
        )

    def get_changes(self) -> torch.Tensor:
        if not self.is_cuda and self.source.s.is_cuda:
            self.is_cuda = True
            self.windows_negative = self.windows_negative.cuda()
            self.windows_positive = self.windows_positive.cuda()
            self.p_window = self.p_window.cuda()
            self.n_window = self.n_window.cuda()

        self.windows_negative = torch.roll(self.windows_negative, -1, 1)
        self.windows_negative[:, -1] = 0

        self.windows_positive = torch.roll(self.windows_positive, -1, 1)
        self.windows_positive[:, -1] = 0

        self.windows_positive[self.source.s.bool()[0]] += self.p_window
        self.windows_negative[self.target.s.bool()[0]] += self.n_window

        changes = (self.target.s.view(self.target.n, 1).float() @ self.windows_positive[:, 0].view(1, self.source.n)).T - \
                  (self.source.s.view(self.source.n, 1).float() @ self.windows_negative[:, 0].view(1, self.target.n))
        return changes

    def update(self, **kwargs) -> None:
        changes = self.alpha(self.soft_bound) * self.get_changes()
        self.connection.w += changes
        self.connection.changes = changes.sum()
        super().update()


class PostPreWMatrix(LearningRule, SoftlyBoundable):
    def __init__(
        self, 
        connection: AbstractConnection, 
        nu: Optional[Union[float, Sequence[float]]] = None, 
        reduction: Optional[callable] = None, 
        weight_decay: float = 0, 
        soft_bound: bool = False,
        **kwargs
    ) -> None:
        super().__init__(connection, nu, reduction, weight_decay, **kwargs)
        self.soft_bound = soft_bound
    """
    Simple STDP rule involving both pre- and post-synaptic spiking activity. By default,
    pre-synaptic update is negative and the post-synaptic update is positive.
    """
    def update(self, **kwargs) -> None:
        batch_size = self.source.batch_size

        source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
        source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
        target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()
        target_x = self.target.x.view(batch_size, -1).unsqueeze(1)

        changes = torch.zeros_like(self.connection.w)
        # Pre-synaptic update.
        if self.nu[0]:
            update = self.reduction(torch.bmm(source_s, target_x), dim=0)
            changes -= self.nu[0] * update

        # Post-synaptic update.
        if self.nu[1]:
            update = self.reduction(torch.bmm(source_x, target_s), dim=0)
            changes += self.nu[1] * update

        self.connection.w += self.alpha(self.soft_bound) * self.changes
        super().update()


class AbstractSeasonalLearning(ABC):
    def trigger(self, reward: float):
        tmp = self.season_sum
        self.connection.w += self.season_sum * reward
        self.season_sum = torch.zeros(self.connection.source.n, self.connection.target.n)
        return tmp

class RSTDP_INH_SEASONAL(PostPreInh, AbstractSeasonalLearning, SoftlyBoundable):
    def __init__(
        self, 
        connection: AbstractConnection, 
        nu: Optional[Union[float, Sequence[float]]] = None, 
        reduction: Optional[callable] = None, 
        weight_decay: float = 0, 
        windows_size: int = 30, 
        windows_std: float = 4, 
        soft_bound: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            connection, 
            nu, 
            reduction, 
            weight_decay, 
            windows_size, 
            windows_std, 
            **kwargs
        )
        self.season_sum = torch.zeros(self.connection.source.n, self.connection.target.n)
        self.soft_bound = soft_bound

    def _connection_update(self, **kwargs) -> None:
        changes = self.alpha(self.soft_bound) * self.get_changes()
        self.season_sum += changes
        super().update()


class RSTDP_SEASONAL(PostPreWMatrix, AbstractSeasonalLearning, SoftlyBoundable):
    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        soft_bound: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )
        self.season_sum = torch.zeros(self.connection.source.n, self.connection.target.n)
        self.soft_bound = soft_bound

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
        source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
        target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()
        target_x = self.target.x.view(batch_size, -1).unsqueeze(1)

        changes = torch.zeros_like(self.season_sum)
        # Pre-synaptic update.
        if self.nu[0]:
            update = self.reduction(torch.bmm(source_s, target_x), dim=0)
            changes -= self.nu[0] * update

        # Post-synaptic update.
        if self.nu[1]:
            update = self.reduction(torch.bmm(source_x, target_s), dim=0)
            changes += self.nu[1] * update

        self.season_sum += self.alpha(self.soft_bound) * changes
        super().update()

class MSTDP_SEASONAL(MSTDP, AbstractSeasonalLearning, SoftlyBoundable):

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        soft_bound: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )
        self.season_sum = torch.zeros(self.connection.source.n, self.connection.target.n)
        self.soft_bound = soft_bound

    def _connection_update(self, **kwargs) -> None:
        batch_size = self.source.batch_size

        # Initialize eligibility, P^+, and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(batch_size, *self.source.shape)
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(batch_size, *self.target.shape)
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(batch_size, *self.connection.w.shape)

        # Reshape pre- and post-synaptic spikes.
        source_s = self.source.s.view(batch_size, -1).float()
        target_s = self.target.s.view(batch_size, -1).float()

        # Parse keyword arguments.
        a_plus = torch.tensor(kwargs.get("a_plus", 1.0))
        a_minus = torch.tensor(kwargs.get("a_minus", -1.0))

        # Compute weight update based on the eligibility value of the past timestep.
        self.season_sum += self.alpha(self.soft_bound) * self.nu[0] * self.reduction(self.eligibility, dim=0)

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(
            self.p_plus.unsqueeze(2), target_s.unsqueeze(1)
        ) + torch.bmm(source_s.unsqueeze(2), self.p_minus.unsqueeze(1))

        super().update()
        

class MSTDPET_SEASONAL(MSTDPET, AbstractSeasonalLearning, SoftlyBoundable):

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        soft_bound: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )
        self.season_sum = torch.zeros(self.connection.source.n, self.connection.target.n)
        self.soft_bound = soft_bound

    def _connection_update(self, **kwargs) -> None:
        # Initialize eligibility, eligibility trace, P^+, and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(self.source.n)
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(self.target.n)
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(*self.connection.w.shape)
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(*self.connection.w.shape)

        # Reshape pre- and post-synaptic spikes.
        source_s = self.source.s.view(-1).float()
        target_s = self.target.s.view(-1).float()

        # Parse keyword arguments.
        a_plus = torch.tensor(kwargs.get("a_plus", 1.0))
        a_minus = torch.tensor(kwargs.get("a_minus", -1.0))

        # Calculate value of eligibility trace based on the value
        # of the point eligibility value of the past timestep.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)
        self.eligibility_trace += self.eligibility / self.tc_e_trace

        # Compute weight update.
        self.season_sum += (
            self.alpha(self.soft_bound) * self.nu[0] * self.connection.dt * self.eligibility_trace
        )

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.ger(self.p_plus, target_s) + torch.ger(
            source_s, self.p_minus
        )

        super().update()

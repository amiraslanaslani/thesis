from abc import ABC
import random

import torch
from typing import Tuple, Union
from bindsnet.network.nodes import LIFNodes, Nodes, AdaptiveLIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network import Network
from bindsnet.learning.reward import AbstractReward
from bindsnet.network.monitors import Monitor

LAYER_23 = 1
LAYER_4 = 2

class AbstractConnectable(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.pops = []
        self.inpops = []
        self.outpops = []
        self.connections = []
        self.monitors = []
        self.submodules = []

    def add_to_network(self, network: Network):
        for name, pop in self.pops:
            network.add_layer(pop, name)
        for source, target, connection in self.connections:
            network.add_connection(connection, source, target)
        for name, monitor in self.monitors:
            network.add_monitor(monitor=monitor, name=name)
        for submodule in self.submodules:
            submodule.add_to_network(network)

    def get_input_pops(self):
        return self.inpops

    def get_output_pops(self):
        return self.inpops

    def classification(self) -> float:
        return 0

    def pops_reset_state_variables(self):
        for pop in self.pops:
            pop[1].reset_state_variables()


class LayerModule(AbstractConnectable):
    def __init__(
        self, 
        exc_size: int, 
        node_type=LIFNodes, 
        connection_type=Connection,
        exc_args={}, 
        exc_con_args={},
        inh_con_args={},
        exc_rec_con_args={},
        monitor: bool = False,
        name=f"layer{random.randint(0,9999)}",
    ):
        super().__init__()
        self.classifications = torch.tensor([])
        self.name = name
        self.classification_len = 10

        exc1 = node_type(n=exc_size, **exc_args)
        exc2 = node_type(n=exc_size, **exc_args)


        exc1_exc1 = connection_type(
            source=exc1, target=exc1 , **exc_rec_con_args
        )

        exc2_exc2 = connection_type(
            source=exc2, target=exc2 , **exc_rec_con_args
        )

        exc1_exc2_inh = connection_type(
            source=exc1, target=exc2 , **inh_con_args
        )
        exc2_exc1_inh = connection_type(
            source=exc2, target=exc1 , **inh_con_args
        )

        self.pops = [
            (f"{self.name}_exc1", exc1), 
            (f"{self.name}_exc2", exc2)
        ]
        self.inpops = [self.pops[0], self.pops[1]]
        self.outpops = [self.pops[0], self.pops[1]]
        self.connections = [
            (self.pops[0][0], self.pops[1][0], exc1_exc2_inh), 
            (self.pops[1][0], self.pops[0][0], exc2_exc1_inh),
            (self.pops[0][0], self.pops[0][0], exc1_exc1),
            (self.pops[1][0], self.pops[1][0], exc2_exc2),
        ]

        if monitor:
            for pop in self.pops:
                self.monitors.append((pop[0], Monitor(obj=pop[1], state_vars=['s'])))
            

    def moment_classification(self) -> float:
        activity1 = self.pops[0][1].s.sum()
        activity2 = self.pops[1][1].s.sum()
        return (activity1 - activity2) / self.pops[0][1].n

    def classification(self) -> float:
        # print(self.classifications)
        self.classifications = torch.cat((self.classifications, torch.tensor([self.moment_classification()])))
        if self.classifications.shape[0] > self.classification_len:
            self.classifications = self.classifications[1:]
        return torch.mean(self.classifications)


class AbstractLayerConnection(ABC):
    def __init__(self):
        super().__init__()
        self.connections = []

    def add_to_network(self, network: Network):
        for source, target, connection in self.connections:
            network.add_connection(connection, source, target)


class LayerConnection(AbstractLayerConnection):
    def __init__(
        self,
        source: Union[AbstractConnectable, Tuple[str, Nodes]],
        target,
        connection_type=Connection,
        connection_args={}
    ):
        super().__init__()

        source_list = source.get_output_pops() if isinstance(source, AbstractConnectable) else [source]
        target_list = target.get_input_pops() if isinstance(target, AbstractConnectable) else [target]
        for pop_source in source_list:
            for pop_target in target_list:
                self.connections.append(
                    (
                        pop_source[0], 
                        pop_target[0], 
                        connection_type(pop_source[1], pop_target[1], **connection_args)
                    )
                )


class CorticalColumn(AbstractConnectable):
    def __init__(
        self,
        connection_args,
        layer_args_23,
        layer_args_l4,
        connection_type=Connection,
        monitor=0,  # LAYER_23 | LAYER_4
        name=f"column{random.randint(0,9999)}",
    ):
        super().__init__()
        monitor = "{0:0>8b}".format(monitor)[::-1]
        self.l23 = LayerModule(**layer_args_23, name=f"{name}_l23_", monitor=monitor[0]=='1')
        self.l4 = LayerModule(**layer_args_l4, name=f"{name}_l4_", monitor=monitor[1]=='1')
        self.l4_l23 = LayerConnection(self.l4, self.l23, connection_type, connection_args)
        
        self.pops = [] + self.l23.pops + self.l4.pops
        self.inpops = [] + self.l4.get_input_pops()
        self.outpops = [] + self.l23.get_output_pops()
        self.connections = [] + self.l4_l23.connections + self.l23.connections + self.l4.connections
        self.submodules = [self.l23, self.l4]

    def classification(self) -> float:
        return self.l23.classification()


class AbstractRewardSystem(AbstractReward):
    def __init__(self):
        self.timestep = 0
        self.record = []
        
    def set_cortical_column_and_classes(self, cc: CorticalColumn, classes: torch.Tensor):
        self.cc = cc
        self.classes = classes
        
    def compute(self, **kwargs):
        reward_value = (1 if self.classes[self.timestep] else -1) * self.cc.classification()
        self.timestep += 1
        self.record.append(reward_value)
        return reward_value

    def update(self, **kwargs) -> None:
        pass

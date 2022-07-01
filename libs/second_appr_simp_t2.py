from abc import ABC
import random
from typing import Tuple, Union, List

import torch
from bindsnet.network.nodes import LIFNodes, Nodes
from bindsnet.network.topology import Connection, AbstractConnection
from bindsnet.network import Network
from bindsnet.learning.learning import NoOp
from bindsnet.network.monitors import Monitor

from libs.connections import MaxPool1dConnection


LAYER_23 = 1
LAYER_4 = 2


def load(file_name):
    return torch.load(file_name)


class AbstractConnectable(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.is_disabled_ever = False

        self.pops = []
        self.inpops = []
        self.outpops = []
        self.connections = []
        self.monitors: Tuple[str, Monitor] = []
        self.submodules: List[AbstractConnectable] = []

    def disable_learning(self, decay=False):
        if not self.is_disabled_ever:
            self.temp_update_rule = {}
            for source, target, connection in self.connections:
                self.temp_update_rule[(source, target)] = connection.update_rule
            self.is_disabled_ever = True

        for source, target, connection in self.connections:
            connection.update_rule = NoOp(
                connection, 
                connection.nu, 
                connection.reduction, 
                connection.weight_decay if decay else 0.0
            )
        for submodul in self.submodules:
            submodul.disable_learning()

    def enable_learning(self):
        if not self.is_disabled_ever:
            return
        for source, target, connection in self.connections:
            connection.update_rule = self.temp_update_rule[(source, target)]
        for submodul in self.submodules:
            submodul.enable_learning()

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
        return self.outpops

    def pops_reset_state_variables(self):
        for _, pop in self.pops:
            pop.reset_state_variables()
        for submodule in self.submodules:
            submodule.pops_reset_state_variables()

    def monitors_reset_state_variables(self):
        for _, monitor in self.monitors:
            if hasattr(monitor, 'network'):
                delattr(monitor, 'network')
            monitor.reset_state_variables()
        for submodule in self.submodules:
            submodule.monitors_reset_state_variables()

    def connections_reset_state_variables(self):
        for _, _, connection in self.connections:
            connection.reset_state_variables()
        for submodule in self.submodules:
            submodule.connections_reset_state_variables()

    def reset_state_variables(self):
        self.monitors_reset_state_variables()
        self.pops_reset_state_variables()
        self.connections_reset_state_variables()

    def save(self, file, reset=True):
        if reset:
            self.reset_state_variables()
        torch.save(self, file)


class ComplexStructure(AbstractConnectable):
    def __init__(self) -> None:
        super().__init__()

    def add_pop(self, name: str, pop: Nodes):
        self.pops.append((name, pop))

    def add_submodule(self, submodule: AbstractConnectable):
        self.submodules.append(submodule)

    def add_inpops(self, inpops_list: list):
        self.inpops += inpops_list

    def add_outpops(self, outpops_list: list):
        self.outpops += outpops_list

    def add_connection(self, source: str, target: str, connection: AbstractConnection):
        self.connections.append((source, target, connection))


class LayerModule(AbstractConnectable):
    def __init__(
        self, 
        exc_size: int, 
        node_type=LIFNodes, 
        connection_type=Connection,
        exc_args={}, 
        inh_con_args={},
        monitor: bool = False,
        name=f"layer{random.randint(0,9999)}",
    ):
        super().__init__()
        self.name = name

        exc1 = node_type(n=exc_size, **exc_args)
        exc2 = node_type(n=exc_size, **exc_args)

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
        ]

        if monitor:
            for pop in self.pops:
                self.monitors.append((pop[0], Monitor(obj=pop[1], state_vars=['s'])))
            

class LayerConnection(AbstractConnectable):
    def __init__(
        self,
        source: Union[AbstractConnectable, Tuple[str, Nodes]],
        target: Union[AbstractConnectable, Tuple[str, Nodes]],
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


class CorticalColumn(ComplexStructure):
    def __init__(
        self,
        connection_args,
        layer_args_23,
        layer_args_l4,
        monitor=0,  # LAYER_23 | LAYER_4
        name=f"column{random.randint(0,9999)}",
    ):
        super().__init__()
        monitor = "{0:0>8b}".format(monitor)[::-1]
        self.l23 = LayerModule(**layer_args_23, name=f"{name}_l23_", monitor=monitor[0]=='1')
        self.l4 = LayerModule(**layer_args_l4, name=f"{name}_l4_", monitor=monitor[1]=='1')
        con1 = MaxPool1dConnection(self.l4.pops[0][1], self.l23.pops[0][1], **connection_args)
        con2 = MaxPool1dConnection(self.l4.pops[1][1], self.l23.pops[1][1], **connection_args)

        self.add_connection(self.l4.pops[0][0], self.l23.pops[0][0], con1)
        self.add_connection(self.l4.pops[1][0], self.l23.pops[1][0], con2)

        self.add_submodule(self.l23)
        self.add_submodule(self.l4)

        self.add_inpops(self.l4.get_input_pops())
        self.add_outpops(self.l23.get_output_pops())
        
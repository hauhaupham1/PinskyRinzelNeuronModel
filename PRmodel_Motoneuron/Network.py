import diffrax
import equinox
import jax
from MotoneuronModel import MotoneuronModel


class Network:
    def __init__(self, num_neurons: int, yaml_file=None, param_set=None):
        self.neurons = [MotoneuronModel(yaml_file=yaml_file, param_set=param_set) for _ in range(num_neurons)]


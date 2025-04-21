# The Pinsky-Rinzel Neuron Model and Network

This is an implementation of the Pinsky-Rinzel model 
(see Pinsky and Rinzel 1994, 
[https://link.springer.com/article/10.1007/BF00962717](https://link.springer.com/article/10.1007/BF00962717)).
The code was used to produce some of the results presented
in Sætra et al. 2020: [An electrodiffusive, ion conserving Pinsky-Rinzel model with homeostatic mechanisms](https://doi.org/10.1371/journal.pcbi.1007661
).

## Project Components

This project includes multiple implementations of the Pinsky-Rinzel model:

1. **Original Implementation (`PRmodel/`)**: Uses SciPy's ODE solvers
2. **JAX/Diffrax Implementation (`PRmodel_Diffrax/`)**: Accelerated using JAX and Diffrax for automatic differentiation and GPU support
3. **Motoneuron Model (`PRmodel_Motoneuron/`)**: An extended version specialized for motoneurons
4. **Neural Network Implementation (`PRmodel_Motoneuron/Network.py`)**: A framework for creating networks of Pinsky-Rinzel neurons with synaptic connections

## Installation 

Clone or download the repo, navigate to the top directory of the repo and enter the following
command in the terminal: 
```bash
python setup.py install
```

### Requirements
- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- JAX (for the Diffrax implementation and network)
- Diffrax (for the Diffrax implementation and network)

## Usage Examples

### Original Pinsky-Rinzel Model
```python
from PRmodel import solve_PRmodel

# Parameters
t_dur = 1000.0
g_c = 15.0
I_stim = 0.5
stim_start = 200.0
stim_end = 400.0

# Run simulation
sol = solve_PRmodel(t_dur, g_c, I_stim, stim_start, stim_end)

# Plot results
import matplotlib.pyplot as plt
plt.plot(sol.t, sol.y[0], label='Vs')
plt.plot(sol.t, sol.y[1], label='Vd')
plt.legend()
plt.show()
```

### JAX/Diffrax Implementation
```python
from PRmodel_Diffrax import solve_PRmodel_diffrax

# Parameters
t_dur = 1000.0
g_c = 15.0
I_stim = 0.5
stim_start = 200.0
stim_end = 400.0

# Run simulation
sol = solve_PRmodel_diffrax(t_dur, g_c, I_stim, stim_start, stim_end)

# Plot results
import matplotlib.pyplot as plt
plt.plot(sol.ts, sol.ys[:, 0], label='Vs')
plt.plot(sol.ts, sol.ys[:, 1], label='Vd')
plt.legend()
plt.show()
```

### Using the Network Implementation
See the example scripts:
- `example_network.py`: Basic network example with different connection types
- `cpg_network_example.py`: Central Pattern Generator (CPG) network implementation

Basic usage of the network:
```python
from PRmodel_Motoneuron.Network import Network
import jax.numpy as jnp

# Create a network with 3 neurons
network = Network(
    n_neurons=3,
    yaml_file="motoneuron.yaml"
)

# Set up connections
network.set_connection(pre=0, post=1, weight=0.5, reversal=0.0)  # Excitatory: 0 -> 1
network.set_connection(pre=1, post=2, weight=0.5, reversal=-75.0)  # Inhibitory: 1 -> 2

# Run simulation
sol = network.solve(
    t_dur=500.0,
    I_stim=[0.7, 0.0, 0.5],  # Different current for each neuron
    stim_start=100.0,
    stim_end=400.0
)

# Analyze and visualize
results = network.extract_voltages(sol)
spike_times = network.get_spike_times(sol)
fig = network.plot_network(sol)
```

## Credits

The original code is from Marte Julie (https://github.com/CINPLA/PRmodel/blob/master/PRmodel/solve_PRmodel.py)

The code was further extended to include JAX/Diffrax implementations and network capabilities.

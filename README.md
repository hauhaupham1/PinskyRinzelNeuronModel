> ⚠️ **Note:** This repository has been merged into [livn](https://github.com/livn-org/livn) and is no longer actively maintained here.

# The Pinsky-Rinzel Neuron Model and Network

This is an implementation of the Pinsky-Rinzel model 
(see Pinsky and Rinzel 1994, 
[https://link.springer.com/article/10.1007/BF00962717](https://link.springer.com/article/10.1007/BF00962717)).
The code was used to produce some of the results presented
in Sætra et al. 2020: [An electrodiffusive, ion conserving Pinsky-Rinzel model with homeostatic mechanisms](https://doi.org/10.1371/journal.pcbi.1007661
).

This implementation extends the Pinsky-Rinzel model with adaptations for Motoneurons.
### Requirements
- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- JAX (for the Diffrax implementation and network)
- Diffrax (for the Diffrax implementation and network)

## Usage Examples
### Using the Network Implementation

Basic usage of the network:
```python
from PRmodel_Motoneuron.Network import Network
import jax.numpy as jnp

if __name__ == "__main__":
    # Define the network structure
    num_neurons = 5
    network = jnp.array([[0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]])
    weight = jnp.array([[0, 0.5, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]])
            

    t_dur = 1
    #I stim is array of shape (num_neurons,) with values of 1
    I_stim = jnp.ones(num_neurons) * 10.0
    stim_start = jnp.zeros(num_neurons)
    stim_end = jnp.ones(num_neurons) * 50.0
    
    # Initialize the network
    network_model = MotoneuronNetwork(
        num_neurons=num_neurons,
        network=network,
        w=weight,
        v_reset=-60.0,
        threshold=-37.0,
        input_current= I_stim,
        stim_start=stim_start,
        stim_end=stim_end,
    )

    sol = network_model(
        t0=0.0, 
        t1=100.0, 
        max_spikes=10, 
        num_samples=1, 
        key=jr.PRNGKey(0),
        dt0=0.01
    )

```

## Credits

The original code is from Marte Julie (https://github.com/CINPLA/PRmodel/blob/master/PRmodel/solve_PRmodel.py) and the code was further extended to include JAX/Diffrax implementations and network capabilities.

paths.py and solution.py are from Cholberg's implementation: (https://github.com/cholberg/snnax)


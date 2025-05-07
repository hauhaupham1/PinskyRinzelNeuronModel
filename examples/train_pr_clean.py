import os
import sys
import time
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

# Add the project root to the path so we can import the My_Network module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PRmodel_Motoneuron.Network import MotoneuronNetwork

# Create a simple network with 4 neurons
num_neurons = 4
network = jnp.array([
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0]
])

initial_weights = jnp.array([
    [0, 0.2, 0.2, 0.2],
    [0.2, 0, 0.2, 0.2],
    [0.2, 0.2, 0, 0.2],
    [0.2, 0.2, 0.2, 0]
])

# Initialize the model
model = MotoneuronNetwork(
    num_neurons=num_neurons,
    network=network,
    w=initial_weights,
    threshold=-20.0,
    v_reset=-60.0,
    diffusion=False
)

# Simple wrapper for creating an input current function from a tensor
class InputCurrentWrapper:
    def __init__(self, inputs_tensor):
        # inputs_tensor shape: [t_end, num_neurons]
        self.inputs = inputs_tensor
        self.max_t = inputs_tensor.shape[0]
    
    def __call__(self, t):
        # Convert time to index
        idx = jnp.clip(jnp.floor(t).astype(jnp.int32), 0, self.max_t - 1)
        return self.inputs[idx]

@eqx.filter_jit
def compute_loss(inputs, model, t_end, key):
    """Loss function that optimizes for mean somatic voltage."""
    # Create input current wrapper
    input_current_fn = InputCurrentWrapper(inputs)
    
    # Run simulation
    solution = model(
        input_current=input_current_fn,
        t0=0.0,
        t1=t_end,
        max_spikes=10,
        num_samples=1,
        key=key,
        dt0=0.01,
        num_save=20,
        max_steps=1500
    )
    
    # Get all somatic voltages (first component of state vector)
    somatic_voltages = solution.ys[0, :, :, :, 0]
    
    # Create a mask for finite values
    finite_mask = jnp.isfinite(somatic_voltages)
    
    # Replace non-finite values with zeros for computation
    masked_voltages = jnp.where(finite_mask, somatic_voltages, 0.0)
    
    # Compute mean over finite values only
    sum_voltages = jnp.sum(masked_voltages)
    count_finite = jnp.sum(finite_mask)
    
    # Avoid division by zero
    mean_voltage = sum_voltages / jnp.maximum(count_finite, 1)
    
    return mean_voltage

@eqx.filter_jit
def make_step(
    model,
    inputs,
    t_end,
    grad_loss,
    optim,
    opt_state,
    key,
):
    loss, grads = grad_loss(inputs, model, t_end, key)
    updates, opt_state = optim.update(grads, opt_state)
    inputs = eqx.apply_updates(inputs, updates)
    return inputs, loss, opt_state

# Initialize inputs tensor: [time, neurons]
t_end = 30.0  # 30 ms simulation
num_time_steps = 30  # 1 ms resolution
inputs = jnp.zeros((num_time_steps, num_neurons))

# Add initial stimulation pattern - stimulate all neurons at the start
inputs = inputs.at[:10, :].set(1.0)  # First 10ms, all neurons get 1.0 current

# optimizer
optim = optax.rmsprop(1e-2, momentum=0.3)  # Slightly higher learning rate
opt_state = optim.init(inputs)
grad_loss = eqx.filter_value_and_grad(compute_loss)

# key
key = jr.PRNGKey(42)

# optimization
num_iterations = 10  # Fewer iterations to demonstrate the concept

print("Starting optimization:")
for iteration in range(num_iterations):
    start_time = time.time()
    
    # Update key
    key, subkey = jr.split(key)
    
    #  optimization step
    inputs, loss, opt_state = make_step(
        model,
        inputs,
        t_end,
        grad_loss,
        optim,
        opt_state,
        subkey,
    )
    
    end_time = time.time()
    
    print(f"Step: {iteration}, Loss: {loss:.4f}, Computation time: {end_time - start_time:.4f}s")


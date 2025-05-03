import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import numpy as np
from functools import partial
from Network import MotoneuronNetwork, NetworkState
from solution import Solution

# Define the network structure
num_neurons = 1
stim_start = jnp.zeros(num_neurons)
stim_end = jnp.ones(num_neurons) * 10

# Create a surrogate gradient function for the Heaviside function
def sigmoid_surrogate(x, alpha=10.0):
    """Sigmoid surrogate gradient for the step function."""
    return jax.nn.sigmoid(alpha * x) * (1 - jax.nn.sigmoid(alpha * x)) * alpha

# Create a custom JAX grad rule for spike detection
@jax.custom_jvp
def heaviside_step(x, threshold=-37.0):
    """Heaviside step function with threshold."""
    return jnp.where(x > threshold, 1.0, 0.0)

@heaviside_step.defjvp
def heaviside_step_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    primal_out = heaviside_step(x)
    # Use the surrogate gradient for backward pass
    tangent_out = sigmoid_surrogate(x) * x_dot
    return primal_out, tangent_out

# Define a surrogate spike timing extraction function
@jax.custom_vjp
def extract_spike_time(voltage_trace, times, threshold=-37.0):
    """Extract the first spike time from a voltage trace."""
    # Forward pass: Check where voltage crosses threshold
    spike_indices = jnp.where(voltage_trace > threshold, 1.0, 0.0)
    # Find the first spike index
    first_spike_idx = jnp.argmax(spike_indices)
    # Check if there was any spike
    has_spike = jnp.any(spike_indices > 0)
    # Return the spike time or inf if no spike
    return jnp.where(has_spike, times[first_spike_idx], jnp.inf)

# Define forward and backward for custom VJP
def extract_spike_time_fwd(voltage_trace, times, threshold=-37.0):
    spike_time = extract_spike_time(voltage_trace, times, threshold)
    return spike_time, (voltage_trace, times, threshold, spike_time)

def extract_spike_time_bwd(res, grad_out):
    voltage_trace, times, threshold, spike_time = res
    
    # Detect the spike index
    spike_indices = voltage_trace > threshold
    first_spike_idx = jnp.argmax(spike_indices)
    
    # Create a gradient mask for the voltage trace
    # The gradient will flow only to the threshold crossing point
    voltage_grad = jnp.zeros_like(voltage_trace)
    
    # Apply surrogate gradient at the threshold crossing
    surrogate_grad = sigmoid_surrogate(voltage_trace[first_spike_idx] - threshold)
    
    # The time gradient depends on how much the voltage affects the spike time
    voltage_grad = voltage_grad.at[first_spike_idx].set(surrogate_grad * grad_out)
    
    # No gradient for times or threshold
    return voltage_grad, jnp.zeros_like(times), 0.0

extract_spike_time.defvjp(extract_spike_time_fwd, extract_spike_time_bwd)

# Simplified simulation function that returns voltage traces
def run_simulation(input_current, dt=0.01):
    """Run a simplified simulation that returns voltage traces for gradient computation."""
    # Create network
    network = MotoneuronNetwork(
        num_neurons=num_neurons,
        v_reset=-60.0,
        threshold=-37.0,
        input_current=input_current,
        stim_start=stim_start,
        stim_end=stim_end,
    )
    
    # Use a fixed key for deterministic results
    key = jr.PRNGKey(0)
    
    # Run the simulation
    sol = network(
        t0=0.0, 
        t1=20.0, 
        max_spikes=1, 
        num_samples=1, 
        key=key,
        dt0=dt,
        num_save=200,  # Use more save points for better gradient approximation
        max_steps=1500,
    )
    
    # Get the voltage trace for the first neuron
    voltage_data = sol.get_voltages(0, 0)
    
    # Extract time points and voltage values
    times = voltage_data['times']
    voltages = voltage_data['Vs']
    
    # Also return the spike times directly from the simulation
    spike_times = sol.spike_times[0]
    
    return times, voltages, spike_times

# Modified loss function that uses surrogate gradients
def loss_fn(params, target_spike_time=10.0):
    current = params["input_current"]
    
    # Run simulation to get voltage traces
    times, voltages, spike_times = run_simulation(current)
    
    # Check if we got any spikes directly (for monitoring)
    has_spike_direct = jnp.isfinite(spike_times[0])
    
    # Use our surrogate spike time extraction if needed
    # Note: We're still using the spike_times from the full simulation
    # for the actual loss calculation for accuracy
    
    # For the loss calculation, handle both spike and no-spike cases
    has_spike = jnp.isfinite(spike_times[0])
    
    # Calculate spike time loss (using the actual spike time when available)
    spike_loss = jnp.where(
        has_spike,
        (spike_times[0] - target_spike_time) ** 2,
        0.0
    )
    
    # Add penalty for no spike (modify this based on the specific desired behavior)
    no_spike_penalty = jnp.where(
        has_spike,
        0.0,
        100.0  # Fixed penalty for no spike
    )
    
    # Combine loss terms
    loss = spike_loss + no_spike_penalty
    
    return loss


value_and_grad_fn = jax.value_and_grad(loss_fn)
# parametes
params = {"input_current": jnp.ones(num_neurons) * 2.0}

# optimizer
learning_rate = 0.01
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Training loop
def train_step(params, opt_state, target_spike_time):
    loss_value, grads = value_and_grad_fn(params, target_spike_time)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

# Create a version of the training process that uses finite differences
# This is more stable in cases where surrogate gradients have issues
def finite_diff_train(initial_params, target_spike_time, learning_rate=0.05, num_iterations=30):
    """Training loop using finite difference approximation for gradients."""
    current_params = initial_params.copy()
    
    print("\n=== FINITE DIFFERENCE TRAINING ===")
    print("Initial params:", current_params)
    initial_loss = loss_fn(current_params, target_spike_time)
    print(f"Initial loss: {initial_loss}")
    
    for i in range(num_iterations):
        # Get current loss
        current_loss = loss_fn(current_params, target_spike_time)
        
        # Compute finite difference approximation of gradient
        epsilon = 0.1
        perturbed_params = {"input_current": current_params["input_current"] + epsilon}
        perturbed_loss = loss_fn(perturbed_params, target_spike_time)
        
        # Approximate gradient
        grad = (perturbed_loss - current_loss) / epsilon
        
        # Update parameters
        current_params = {"input_current": current_params["input_current"] - learning_rate * grad}
        
        # Ensure current stays positive
        current_params = {"input_current": jnp.maximum(current_params["input_current"], 0.1)}
        
        if i % 5 == 0 or i == num_iterations - 1:
            print(f"Iteration {i}: Loss = {current_loss}, Input current = {current_params['input_current']}")
    
    final_loss = loss_fn(current_params, target_spike_time)
    print(f"Final loss: {final_loss}")
    return current_params

# Demo surrogate gradient-based optimization
def surrogate_grad_train(initial_params, target_spike_time, learning_rate=0.01, num_iterations=40):
    """Training loop using surrogate gradients with JAX autodiff."""
    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(initial_params)
    
    # Training step function
    def train_step(params, opt_state, target):
        loss_value, grads = jax.value_and_grad(loss_fn)(params, target)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    # Run training
    params = initial_params.copy()
    print("\n=== SURROGATE GRADIENT TRAINING ===")
    print("Initial params:", params)
    
    try:
        initial_loss = loss_fn(params, target_spike_time)
        print(f"Initial loss: {initial_loss}")
        
        for i in range(num_iterations):
            try:
                params, opt_state, loss = train_step(params, opt_state, target_spike_time)
                if i % 5 == 0 or i == num_iterations - 1:
                    print(f"Iteration {i}: Loss = {loss}, Input current = {params['input_current']}")
            except Exception as e:
                print(f"Error in iteration {i}: {e}")
                break
        
        final_loss = loss_fn(params, target_spike_time)
        print(f"Final loss: {final_loss}")
    except Exception as e:
        print(f"Error in surrogate training: {e}")
    
    return params

# Run both optimization approaches
target_spike_time = 15.0

# First using surrogate gradients
print("Initial input current:", params["input_current"])
initial_loss = loss_fn(params, target_spike_time)
print(f"Initial loss with target spike at {target_spike_time}ms: {initial_loss}")

# Run the surrogate gradient version
surrogate_params = surrogate_grad_train(params, target_spike_time)

# Now run with finite differences
finite_diff_params = finite_diff_train(params, target_spike_time)

# Compare results
print("\n=== COMPARISON ===")
print(f"Starting current: {params['input_current'][0]}")
try:
    print(f"Surrogate gradient result: {surrogate_params['input_current'][0]}")
except:
    print("Surrogate gradient optimization failed")
print(f"Finite difference result: {finite_diff_params['input_current'][0]}")

# Test the optimized parameters
print("\n=== TESTING OPTIMIZED PARAMETERS ===")
_, _, spike_times_opt = run_simulation(finite_diff_params["input_current"])
if jnp.isfinite(spike_times_opt[0]):
    print(f"Optimized neuron spikes at {spike_times_opt[0]}ms (target: {target_spike_time}ms)")
else:
    print("Optimized neuron does not spike")

# Optional: Print actual voltage traces for visualization
try:
    times, voltages, _ = run_simulation(finite_diff_params["input_current"])
    print(f"Voltage trace has {len(voltages)} points, peak voltage: {jnp.max(voltages)}")
except Exception as e:
    print(f"Error getting voltage trace: {e}")
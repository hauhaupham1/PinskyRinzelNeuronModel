import os
import sys
import jax.numpy as jnp
import jax.random as jr
import jax
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PRmodel_Motoneuron.Network import MotoneuronNetwork
from PRmodel_Motoneuron.paths import marcus_lift
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True) 


#Lift

def compute_windowed_lift(
        spike_times: jnp.ndarray,
        spike_marks: jnp.ndarray,  # Changed name for clarity
        window_size = 10.0,
        total_duration = 100.0,
        ):
    num_windows = int(total_duration / window_size)
    lifts = []
    for i in range(1, num_windows + 1):
        window_end = i * window_size
        valid_idx = (spike_times > (i - 1) * window_size) & (spike_times <= window_end)
        window_spike_times = jnp.where(valid_idx, spike_times, jnp.inf)
        window_spike_marks = jnp.where(valid_idx.reshape(-1, 1), spike_marks, 0.0)

        lift = marcus_lift(
            t0 = (i - 1) * window_size,
            t1 = window_end,
            spike_times = window_spike_times,
            spike_mask = window_spike_marks, 
        )
        lifts.append(lift)
    return jnp.array(lifts)

def create_input_current(amplitude, total_duration=100.0, num_neurons=2):
    """Create a simple constant current function."""
    
    def current_fn(t):
        # Return current for single simulation (batching handled by Network)
        current = jnp.zeros((num_neurons,))
        
        # Apply constant current to both neurons
        current = current.at[0].set(jnp.where(t < total_duration, amplitude, 0.0))
        current = current.at[1].set(jnp.where(t < total_duration, amplitude * 0.8, 0.0))
        
        return current
    
    return current_fn

def generate_lift_sequence(
        num_simulations=100,
        num_neurons=2,
        window_size=10.0,
        total_duration=100.0,
        sequence_length=9,
        amplitude=3.0,  # Fixed amplitude - noise provides diversity
        key=jr.PRNGKey(0),
    ):
    model = MotoneuronNetwork(
        num_neurons=num_neurons,
        threshold=-37.0,
        v_reset=-60.0,
        diffusion=True,  # Enable noise for diversity
    )
    
    batch_size = 30
    num_batches = num_simulations // batch_size
    all_sequences = []
    all_targets = []
    
    for batch_idx in range(num_batches):
        batch_key = jr.fold_in(key, batch_idx)
        
        # Create simple constant current function
        input_current = create_input_current(
            amplitude, total_duration, num_neurons
        )
        
        # Run batch simulation
        batch_sol = model(
            input_current=input_current,
            t0=0.0,
            t1=total_duration,
            max_spikes=50,
            num_samples=batch_size,
            key=batch_key,
            spike_only=True
        )
        
        # Process each sample in the batch
        for i in range(batch_size):
            spike_times = batch_sol.spike_times[i]
            spike_marks = batch_sol.spike_marks[i] 
            
            # Compute windowed lifts
            lifts = compute_windowed_lift(
                spike_times, spike_marks, window_size, total_duration
            )
            
            # Create training pairs
            for start_idx in range(len(lifts) - sequence_length):
                sequence = lifts[start_idx:start_idx + sequence_length]
                target =   lifts[start_idx + sequence_length]
                all_sequences.append(sequence)
                all_targets.append(target)
    
    return jnp.array(all_sequences), jnp.array(all_targets)


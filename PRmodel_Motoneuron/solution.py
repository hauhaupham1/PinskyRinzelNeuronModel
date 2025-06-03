"""
Credits:
    - Solution's implementation by Christian Holberg
"""
from typing import Optional, Dict
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Real, PyTree
from typing import Literal, Optional
from typing_extensions import TypeAlias


class Solution(eqx.Module):
    """Solution of the event-driven model."""

    t1: Real
    ys: Float[Array, "samples spikes neurons times state_vars"]  # state_vars can be 1 (Vs) or more
    ts: Float[Array, "samples spikes times"]
    spike_times: Float[Array, "samples spikes"]
    spike_marks: Float[Array, "samples spikes neurons"]
    num_spikes: int
    max_spikes: int
    stored_vars: tuple = eqx.field(static=True, default=('Vs',))
    spike_only: bool = eqx.field(static=True, default=False)
    # For chunk continuity
    final_state: Optional[Float[Array, "samples neurons 8"]] = None
    final_synaptic_I: Optional[Float[Array, "samples neurons"]] = None
    # Solver continuity states
    solver_state: Optional[PyTree] = None
    controller_state: Optional[PyTree] = None
    made_jump: Optional[Bool[Array, "samples"]] = None

    def get_spikes(self, sample_idx=0):
        """
        Get the spike times for a specific sample.
        
        Args:
            sample_idx: Index of the sample to get spikes for
            
        Returns:
            Array of spike times for the sample
        """
        # Get the spike times for the sample
        spike_times = self.spike_times[sample_idx]
        
        # Filter out invalid (inf) spike times
        valid_spikes = jnp.isfinite(spike_times)
        return spike_times[valid_spikes]

    def get_state_trace(self, sample_idx: int = 0, neuron_idx: int = 0, state_name: str = "Vs") -> Dict[str, Array]:
        """
        Get the traces for a specific state variable (e.g., 'Vs' or 'Vd')
        for a specific sample and neuron.
        """
        # In spike-only mode, no state traces are available
        if self.spike_only:
            return {'times': jnp.array([], dtype=jnp.float32), state_name: jnp.array([], dtype=jnp.float32)}
            
        # Check if the requested state variable was stored
        if state_name not in self.stored_vars:
            return {'times': jnp.array([], dtype=jnp.float32), state_name: jnp.array([], dtype=jnp.float32)}
        
        # Find index of the state variable in stored variables
        state_idx = self.stored_vars.index(state_name)
        
        # Extract time points and state values
        all_times = []
        all_values = []
        
        # Number of segments (spike-to-spike) to process
        valid_spike_count = jnp.sum(jnp.isfinite(self.spike_times[sample_idx]))
        num_segments = min(valid_spike_count + 1, self.max_spikes)
        
        # Iterate through each segment
        for spike_idx in range(num_segments):
            times = self.ts[sample_idx, spike_idx]
            
            # Extract the requested state variable
            if len(self.stored_vars) == 1 and self.stored_vars[0] == state_name:
                # If we're only storing this one variable
                if self.ys.shape[-1] == 1:
                    values = self.ys[sample_idx, spike_idx, neuron_idx, :]
                    # Squeeze out the last dimension if it exists
                    if values.ndim > 1:
                        values = values.squeeze(-1)
                else:
                    values = self.ys[sample_idx, spike_idx, neuron_idx, :, 0]
            else:
                # Multiple variables stored, use the correct index
                values = self.ys[sample_idx, spike_idx, neuron_idx, :, state_idx]
            
            # Ensure times and values are 1D
            times = jnp.atleast_1d(times.squeeze())
            values = jnp.atleast_1d(values.squeeze())
            
            # Filter out invalid (inf) values
            valid_idx = jnp.isfinite(times) & jnp.isfinite(values)
            valid_times = times[valid_idx]
            valid_values = values[valid_idx]
            
            if len(valid_times) > 0:
                all_times.append(valid_times)
                all_values.append(valid_values)
        
        # Combine all segments
        if all_times:
            times_combined = jnp.concatenate(all_times)
            values_combined = jnp.concatenate(all_values)
            
            # Sort by time
            sort_idx = jnp.argsort(times_combined)
            times_sorted = times_combined[sort_idx]
            values_sorted = values_combined[sort_idx]
            
            return {
                'times': times_sorted,
                state_name: values_sorted
            }
        return {
            'times': jnp.array([], dtype=jnp.float32),
            state_name: jnp.array([], dtype=jnp.float32)
        }
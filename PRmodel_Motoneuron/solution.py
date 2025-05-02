import equinox as eqx
from jaxtyping import Array, Float, Real
import jax.numpy as jnp

class Solution(eqx.Module):
    """Solution of the event-driven model."""

    t1: Real
    ys: Float[Array, "samples spikes neurons times 8"]
    ts: Float[Array, "samples spikes times"]
    spike_times: Float[Array, "samples spikes"]
    spike_marks: Float[Array, "samples spikes neurons"]
    num_spikes: int
    max_spikes: int
    # synaptic_I: Float[Array, "sample neurons"]

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
        
    def get_voltages(self, sample_idx=0, neuron_idx=0):
        """
        Get the voltage traces for a specific sample and neuron.
        
        Args:
            sample_idx: Index of the sample to get voltages for
            neuron_idx: Index of the neuron to get voltages for
            
        Returns:
            Dictionary with 'times' and 'Vs' (somatic voltage) keys
        """
        # Extract time points and voltages
        all_times = []
        all_voltages = []
        
        # Number of segments (spike-to-spike) to process
        valid_spike_count = jnp.sum(jnp.isfinite(self.spike_times[sample_idx]))
        num_segments = min(valid_spike_count + 1, self.max_spikes)
        
        # Iterate through each segment
        for spike_idx in range(num_segments):
            times = self.ts[sample_idx, spike_idx]
            voltages = self.ys[sample_idx, spike_idx, neuron_idx, :, 0]  # Vs is at index 0
            
            # Filter out invalid (inf) values
            valid_idx = jnp.isfinite(times) & jnp.isfinite(voltages)
            valid_times = times[valid_idx]
            valid_voltages = voltages[valid_idx]
            
            if len(valid_times) > 0:
                all_times.append(valid_times)
                all_voltages.append(valid_voltages)
        
        # Combine all segments
        if all_times:
            times_combined = jnp.concatenate(all_times)
            voltages_combined = jnp.concatenate(all_voltages)
            
            # Sort by time
            sort_idx = jnp.argsort(times_combined)
            times_sorted = times_combined[sort_idx]
            voltages_sorted = voltages_combined[sort_idx]
            
            return {
                'times': times_sorted,
                'Vs': voltages_sorted
            }
        
        return {
            'times': jnp.array([]),
            'Vs': jnp.array([])
        }
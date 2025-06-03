"""
Pinsky-Rinzel Motoneuron Network Implementation

Credits:
    - buffers and _build_w are implemented by Christian Holberg
"""

from .network_state import NetworkState
from jax import random as jr
from typing import Sequence, Optional
from jaxtyping import Int
from .solution import Solution
import matplotlib.pyplot as plt


def buffers(state: NetworkState):
    assert type(state) is NetworkState
    return state.tevents, state.ts, state.ys, state.event_types

def _build_w(w, network, key, minval, maxval):
    if w is not None:
        return w
    w_a = jr.uniform(key, network.shape, minval=minval, maxval=maxval)
    return w_a.at[network].set(0.0)

def plot_simulation_results(
    sol: Solution,
    neurons_to_plot: Optional[Sequence[Int]] = None,
    plot_spikes: bool = True,
    plot_dendrite: bool = True  # Parameter to control Vd plotting
):
    """
    Plot voltage traces from a simulation solution.
    
    Args:
        sol: Solution object containing simulation results
        neurons_to_plot: Specific neurons to plot. If None, plots all neurons
        plot_spikes: Whether to mark spike times on the plot
        plot_dendrite: Whether to plot dendritic voltage traces (if available)
    """
    num_samples = sol.ys.shape[0]
    num_neurons = sol.ys.shape[2]  # Extract neuron count from solution
    
    # Check if dendrite plotting is possible
    if plot_dendrite and hasattr(sol, 'stored_vars'):
        if 'Vd' not in sol.stored_vars:
            plot_dendrite = False
            print("Warning: Dendritic voltage (Vd) not stored, dendrite plotting disabled")
    if neurons_to_plot is None:
        neurons_to_plot = range(num_neurons)
    elif not isinstance(neurons_to_plot, Sequence) or not all(isinstance(n, int) for n in neurons_to_plot):
        raise ValueError("neurons_to_plot must be a sequence of integers or None.")

    for sample_idx in range(num_samples):
        plt.figure(figsize=(12, 6))
        print(f"Processing plot for Sample {sample_idx}...")
        plotted_lines = []
        plotted_labels = []
        # Iterate through the neurons requested for plotting
        for neuron_index in neurons_to_plot:
            if neuron_index < 0 or neuron_index >= num_neurons:
                print(f"Warning: Neuron index {neuron_index} out of range (0-{num_neurons-1}). Skipping.")
                continue

            # Get somatic voltage using the Solution's method
            v_data = sol.get_state_trace(sample_idx, neuron_index, 'Vs')
            
            if len(v_data['times']) > 0:
                # Plot soma voltage
                line_soma, = plt.plot(v_data['times'], v_data['Vs'], 
                                     label=f"Neuron {neuron_index} (Soma)",
                                     color=f"C{neuron_index}")
                plotted_lines.append(line_soma)
                plotted_labels.append(f"Neuron {neuron_index} (Soma)")
                
                # Plot dendrite voltage if requested
                if plot_dendrite:
                    # Get dendritic voltage using the Solution's method
                    vd_data = sol.get_state_trace(sample_idx, neuron_index, 'Vd')
                    if len(vd_data['times']) > 0:
                        line_dend, = plt.plot(vd_data['times'], vd_data['Vd'],
                                            label=f"Neuron {neuron_index} (Dendrite)",
                                            linestyle='--', color=f"C{neuron_index}")
                        plotted_lines.append(line_dend)
                        plotted_labels.append(f"Neuron {neuron_index} (Dendrite)")
            else:
                print(f"No valid data found for Neuron {neuron_index} in Sample {sample_idx}.")

        # Plot spike markers
        if plot_spikes:
            # Use get_spikes method to get valid spike times
            valid_spike_times = sol.get_spikes(sample_idx)
            added_spike_legend = False
            if len(plotted_lines) > 0:
                min_t_plot, max_t_plot = plt.xlim() # Get current plot time range
                for spike_t in valid_spike_times:
                    if spike_t >= min_t_plot and spike_t <= max_t_plot: # Ensure spike is within plot range
                        label = 'Spike' if not added_spike_legend else ""
                        plt.axvline(spike_t, color='r', linestyle='--', alpha=0.6, label=label)
                        added_spike_legend = True

        # Finalize plot for the current sample
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.title(f"Sample {sample_idx}: Neuron Voltage Traces")
        
        if plotted_lines:
            custom_legend_handles = plotted_lines
            custom_legend_labels = plotted_labels
            if plot_spikes and added_spike_legend: # Only add spike to legend if it was plotted
                from matplotlib.lines import Line2D
                spike_legend_entry = Line2D([0], [0], color='r', linestyle='--', alpha=0.6, label='Spike')
                custom_legend_handles.append(spike_legend_entry)
                custom_legend_labels.append('Spike')

            plt.legend(handles=custom_legend_handles, labels=custom_legend_labels)

        plt.grid(True)
        plt.show() # Show the plot for the current sample
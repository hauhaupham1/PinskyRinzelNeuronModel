import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp


def visualize_predictions(
    actual_spikes,
    predicted_rates,
    sample_idx=0,
    neuron_ids=[0, 20, 40, 60, 80],
    save_path=None,
):
    """
    Visualize actual spikes vs predicted rates using line plots for selected neurons

    Args:
        actual_spikes: Shape (batch_size, time_bins, neurons)
        predicted_rates: Shape (batch_size, time_bins, neurons)
        sample_idx: Which sample in the batch to visualize
        neuron_ids: List of specific neuron indices to plot
        save_path: If provided, save the figure to this path
    """
    # Convert to numpy if needed
    if hasattr(actual_spikes, "device"):
        actual_spikes = np.array(actual_spikes)
    if hasattr(predicted_rates, "device"):
        predicted_rates = np.array(predicted_rates)

    # Get single sample
    actual = actual_spikes[sample_idx]  # Shape: (time_bins, neurons)
    predicted = predicted_rates[sample_idx]  # Shape: (time_bins, neurons)

    n_neurons_to_plot = len(neuron_ids)
    fig, axes = plt.subplots(
        n_neurons_to_plot, 1, figsize=(12, 2 * n_neurons_to_plot), sharex=True
    )

    if n_neurons_to_plot == 1:
        axes = [axes]

    for i, neuron_id in enumerate(neuron_ids):
        axes[i].plot(
            actual[:, neuron_id], "b-", alpha=0.7, label="Actual", linewidth=1.5
        )
        axes[i].plot(
            predicted[:, neuron_id], "r-", alpha=0.7, label="Predicted", linewidth=1.5
        )
        axes[i].set_ylabel(f"Neuron {neuron_id}")
        axes[i].set_ylim(0, 3)  # Set y-axis range from 0 to 3
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc="upper right")

    axes[-1].set_xlabel("Time Bin")
    fig.suptitle("Actual Spikes vs Predicted Rates", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig


def plot_neuron_traces(
    actual_spikes,
    predicted_rates,
    neuron_ids=[0, 10, 20, 30, 40],
    sample_idx=0,
    save_path=None,
):
    """
    Plot time series for specific neurons

    Args:
        actual_spikes: Shape (batch_size, time_bins, neurons)
        predicted_rates: Shape (batch_size, time_bins, neurons)
        neuron_ids: List of neuron indices to plot
        sample_idx: Which sample in the batch to visualize
        save_path: If provided, save the figure to this path
    """
    # Convert to numpy if needed
    if hasattr(actual_spikes, "device"):
        actual_spikes = np.array(actual_spikes)
    if hasattr(predicted_rates, "device"):
        predicted_rates = np.array(predicted_rates)

    actual = actual_spikes[sample_idx]
    predicted = predicted_rates[sample_idx]

    n_neurons = len(neuron_ids)
    fig, axes = plt.subplots(n_neurons, 1, figsize=(12, 2 * n_neurons), sharex=True)

    if n_neurons == 1:
        axes = [axes]

    for i, neuron_id in enumerate(neuron_ids):
        axes[i].plot(actual[:, neuron_id], "b-", alpha=0.7, label="Actual")
        axes[i].plot(predicted[:, neuron_id], "r-", alpha=0.7, label="Predicted")
        axes[i].set_ylabel(f"Neuron {neuron_id}")
        axes[i].set_ylim(0, 3)  # Set y-axis range from 0 to 3
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Bin")
    fig.suptitle("Actual Spikes vs Predicted Rates - Individual Neurons")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig


def plot_population_activity(
    actual_spikes, predicted_rates, sample_idx=0, save_path=None
):
    """
    Plot population-level statistics

    Args:
        actual_spikes: Shape (batch_size, time_bins, neurons)
        predicted_rates: Shape (batch_size, time_bins, neurons)
        sample_idx: Which sample in the batch to visualize
        save_path: If provided, save the figure to this path
    """
    # Convert to numpy if needed
    if hasattr(actual_spikes, "device"):
        actual_spikes = np.array(actual_spikes)
    if hasattr(predicted_rates, "device"):
        predicted_rates = np.array(predicted_rates)

    actual = actual_spikes[sample_idx]
    predicted = predicted_rates[sample_idx]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Sum across neurons (population activity)
    actual_pop = np.sum(actual, axis=1)
    predicted_pop = np.sum(predicted, axis=1)

    # Plot 1: Population activity over time
    axes[0].plot(actual_pop, "b-", alpha=0.7, label="Actual", linewidth=2)
    axes[0].plot(predicted_pop, "r-", alpha=0.7, label="Predicted", linewidth=2)
    axes[0].set_ylabel("Total Spike Count")
    axes[0].set_xlabel("Time Bin")
    axes[0].set_title("Population Activity Over Time")
    axes[0].set_ylim(0, 3)  # Set y-axis range from 0 to 3
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Scatter plot of actual vs predicted (per time bin)
    axes[1].scatter(actual_pop, predicted_pop, alpha=0.5)
    max_val = max(np.max(actual_pop), np.max(predicted_pop))
    axes[1].plot(
        [0, max_val], [0, max_val], "k--", alpha=0.3, label="Perfect prediction"
    )
    axes[1].set_xlabel("Actual Population Spike Count")
    axes[1].set_ylabel("Predicted Population Spike Count")
    axes[1].set_title("Actual vs Predicted Population Activity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig

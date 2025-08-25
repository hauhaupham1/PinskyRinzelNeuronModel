import numpy as np
import h5py
from scipy.sparse import csr_matrix


def process_nwb_to_h5(nwb, bin_size=0.005, window=(-0.25, 0.45)):
    """
    Process NWB file to H5 format for STNDT.

    Args:
        nwb: NWB file object
        bin_size: Bin size in seconds (5ms default)
        window: Time window around go cue (seconds)
    """
    n_neurons = len(nwb.units)
    n_trials = len(nwb.trials)

    # Calculate number of bins
    window_duration = window[1] - window[0]
    n_bins = int(window_duration / bin_size)

    # Initialize arrays
    all_spike_counts = np.zeros((n_trials, n_bins, n_neurons))

    # Process each trial
    for trial_idx in range(n_trials):
        # Get trial timing
        move_onset = nwb.trials["move_onset_time"][trial_idx]
        trial_start = move_onset + window[0]  # -250ms before movement
        trial_end = move_onset + window[1]

        # Bin spikes for each neuron
        for neuron_idx in range(n_neurons):
            spike_times = nwb.units.get_unit_spike_times(neuron_idx)

            # Get spikes in trial window
            trial_spikes = spike_times[
                (spike_times >= trial_start) & (spike_times < trial_end)
            ]

            # Bin the spikes
            counts, _ = np.histogram(
                trial_spikes,
                bins=np.arange(trial_start, trial_end + bin_size, bin_size),
            )

            all_spike_counts[trial_idx, :, neuron_idx] = counts[:n_bins]

    # Split by trial type
    splits = nwb.trials["split"][:]
    train_mask = splits == "train"
    val_mask = splits == "val"

    # Get heldout neurons
    heldout_mask = nwb.units["heldout"][:]
    heldin_mask = ~heldout_mask

    # Create final arrays - include ALL neurons for STNDT training
    # The model will mask heldout neurons during training
    train_data = all_spike_counts[train_mask]  # All neurons for input
    train_labels = all_spike_counts[train_mask]  # All neurons for labels

    val_data = all_spike_counts[val_mask]  # All neurons for input
    val_labels = all_spike_counts[val_mask]  # All neurons for labels

    return {
        "train_data": train_data,
        "val_data": val_data,
        "heldout_mask": heldout_mask,
        "heldin_mask": heldin_mask,
        "n_heldin": np.sum(heldin_mask),
        "n_heldout": np.sum(heldout_mask),
    }

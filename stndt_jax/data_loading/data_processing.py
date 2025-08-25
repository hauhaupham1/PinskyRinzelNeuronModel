import numpy as np
import jax.numpy as jnp
import jax.random as jr

from data_loading.get_data_S1 import num_time_bins


def process_sample_vectorized(it, t, bin_size, simulation_length, num_neurons):
    num_time_bins = simulation_length // bin_size
    bin_indices = (t // bin_size).astype(jnp.int32)

    # Create a mask for valid spikes (not padding and within bounds)
    valid_mask = (
        (bin_indices >= 0)
        & (bin_indices < num_time_bins)
        & (it >= 0)
        & (it < num_neurons)
    )

    safe_bins = jnp.where(valid_mask, bin_indices, 0)
    safe_neurons = jnp.where(valid_mask, it, 0)
    values_to_add = jnp.where(valid_mask, 1, 0)

    sample_matrix = jnp.zeros((num_time_bins, num_neurons), dtype=jnp.int32)
    sample_matrix = sample_matrix.at[safe_bins, safe_neurons].add(values_to_add)
    return sample_matrix


# loading all data at once
# def preprocess_data(train_data, test_data):
#     processed_train = []
#     processed_test = []
#     for data in (train_data):
#         it, t = data
#         sample_matrix = process_sample_vectorized(it, t)
#         processed_train.append(sample_matrix)

#     for data in (test_data):
#         it, t = data
#         sample_matrix = process_sample_vectorized(it, t)
#         processed_test.append(sample_matrix)

#     return jnp.array(processed_train), jnp.array(processed_test)


def data_loading_for_batch(
    train_data,
    batch_size=128,
    batch_idx=0,
    bin_size=20,
    simulation_length=1000,
    num_neurons=100,
):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(train_data))
    batch_indices = range(start_idx, end_idx)
    batch_raw = [train_data[i] for i in batch_indices]
    batch_binned = []
    for it, t in batch_raw:
        sample_matrix = process_sample_vectorized(
            it, t, bin_size, simulation_length, num_neurons
        )
        batch_binned.append(sample_matrix)
    batch_binned = jnp.array(batch_binned)
    return batch_binned


def find_max_spikes_from_data(train_data, test_data=None, sample_size=None):
    """
    Find the maximum spike count in the dataset

    Args:
        train_data: Training data list
        test_data: Optional test data list
        sample_size: If provided, only check this many samples (for speed)
    """
    max_spike_count = 0

    # Sample from train data if sample_size specified
    data_to_check = train_data
    if sample_size and sample_size < len(train_data):
        indices = jr.choice(
            jr.PRNGKey(0), len(train_data), (sample_size,), replace=False
        )
        data_to_check = [train_data[i] for i in indices]

    # Check train data
    for it, t in data_to_check:
        sample_matrix = process_sample_vectorized(it, t)
        max_in_sample = jnp.max(sample_matrix)
        max_spike_count = max(max_spike_count, int(max_in_sample))

    # Also check test data if provided
    if test_data:
        test_to_check = test_data[
            : min(1000, len(test_data))
        ]  # Check first 1000 test samples
        for it, t in test_to_check:
            sample_matrix = process_sample_vectorized(it, t)
            max_in_sample = jnp.max(sample_matrix)
            max_spike_count = max(max_spike_count, int(max_in_sample))

    return max_spike_count

import jax.random as jr
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset

# jnp.set_printoptions(threshold=jnp.inf)

trial_length = 1000
bin_size = 40  # Changed from 8ms to 20ms for better class balance
num_time_bins = trial_length // bin_size
num_neurons = 11

def load_s1_train():
    dataset = load_dataset("livn-org/livn", name="S1")
    train_data = []
    for i, sample in enumerate(dataset["train_with_noise"]):
        if i >= 30000:
            break
        it = sample["trial_it"][0]
        t = sample["trial_t"][0]
        train_data.append((it, t))

    return train_data

def load_s1_test():
    dataset = load_dataset("livn-org/livn", name="S1")
    test_data = []
    for sample in dataset["test_with_noise"]:
        it = sample["trial_it"][0]
        t = sample["trial_t"][0]
        test_data.append((it, t))

    return test_data

def process_sample_vectorized(it, t):
    it = jnp.array(it)
    t = jnp.array(t)
    bin_indices = (t // bin_size).astype(int)

    valid_mask = (bin_indices >= 0) & (bin_indices < num_time_bins) & (it >= 0) & (it < num_neurons)
    valid_bins = bin_indices[valid_mask]
    valid_neurons = it[valid_mask]

    sample_matrix = jnp.zeros((num_time_bins, num_neurons))
    return sample_matrix.at[valid_bins, valid_neurons].add(1)

#loading all data at once
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



def data_loading_for_batch(train_data, batch_size = 128, batch_idx=0):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(train_data))
    batch_indices = range(start_idx, end_idx)
    batch_raw = [train_data[i] for i in batch_indices]
    batch_binned = []
    for it, t in batch_raw:
        sample_matrix = process_sample_vectorized(it, t)
        batch_binned.append(sample_matrix)
    batch_binned = jnp.array(batch_binned)
    return batch_binned

# train_data, test_data = load_s1_data()
# first_batch = data_loading_for_batch(train_data, batch_size=128, batch_idx=0)
# print("Second sample:", first_batch[0])  # Print the second sample in the

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
        indices = jr.choice(jr.PRNGKey(0), len(train_data), (sample_size,), replace=False)
        data_to_check = [train_data[i] for i in indices]
    
    # Check train data
    for it, t in data_to_check:
        sample_matrix = process_sample_vectorized(it, t)
        max_in_sample = jnp.max(sample_matrix)
        max_spike_count = max(max_spike_count, int(max_in_sample))
    
    # Also check test data if provided
    if test_data:
        test_to_check = test_data[:min(1000, len(test_data))]  # Check first 1000 test samples
        for it, t in test_to_check:
            sample_matrix = process_sample_vectorized(it, t)
            max_in_sample = jnp.max(sample_matrix)
            max_spike_count = max(max_spike_count, int(max_in_sample))
    
    return max_spike_count


# train_data = load_s1_train()
# test_data = load_s1_test()
# max_spikes = find_max_spikes_from_data(train_data, test_data)


# print(f"Max spikes in dataset: {max_spikes}")

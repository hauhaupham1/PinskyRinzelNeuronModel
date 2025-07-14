import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset

# jnp.set_printoptions(threshold=jnp.inf)

trial_length = 1000 
bin_size = 8
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

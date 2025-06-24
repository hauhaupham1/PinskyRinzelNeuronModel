import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset

trial_length = 1000 
bin_size = 20
num_time_bins = trial_length // bin_size
num_neurons = 10

def load_s1_data():
    dataset = load_dataset("livn-org/livn", name="S1")
    train_data = []
    test_data = []
    for sample in dataset["train"]:
        it = sample["trial_it"][0]
        t = sample["trial_t"][0]
        train_data.append((it, t))

    for sample in dataset["test"]:
        it = sample["trial_it"][0]
        t = sample["trial_t"][0]
        test_data.append((it, t))
    return train_data, test_data

def process_sample_vectorized(it, t):
    it = jnp.array(it)
    t = jnp.array(t)
    bin_indices = (t // bin_size).astype(int)

    valid_mask = (bin_indices >= 0) & (bin_indices < num_time_bins) & (it >= 0) & (it < num_neurons)
    valid_bins = bin_indices[valid_mask]
    valid_neurons = it[valid_mask]

    sample_matrix = jnp.zeros((num_time_bins, num_neurons))
    return sample_matrix.at[valid_bins, valid_neurons].add(1)


def preprocess_data(train_data, test_data):
    processed_train = []
    processed_test = []
    for sample_idx,data in enumerate(train_data):
        it, t = data
        sample_matrix = process_sample_vectorized(it, t)
        processed_train.append(sample_matrix)

    for sample_idx,data in enumerate(test_data):
        it, t = data
        sample_matrix = process_sample_vectorized(it, t)

        processed_test.append(sample_matrix)

    return jnp.array(processed_train), jnp.array(processed_test)


train_data, test_data = load_s1_data()
#print it and t for the first sample
print("First Train Sample it:", len(train_data[0][0]))
print("First Train Sample t:", len(train_data[0][1]))
print("Max t in first train sample:", jnp.max(jnp.array(train_data[0][1])))
processed_train, processed_test = preprocess_data(train_data, test_data)

print("Processed Train Data Shape:", processed_train.shape)
print("Processed train first sample:\n", jnp.sum(processed_train[0]))


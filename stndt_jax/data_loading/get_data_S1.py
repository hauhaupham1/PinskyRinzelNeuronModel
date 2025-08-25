import jax.random as jr
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset

# jnp.set_printoptions(threshold=jnp.inf)

trial_length = 1000
bin_size = 20  # Changed from 8ms to 20ms for better class balance
num_time_bins = trial_length // bin_size
num_neurons = 11


def load_s1_train():
    dataset = load_dataset("livn-org/livn", name="S1", streaming=True)
    train_data = []
    for i, sample in enumerate(dataset["train_with_noise"]):
        if i >= 30000:
            break
        it = sample["trial_it"][0]
        t = sample["trial_t"][0]
        train_data.append((it, t))

    return train_data


def load_s1_test():
    dataset = load_dataset("livn-org/livn", name="S1", streaming=True)
    test_data = []
    for sample in dataset["test_with_noise"]:
        it = sample["trial_it"][0]
        t = sample["trial_t"][0]
        test_data.append((it, t))

    return test_data


# train_data, test_data = load_s1_data()
# first_batch = data_loading_for_batch(train_data, batch_size=128, batch_idx=0)
# print("Second sample:", first_batch[0])  # Print the second sample in the


# train_data = load_s1_train()
# test_data = load_s1_test()
# max_spikes = find_max_spikes_from_data(train_data, test_data)


# print(f"Max spikes in dataset: {max_spikes}")

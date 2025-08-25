import jax.numpy as jnp
from datasets import load_dataset

# num_neurons = find_num_neurons()
num_neurons = 100
t_end = 1000
bin_size = 20
num_time_bins = t_end / bin_size


def load_s2_train():
    dataset = load_dataset("livn-org/livn", name="S2")
    train_data = []
    for i, sample in enumerate(dataset["train_with_noise"]):
        if i >= 30000:
            break
        it = sample["trial_it"][0]
        t = sample["trial_t"][0]
        train_data.append((it, t))

    return train_data


def load_s2_val():
    dataset = load_dataset("livn-org/livn", name="S2")
    train_data = []
    for i, sample in enumerate(dataset["test_with_noise"]):
        if i >= 30000:
            break
        it = sample["trial_it"][0]
        t = sample["trial_t"][0]
        train_data.append((it, t))

    return train_data


def find_num_neurons(dataset):
    sample = dataset["train_with_noise"][1]
    it = sample["trial_it"][0]
    return jnp.unique(jnp.array(it))[-1] + 1


# system_name = "S2"

# dataset = load_dataset("livn-org/livn", name=system_name)
# sample = dataset["train_with_noise"][0]
# it = sample["trial_it"][0]
# t = sample["trial_t"][0]

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax
from data_loading.data_processing import process_sample_vectorized


def preprocess_data(data, bin_size, simulation_length, num_neurons):
    """
    Process all raw data samples into binned spike count matrices using vmap.
    """
    # pad for vmapping
    max_len = max(len(it) for it, t in data)

    padded_its = []
    padded_ts = []
    for it, t in data:
        pad_len = max_len - len(it)
        padded_its.append(
            jnp.pad(jnp.array(it), (0, pad_len), "constant", constant_values=-1)
        )
        padded_ts.append(
            jnp.pad(jnp.array(t), (0, pad_len), "constant", constant_values=-1)
        )

    its_array = jnp.stack(padded_its)
    ts_array = jnp.stack(padded_ts)

    vmapped_processor = jax.vmap(
        lambda it, t: process_sample_vectorized(
            it, t, bin_size, simulation_length, num_neurons
        )
    )
    return vmapped_processor(its_array, ts_array)


class DataLoader:
    def __init__(
        self,
        data,
        batch_size=64,
        shuffle=True,
        seed=0,
        bin_size=20,
        simulation_length=1000,
        num_neurons=100,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng_key = jr.PRNGKey(seed)

        # Pre-process the entire dataset
        print("Pre-processing data... (this may take a moment)")
        self.processed_data = preprocess_data(
            data, bin_size, simulation_length, num_neurons
        )
        print("Data pre-processing complete.")

        self.num_samples = self.processed_data.shape[0]
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self.current_epoch = 0

        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        self.rng_key, subkey = jr.split(self.rng_key)
        self.indices = jr.permutation(subkey, self.indices)

    def get_batch(self, batch_idx):
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)

        batch_indices = self.indices[start_idx:end_idx]
        return self.processed_data[batch_indices]

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        batch = self.get_batch(self.current_batch)
        self.current_batch += 1
        return batch

    def on_epoch_end(self):
        self.current_epoch += 1
        if self.shuffle:
            self._shuffle_data()

    def __len__(self):
        return self.num_batches


def create_data_loader(
    data,
    batch_size=128,
    shuffle=True,
    seed=0,
    bin_size=20,
    simulation_length=1000,
    num_neurons=100,
):
    return DataLoader(
        data, batch_size, shuffle, seed, bin_size, simulation_length, num_neurons
    )

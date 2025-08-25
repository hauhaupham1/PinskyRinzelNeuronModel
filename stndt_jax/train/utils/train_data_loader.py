"""Simple data loader for training without lograte transformation."""

import jax
import jax.numpy as jnp
import numpy as np
from pynwb import read_nwb

from stndt_jax.train.utils.extract_nwb_data import process_nwb_to_h5


def load_and_prepare_data(nwb_path):
    """Load NWB data and prepare for training (no lograte)."""

    print("Loading NWB file...")
    nwb = read_nwb(nwb_path)

    print("Processing data (binning spikes)...")
    data_dict = process_nwb_to_h5(nwb, bin_size=0.005, window=(-0.25, 0.45))

    return data_dict


def get_batch(data_dict, batch_size=64, split="train", key=None):
    """Get a random batch from the dataset."""

    if key is None:
        key = jax.random.PRNGKey(0)

    data = data_dict[f"{split}_data"]
    n_trials = data.shape[0]

    # Random sample indices
    key, subkey = jax.random.split(key)
    indices = jax.random.choice(subkey, n_trials, shape=(batch_size,), replace=False)

    # Get batch and convert to JAX arrays

    return jnp.array(data[indices])


def apply_masking_for_training(batch, data_dict, config=None):
    """Apply masking to a batch for STNDT training."""

    from stndt_jax.train.utils.mask import mask_batch

    # Split batch into heldin and heldout neurons
    heldin_mask = data_dict["heldin_mask"]
    heldout_mask = data_dict["heldout_mask"]

    heldin_batch = batch[:, :, heldin_mask]  # Input neurons
    heldout_spikes = batch[:, :, heldout_mask]  # Heldout neurons for targets

    masked_batch, labels, _, _, _, _ = mask_batch(
        contrast_mode=False,
        batch=heldin_batch,
        heldout_spikes=heldout_spikes,
        config=config,
        expand_prob=config["MASK_SPAN_PROB"],
    )

    return masked_batch, labels


def apply_masking_for_training_contrast(batch, data_dict, contrast_config=None):
    """Apply masking to a batch for contrastive STNDT training."""

    from stndt_jax.train.utils.mask import mask_batch

    # Split batch into heldin and heldout neurons
    heldin_mask = data_dict["heldin_mask"]
    heldout_mask = data_dict["heldout_mask"]

    heldin_batch = batch[:, :, heldin_mask]  # Input neurons
    heldout_spikes = batch[:, :, heldout_mask]  # Heldout neurons for targets

    masked_batch, labels, _, _, _, _ = mask_batch(
        contrast_mode=True,
        batch=heldin_batch,
        heldout_spikes=heldout_spikes,
        config=contrast_config,
        expand_prob=contrast_config["CONTRAST_MASK_SPAN_PROB"],
    )

    return masked_batch, labels


# Example usage

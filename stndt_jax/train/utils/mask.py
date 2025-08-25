import jax
import jax.numpy as jnp
import jax.random as jrandom
# Original file available at: https://github.com/trungle93/STNDT/blob/main/src/mask.py
# Adapted by Hau Pham
# Convert to JAX from PyTorch

DEFAULT_MASK_VAL = 30
UNMASKED_LABEL = -100
SUPPORTED_MODES = ["full", "timestep", "neuron", "timestep_only"]


def create_full_batch(batch, labels, heldout_spikes=None, forward_spikes=None):
    # If provide both held-out and forward, then last dimension of forward_spike must be equal batch[-1] + heldout_spike[-1]
    """Concatenate training data"""
    # Add held-out neurons (new neurons)
    if heldout_spikes is not None:
        batch = jnp.concatenate([batch, heldout_spikes], axis=-1)
        labels = jnp.concatenate([labels, heldout_spikes], axis=-1)
    # Add forward time (future timesteps)
    if forward_spikes is not None:
        batch = jnp.concatenate([batch, forward_spikes], axis=1)
        labels = jnp.concatenate([labels, forward_spikes], axis=1)
    return batch, labels


def create_full_batch_zero(batch, labels, heldout_spikes=None, forward_spikes=None):
    # If provide both held-out and forward, then last dimension of forward_spike must be equal batch[-1] + heldout_spike[-1]
    """Concatenate training data"""
    # Add held-out neurons (new neurons)
    if heldout_spikes is not None:
        batch = jnp.concatenate([batch, jnp.zeros_like(heldout_spikes)], axis=-1)
        labels = jnp.concatenate([labels, heldout_spikes], axis=-1)
    # Add forward time (future timesteps)
    if forward_spikes is not None:
        batch = jnp.concatenate([batch, jnp.zeros_like(forward_spikes)], axis=1)
        labels = jnp.concatenate([labels, forward_spikes], axis=1)
    return batch, labels


def expand_mask(mask, width):
    """
    Expand mask positions into contiguous spans using 1D convolution.

    How it works:
    - Convolves a kernel of ones across the mask
    - Wherever the kernel overlaps with a 1, outputs become non-zero
    - This "spreads" each 1 into a span of width `width`

    Args:
        mask: Shape [N, T] where N is batch, T is sequence length
              Binary mask with 1s to expand
        width: Integer span width (how wide each 1 becomes)

    Returns:
        Expanded mask with same shape as input

    Example:
        mask = [0, 0, 1, 0, 0]
        width = 3
        result = [0, 1, 1, 1, 0]  # The 1 expanded to width 3
    """
    kernel = jnp.ones(width)
    mask_3d = mask[:, jnp.newaxis, :]
    kernel_3d = kernel[jnp.newaxis, jnp.newaxis, :]
    padding = width // 2

    dn = jax.lax.conv_dimension_numbers(
        mask_3d.shape, kernel_3d.shape, ("NCH", "OIH", "NCH")
    )

    expanded = jax.lax.conv_general_dilated(
        mask_3d,
        kernel_3d,
        window_strides=(1,),  # Slide one position at a time
        padding=((padding, padding),),  # Pad both sides
        dimension_numbers=dn,
    )

    expanded = jnp.clip(expanded, 0, 1)[:, 0, :]

    if width % 2 == 0:
        expanded = expanded[:, :-1]

    return expanded


def create_mask(labels, mode, mask_ratio, key):
    if mode == "full":
        mask = jrandom.bernoulli(key=key, p=mask_ratio, shape=labels.shape)
    elif mode == "timestep":
        single_timestep = labels[:, :, 0]  # B x T
        mask = jrandom.bernoulli(key=key, p=mask_ratio, shape=single_timestep.shape)
    elif mode == "neuron":
        single_neuron = labels[:, 0]  # B x N
        mask = jrandom.bernoulli(key=key, p=mask_ratio, shape=single_neuron.shape)
    elif mode == "timestep_only":
        single_timestep = labels[0, :, 0]  # T
        mask = jrandom.bernoulli(key=key, p=mask_ratio, shape=single_timestep.shape)

    return mask


def process_mode_width_ratio(contrast_mode, config, expand_prob, key):
    keys = jrandom.split(key, 2)
    if contrast_mode:
        mode = config["CONTRAST_MASK_MODE"]
        should_expand = (
            config["CONTRAST_MASK_MAX_SPAN"] > 1
            and expand_prob > 0.0
            and jrandom.uniform(keys[0]) < expand_prob
        )
        width = (
            jrandom.randint(keys[1], (1,), 1, config["CONTRAST_MASK_MAX_SPAN"] + 1)
            if should_expand
            else 1
        )
        mask_ratio = (
            config["CONTRAST_MASK_RATIO"]
            if width == 1
            else config["CONTRAST_MASK_RATIO"] / width
        )
    else:
        mode = config["MASK_MODE"]
        should_expand = (
            config["MASK_MAX_SPAN"] > 1
            and expand_prob > 0.0
            and jrandom.uniform(keys[0]) < expand_prob
        )
        width = (
            jrandom.randint(keys[1], (1,), 1, config.MASK_MAX_SPAN + 1)
            if should_expand
            else 1
        )
        mask_ratio = (
            config["MASK_RATIO"] if width == 1 else config["MASK_RATIO"] / width
        )
    return mode, width, mask_ratio


def mask_batch(
    batch,
    contrast_mode: bool,
    mask=None,
    max_spikes=DEFAULT_MASK_VAL - 1,
    should_mask=True,
    expand_prob=0.0,
    heldout_spikes=None,
    forward_spikes=None,
    config=None,
):
    keys = jrandom.split(jrandom.PRNGKey(29082003123), 6)
    mode, width, mask_ratio = process_mode_width_ratio(
        contrast_mode=contrast_mode, config=config, expand_prob=expand_prob, key=keys[0]
    )

    labels = batch
    batch_full, labels_full = create_full_batch(
        batch, labels, heldout_spikes, forward_spikes
    )
    spikes_full = batch_full

    if mask is None:
        mask = create_mask(labels, mode, mask_ratio, keys[1])
        mask_full = create_mask(labels_full, mode, mask_ratio, keys[5])
        mask = mask.astype(jnp.bool)
        mask_full = mask_full.astype(jnp.bool)

        if width > 1:
            mask = expand_mask(mask=mask, width=width)
            mask_full = expand_mask(mask=mask_full, width=width)

        if mode == "timestep":
            mask = jnp.broadcast_to(jnp.expand_dims(mask, 2), labels.shape)
            mask_full = jnp.broadcast_to(
                jnp.expand_dims(mask_full, 2), labels_full.shape
            )
        elif mode == "neuron":
            mask = jnp.broadcast_to(jnp.expand_dims(mask, 1), labels.shape)
            mask_full = jnp.broadcast_to(
                jnp.expand_dims(mask_full, 1), labels_full.shape
            )
        elif mode == "timestep_only":
            mask = jnp.broadcast_to(
                jnp.expand_dims(jnp.expand_dims(mask, 0), 2), labels.shape
            )
            mask_full = jnp.broadcast_to(
                jnp.expand_dims(jnp.expand_dims(mask_full, 0), 2), labels_full.shape
            )
    elif mask.shape != labels.shape:
        raise Exception("mask shape and labels must have same shape")

    labels = jnp.where(mask, labels, UNMASKED_LABEL)
    labels_full = jnp.where(mask_full, labels_full, UNMASKED_LABEL)

    if not should_mask:
        return batch, labels

    if not contrast_mode:
        indices_replaced = (
            jrandom.bernoulli(
                key=keys[2], p=config["MASK_TOKEN_RATIO"], shape=labels.shape
            )
            & mask
        )
        indices_replaced_full = (
            jrandom.bernoulli(
                key=keys[2], p=config["MASK_TOKEN_RATIO"], shape=labels_full.shape
            )
            & mask_full
        )
    else:
        indices_replaced = (
            jrandom.bernoulli(
                key=keys[2], p=config["CONTRAST_MASK_TOKEN_RATIO"], shape=labels.shape
            )
            & mask
        )
        indices_replaced_full = (
            jrandom.bernoulli(
                key=keys[2],
                p=config["CONTRAST_MASK_TOKEN_RATIO"],
                shape=labels_full.shape,
            )
            & mask_full
        )

    if config["USE_ZERO_MASK"]:
        batch = jnp.where(indices_replaced, 0, batch)
        batch_full = jnp.where(indices_replaced_full, 0, batch_full)
    else:
        batch = jnp.where(indices_replaced, max_spikes + 1, batch)
        batch_full = jnp.where(indices_replaced_full, max_spikes + 1, batch_full)

    if not contrast_mode:
        indices_random = (
            jrandom.bernoulli(
                key=keys[3], p=config["MASK_RANDOM_RATIO"], shape=labels.shape
            )
            & mask
            & ~indices_replaced
        )
        indices_random_full = (
            jrandom.bernoulli(
                key=keys[3], p=config["MASK_RANDOM_RATIO"], shape=labels_full.shape
            )
            & mask_full
            & ~indices_replaced_full
        )
    else:
        indices_random = (
            jrandom.bernoulli(
                key=keys[3], p=config["CONTRAST_MASK_RANDOM_RATIO"], shape=labels.shape
            )
            & mask
            & ~indices_replaced
        )
        indices_random_full = (
            jrandom.bernoulli(
                key=keys[3],
                p=config["CONTRAST_MASK_RANDOM_RATIO"],
                shape=labels_full.shape,
            )
            & mask_full
            & ~indices_replaced_full
        )
    random_spikes = jrandom.randint(
        key=keys[4],
        shape=labels.shape,
        dtype=jnp.int32,
        minval=0,
        maxval=jnp.array(jnp.max(batch), int),
    )

    random_spikes_full = jrandom.randint(
        key=keys[4],
        shape=labels_full.shape,
        dtype=jnp.int32,
        minval=0,
        maxval=jnp.array(jnp.max(batch_full), int),
    )

    batch = jnp.where(indices_random, random_spikes, batch)
    batch_full = jnp.where(indices_random_full, random_spikes_full, batch_full)
    masked_heldin = batch

    batch, labels = create_full_batch_zero(
        batch, labels, heldout_spikes=heldout_spikes, forward_spikes=forward_spikes
    )
    return batch, labels, masked_heldin, batch_full, labels_full, spikes_full


# config = {
#     "MASK_MODE": "neuron",
#     "MASK_RATIO": 0.3,
#     "MASK_MAX_SPAN": 1,
#     "MASK_TOKEN_RATIO": 0.8,
#     "MASK_RANDOM_RATIO": 0.1,
#     "USE_ZERO_MASK": True,
# }
#
# ##TESTING
# dummy = jnp.ones((2, 6, 5))
# heldout = jnp.ones((2, 6, 2))
# forward = jnp.ones((2, 3, 7))
# batched, labels, _, batched_full, labels_full, _ = mask_batch(
#     batch=dummy,
#     contrast_mode=False,
#     should_mask=True,
#     expand_prob=0.0,
#     heldout_spikes=heldout,
#     forward_spikes=forward,
#     config=config,
# )
#
# print(f"batched_full: {batched_full.shape}")
# print(f"Labels Full {labels_full.shape}")

from random import random
import jax.numpy as jnp
import jax

DEFAULT_MASK_VALUE = 30
UNMASKED_LABEL = -100
SUPPORTED_MODES = ["full", "timestep", "neuron", "timestep_only"]


def random_mask(data, mask_ratio=0.15, key=None):
    """Create random mask for individual positions"""
    if key is None:
        key = jax.random.PRNGKey(0)
    mask = jax.random.bernoulli(key, mask_ratio, shape=data.shape)
    return mask


def timestep_mask(data, mask_ratio=0.15, key=None):
    """Mask entire timesteps across all neurons"""
    if key is None:
        key = jax.random.PRNGKey(0)
    batch_size, timesteps, neurons = data.shape

    # Create mask for timesteps only
    timestep_mask = jax.random.bernoulli(key, mask_ratio, shape=(batch_size, timesteps))
    # Expand to all neurons
    return jnp.broadcast_to(timestep_mask[:, :, None], data.shape)


def neuron_mask(data, mask_ratio=0.15, key=None):
    """Mask entire neurons across all timesteps"""
    if key is None:
        key = jax.random.PRNGKey(0)
    batch_size, timesteps, neurons = data.shape

    # Create mask for neurons only
    neuron_mask = jax.random.bernoulli(key, mask_ratio, shape=(batch_size, neurons))
    # Expand to all timesteps
    return jnp.broadcast_to(neuron_mask[:, None, :], data.shape)


def timestep_only_mask(data, mask_ratio=0.15, key=None):
    """Mask same timesteps across all samples in batch"""
    if key is None:
        key = jax.random.PRNGKey(0)
    batch_size, timesteps, neurons = data.shape

    # Create mask for timesteps (same for all batch samples)
    timestep_mask = jax.random.bernoulli(key, mask_ratio, shape=(timesteps,))
    # Expand to all batch samples and neurons
    return jnp.broadcast_to(timestep_mask[None, :, None], data.shape)


def create_mask(data, mode="full", mask_ratio=0.15, key=None):
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Mode {mode} not supported. Supported modes: {SUPPORTED_MODES}"
        )

    if key is None:
        key = jax.random.PRNGKey(0)

    if mode == "full":
        return random_mask(data, mask_ratio, key)
    elif mode == "timestep":
        return timestep_mask(data, mask_ratio, key)
    elif mode == "neuron":
        return neuron_mask(data, mask_ratio, key)
    elif mode == "timestep_only":
        return timestep_only_mask(data, mask_ratio, key)


def apply_masking(
    data, mask, key, mask_token_ratio=0.8, max_spikes=29, use_zero_mask=True
):
    """Apply two-tier masking strategy - ALL masked positions get corrupted"""

    # Split the key for different random operations
    key1, key2 = jax.random.split(key, 2)

    # Generate random assignment for each position
    assignment = jax.random.bernoulli(key1, mask_token_ratio, shape=mask.shape)

    # Create mask token values and random values
    mask_token_value = 0 if use_zero_mask else max_spikes + 1
    random_values = jax.random.randint(key2, mask.shape, 0, max_spikes + 1)

    # Apply masking: where mask is True, replace with either mask token or random value
    # assignment = True -> mask token, assignment = False -> random value
    masked_data = jnp.where(
        mask,  # If position should be masked
        jnp.where(
            assignment, mask_token_value, random_values
        ),  # Replace with mask token or random
        data,  # Otherwise keep original
    )

    return masked_data


def expand_mask(mask, width):
    kernel = jnp.ones(width)
    expanded = jax.scipy.signal.convolve(mask, kernel, mode="same")
    return jnp.clip(expanded, 0, 1)


def create_forward_prediction_mask(batch_data, num_forward_steps=1):
    # batch_data: (batch, 125, 11)
    context_length = batch_data.shape[1] - num_forward_steps
    assert context_length > 0, "Context length must be positive"
    input_data = batch_data.copy()
    input_data = input_data.at[:, context_length:, :].set(0)

    # Labels only for the positions we want to predict
    mask_labels = jnp.full_like(batch_data, -100)
    mask_labels = mask_labels.at[:, context_length:, :].set(
        batch_data[:, context_length:, :]
    )

    return input_data, mask_labels


def create_reconstruction_mask(
    batch_data,
    key,
    mask_ratio=0.15,
    mode="full",
    mask_token_ratio=0.8,
    use_zero_mask=True,
    max_spikes=29,
):
    """Create BERT-style reconstruction mask for neural spike data"""

    # Split key for mask creation and masking application
    key1, key2 = jax.random.split(key)

    # Create mask based on mode
    mask = create_mask(batch_data, mode=mode, mask_ratio=mask_ratio, key=key1)

    # Apply two-tier masking strategy
    masked_input = apply_masking(
        batch_data,
        mask,
        key2,
        mask_token_ratio=mask_token_ratio,
        max_spikes=max_spikes,
        use_zero_mask=use_zero_mask,
    )

    # Create labels: -100 for unmasked, true values for masked positions
    labels = jnp.where(mask, batch_data, UNMASKED_LABEL)

    return masked_input, labels


def get_mixed_batch(data, forecast_ratio=0.3, num_forward_steps=1, key=None):
    batch_size = data.shape[0]
    forecast_samples = int(batch_size * forecast_ratio)

    # Split key for different operations
    if key is None:
        key = jax.random.PRNGKey(0)
    key_recon = key  # Use key directly for reconstruction

    forecast_data = data[:forecast_samples]
    forecast_input, forecast_labels = create_forward_prediction_mask(
        forecast_data, num_forward_steps
    )

    recon_data = data[forecast_samples:]
    recon_input, recon_labels = create_reconstruction_mask(recon_data, key_recon)

    return (forecast_input, forecast_labels), (recon_input, recon_labels)

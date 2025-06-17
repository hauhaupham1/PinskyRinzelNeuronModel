"""
Augmentation strategies for neural spike data in STNDT
"""
import jax
import jax.numpy as jnp
import jax.random as jr


def add_noise_augmentation(spike_counts, noise_level=0.1, key=None):
    """
    Add Gaussian noise to spike counts
    
    Args:
        spike_counts: (B, T, N) array of spike counts
        noise_level: Standard deviation of noise relative to mean spike count
        key: JAX random key
    
    Returns:
        Augmented spike counts
    """
    if key is None:
        key = jr.PRNGKey(0)
    
    mean_rate = jnp.mean(spike_counts)
    noise = jr.normal(key, spike_counts.shape) * noise_level * mean_rate
    augmented = jnp.clip(spike_counts + noise, 0, None)
    return jnp.round(augmented).astype(spike_counts.dtype)


def temporal_jitter_augmentation(spike_counts, max_shift=2, key=None):
    """
    Apply temporal jittering to spike counts
    
    Args:
        spike_counts: (B, T, N) array of spike counts
        max_shift: Maximum temporal shift in time bins
        key: JAX random key
    
    Returns:
        Augmented spike counts
    """
    if key is None:
        key = jr.PRNGKey(0)
    
    B, T, N = spike_counts.shape
    
    # Generate random shifts for each neuron
    shifts = jr.randint(key, (B, N), -max_shift, max_shift + 1)
    
    # Apply shifts
    augmented = jnp.zeros_like(spike_counts)
    for b in range(B):
        for n in range(N):
            shift = shifts[b, n]
            if shift > 0:
                augmented = augmented.at[b, shift:, n].set(spike_counts[b, :-shift, n])
            elif shift < 0:
                augmented = augmented.at[b, :shift, n].set(spike_counts[b, -shift:, n])
            else:
                augmented = augmented.at[b, :, n].set(spike_counts[b, :, n])
    
    return augmented


def neuron_dropout_augmentation(spike_counts, dropout_rate=0.1, key=None):
    """
    Randomly drop out entire neurons
    
    Args:
        spike_counts: (B, T, N) array of spike counts
        dropout_rate: Fraction of neurons to drop
        key: JAX random key
    
    Returns:
        Augmented spike counts
    """
    if key is None:
        key = jr.PRNGKey(0)
    
    B, T, N = spike_counts.shape
    
    # Create dropout mask
    mask = jr.uniform(key, (B, 1, N)) > dropout_rate
    
    return spike_counts * mask


def rate_scaling_augmentation(spike_counts, scale_range=(0.8, 1.2), key=None):
    """
    Scale firing rates by a random factor
    
    Args:
        spike_counts: (B, T, N) array of spike counts
        scale_range: (min_scale, max_scale) tuple
        key: JAX random key
    
    Returns:
        Augmented spike counts
    """
    if key is None:
        key = jr.PRNGKey(0)
    
    B, T, N = spike_counts.shape
    min_scale, max_scale = scale_range
    
    # Generate random scales for each sample
    scales = jr.uniform(key, (B, 1, 1), minval=min_scale, maxval=max_scale)
    
    augmented = spike_counts * scales
    return jnp.round(augmented).astype(spike_counts.dtype)


def combined_augmentation(spike_counts, key=None):
    """
    Apply multiple augmentations in sequence
    
    Args:
        spike_counts: (B, T, N) array of spike counts
        key: JAX random key
    
    Returns:
        Augmented spike counts
    """
    if key is None:
        key = jr.PRNGKey(0)
    
    keys = jr.split(key, 4)
    
    # Apply augmentations in sequence
    augmented = spike_counts
    augmented = add_noise_augmentation(augmented, noise_level=0.05, key=keys[0])
    augmented = temporal_jitter_augmentation(augmented, max_shift=1, key=keys[1])
    augmented = neuron_dropout_augmentation(augmented, dropout_rate=0.05, key=keys[2])
    augmented = rate_scaling_augmentation(augmented, scale_range=(0.9, 1.1), key=keys[3])
    
    return augmented


# Example usage for contrastive learning
def create_contrastive_pairs(spike_counts, augmentation_fn=combined_augmentation, key=None):
    """
    Create two augmented views for contrastive learning
    
    Args:
        spike_counts: (B, T, N) array of spike counts
        augmentation_fn: Function to apply augmentations
        key: JAX random key
    
    Returns:
        (view1, view2): Two augmented versions of the input
    """
    if key is None:
        key = jr.PRNGKey(0)
    
    key1, key2 = jr.split(key)
    
    view1 = augmentation_fn(spike_counts, key=key1)
    view2 = augmentation_fn(spike_counts, key=key2)
    
    return view1, view2
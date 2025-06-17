# Author: Trung Le
# Original file available at https://github.com/trungle93/STNDT
# Adapted by Hau Pham
"""
JAX/Equinox implementation of masking utilities for STNDT
Translated from PyTorch implementation
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Dict, Any, Optional, Tuple

# Constants
DEFAULT_MASK_VAL = 30
UNMASKED_LABEL = -100
SUPPORTED_MODES = ["full", "timestep", "neuron", "timestep_only"]


def binary_mask_to_attn_mask(x: jnp.ndarray) -> jnp.ndarray:
    """Convert binary mask to attention mask"""
    return jnp.where(x == 0, -jnp.inf, 0.0)


def expand_mask(mask: jnp.ndarray, width: int) -> jnp.ndarray:
    """
    Expand mask with given width using convolution
    
    Args:
        mask: Binary mask of shape (N, T) 
        width: Expansion width
        
    Returns:
        Expanded mask with same shape as input
    """
    if width <= 1:
        return mask
        
    # Create convolution kernel
    pad_width = width // 2
    
    # Pad the mask
    padded_mask = jnp.pad(mask, ((0, 0), (pad_width, pad_width)), mode='constant')
    
    # Apply convolution-like operation using rolling
    expanded = jnp.zeros_like(padded_mask)
    for i in range(width):
        shift = i - pad_width
        expanded = expanded + jnp.roll(padded_mask, shift, axis=1)
    
    # Crop back to original size
    if width % 2 == 0:
        # For even width, we need to remove one extra padding
        expanded = expanded[:, pad_width:-pad_width-1]
    else:
        # For odd width, remove symmetric padding
        expanded = expanded[:, pad_width:-pad_width]
    
    # Ensure we return exact original shape
    if expanded.shape[1] != mask.shape[1]:
        # Adjust to match exact original length
        expanded = expanded[:, :mask.shape[1]]
    
    return jnp.clip(expanded, 0, 1).astype(mask.dtype)


class JAXMasker:
    """JAX implementation of the masking utilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validate_config()
    
    def validate_config(self):
        """Validate masking configuration"""
        mask_mode = self.config.get('MASK_MODE', 'timestep')
        contrast_mask_mode = self.config.get('CONTRAST_MASK_MODE', 'timestep')
        
        if mask_mode not in SUPPORTED_MODES:
            raise ValueError(f"MASK_MODE {mask_mode} not in supported {SUPPORTED_MODES}")
        if contrast_mask_mode not in SUPPORTED_MODES:
            raise ValueError(f"CONTRAST_MASK_MODE {contrast_mask_mode} not in supported {SUPPORTED_MODES}")
    
    def create_mask_probabilities(
        self, 
        shape: Tuple[int, ...], 
        mode: str, 
        mask_ratio: float
    ) -> jnp.ndarray:
        """Create mask probability matrix based on mode"""
        
        if mode == "full":
            return jnp.full(shape, mask_ratio)
        elif mode == "timestep":
            # Mask entire timesteps: (N, T) -> broadcast to (N, T, H)
            single_timestep_shape = shape[:2]  # (N, T)
            timestep_probs = jnp.full(single_timestep_shape, mask_ratio)
            return jnp.broadcast_to(
                timestep_probs[..., jnp.newaxis], shape
            )
        elif mode == "neuron":
            # Mask entire neurons: (N, H) -> broadcast to (N, T, H)  
            single_neuron_shape = (shape[0], shape[2])  # (N, H)
            neuron_probs = jnp.full(single_neuron_shape, mask_ratio)
            return jnp.broadcast_to(
                neuron_probs[:, jnp.newaxis, :], shape
            )
        elif mode == "timestep_only":
            # Mask same timesteps across all trials: (T,) -> broadcast to (N, T, H)
            single_timestep_shape = (shape[1],)  # (T,)
            timestep_probs = jnp.full(single_timestep_shape, mask_ratio)
            return jnp.broadcast_to(
                timestep_probs[jnp.newaxis, :, jnp.newaxis], shape
            )
        else:
            raise ValueError(f"Unsupported mask mode: {mode}")
    
    def mask_batch(
        self,
        batch: jnp.ndarray,
        contrast_mode: bool = False,
        custom_mask: Optional[jnp.ndarray] = None,
        max_spikes: int = DEFAULT_MASK_VAL - 1,
        should_mask: bool = True,
        expand_prob: float = 0.0,
        heldout_spikes: Optional[jnp.ndarray] = None,
        forward_spikes: Optional[jnp.ndarray] = None,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Mask a batch of spike data
        
        Args:
            batch: Input spikes (N, T, H)
            contrast_mode: Whether to use contrastive masking config
            custom_mask: Optional custom mask to use
            max_spikes: Maximum spike count for mask token
            should_mask: Whether to actually apply masking
            expand_prob: Probability of expanding mask spans
            heldout_spikes: Optional heldout spikes for co-smoothing
            forward_spikes: Optional forward spikes for co-smoothing  
            key: JAX random key
            
        Returns:
            Tuple of (masked_batch, labels, masked_heldin, batch_full, labels_full, spikes_full)
        """
        if key is None:
            key = jr.PRNGKey(0)
        
        # Get masking configuration
        if contrast_mode:
            mode = self.config.get('CONTRAST_MASK_MODE', 'timestep')
            mask_ratio = self.config.get('CONTRAST_MASK_RATIO', 0.2)
            max_span = self.config.get('CONTRAST_MASK_MAX_SPAN', 1)
            token_ratio = self.config.get('CONTRAST_MASK_TOKEN_RATIO', 1.0)
            random_ratio = self.config.get('CONTRAST_MASK_RANDOM_RATIO', 0.5)
        else:
            mode = self.config.get('MASK_MODE', 'timestep')
            mask_ratio = self.config.get('MASK_RATIO', 0.2)
            max_span = self.config.get('MASK_MAX_SPAN', 1)
            token_ratio = self.config.get('MASK_TOKEN_RATIO', 1.0)
            random_ratio = self.config.get('MASK_RANDOM_RATIO', 0.5)
        
        use_zero_mask = self.config.get('USE_ZERO_MASK', True)
        
        # Create labels (copy of original batch)
        labels = batch.copy()
        
        # Create full batch (includes heldout and forward if provided)
        batch_full = batch.copy()
        if heldout_spikes is not None:
            batch_full = jnp.concatenate([batch_full, heldout_spikes], axis=-1)
        if forward_spikes is not None:
            batch_full = jnp.concatenate([batch_full, forward_spikes], axis=1)
        
        labels_full = batch_full.copy()
        spikes_full = batch_full.copy()
        
        # Generate or use custom mask
        if custom_mask is None:
            key, subkey = jr.split(key)
            
            # Determine span width
            key, subkey = jr.split(key)
            should_expand = (max_span > 1 and 
                           expand_prob > 0.0 and 
                           jr.uniform(subkey) < expand_prob)
            
            if should_expand:
                key, subkey = jr.split(key)
                width = jr.randint(subkey, (), 1, max_span + 1)
                effective_ratio = mask_ratio / width
            else:
                width = 1
                effective_ratio = mask_ratio
            
            # Create mask probabilities
            mask_probs = self.create_mask_probabilities(batch.shape, mode, effective_ratio)
            mask_probs_full = self.create_mask_probabilities(batch_full.shape, mode, effective_ratio)
            
            # Sample binary masks
            key, k1, k2 = jr.split(key, 3)
            mask = jr.bernoulli(k1, mask_probs)
            mask_full = jr.bernoulli(k2, mask_probs_full)
            
            # Expand masks if needed
            if width > 1 and mode != "full":
                if mode == "timestep":
                    # For timestep mode, expand along time dimension
                    mask_2d = mask[:, :, 0]  # (N, T)
                    expanded_mask_2d = expand_mask(mask_2d, width)
                    # Ensure expanded mask is same shape as original
                    if expanded_mask_2d.shape != mask_2d.shape:
                        # Pad or crop to match
                        if expanded_mask_2d.shape[1] > mask_2d.shape[1]:
                            expanded_mask_2d = expanded_mask_2d[:, :mask_2d.shape[1]]
                        elif expanded_mask_2d.shape[1] < mask_2d.shape[1]:
                            pad_width = mask_2d.shape[1] - expanded_mask_2d.shape[1]
                            expanded_mask_2d = jnp.pad(expanded_mask_2d, ((0, 0), (0, pad_width)))
                    mask = jnp.broadcast_to(expanded_mask_2d[..., jnp.newaxis], batch.shape)
                    
                    mask_full_2d = mask_full[:, :, 0]  # (N, T) for full
                    expanded_mask_full_2d = expand_mask(mask_full_2d, width)
                    # Ensure expanded mask is same shape as original
                    if expanded_mask_full_2d.shape != mask_full_2d.shape:
                        if expanded_mask_full_2d.shape[1] > mask_full_2d.shape[1]:
                            expanded_mask_full_2d = expanded_mask_full_2d[:, :mask_full_2d.shape[1]]
                        elif expanded_mask_full_2d.shape[1] < mask_full_2d.shape[1]:
                            pad_width = mask_full_2d.shape[1] - expanded_mask_full_2d.shape[1]
                            expanded_mask_full_2d = jnp.pad(expanded_mask_full_2d, ((0, 0), (0, pad_width)))
                    mask_full = jnp.broadcast_to(expanded_mask_full_2d[..., jnp.newaxis], batch_full.shape)
            
            mask = mask.astype(bool)
            mask_full = mask_full.astype(bool)
        else:
            if custom_mask.shape != batch.shape:
                raise ValueError(f"Custom mask shape {custom_mask.shape} != batch shape {batch.shape}")
            mask = custom_mask.astype(bool)
            mask_full = mask  # Simplified for custom masks
        
        # Set labels for unmasked positions
        labels = jnp.where(mask, labels, UNMASKED_LABEL)
        labels_full = jnp.where(mask_full, labels_full, UNMASKED_LABEL)
        
        if not should_mask:
            return batch, labels, batch, batch_full, labels_full, spikes_full
        
        # Apply masking with different strategies
        masked_batch = batch.copy()
        masked_batch_full = batch_full.copy()
        
        # Strategy 1: Replace with mask tokens
        key, subkey = jr.split(key)
        token_mask = jr.bernoulli(subkey, jnp.full(batch.shape, token_ratio)) & mask
        key, subkey = jr.split(key)
        token_mask_full = jr.bernoulli(subkey, jnp.full(batch_full.shape, token_ratio)) & mask_full
        
        if use_zero_mask:
            masked_batch = jnp.where(token_mask, 0, masked_batch)
            masked_batch_full = jnp.where(token_mask_full, 0, masked_batch_full)
        else:
            masked_batch = jnp.where(token_mask, max_spikes + 1, masked_batch)
            masked_batch_full = jnp.where(token_mask_full, max_spikes + 1, masked_batch_full)
        
        # Strategy 2: Replace with random values
        key, subkey = jr.split(key)
        random_mask = (jr.bernoulli(subkey, jnp.full(batch.shape, random_ratio)) & 
                      mask & ~token_mask)
        key, subkey = jr.split(key)
        random_mask_full = (jr.bernoulli(subkey, jnp.full(batch_full.shape, random_ratio)) & 
                           mask_full & ~token_mask_full)
        
        # Generate random spike counts
        key, k1, k2 = jr.split(key, 3)
        max_val = jnp.max(batch)
        random_spikes = jr.randint(k1, batch.shape, 0, max_val + 1)
        random_spikes_full = jr.randint(k2, batch_full.shape, 0, jnp.max(batch_full) + 1)
        
        masked_batch = jnp.where(random_mask, random_spikes, masked_batch)
        masked_batch_full = jnp.where(random_mask_full, random_spikes_full, masked_batch_full)
        
        # Store masked heldin portion
        masked_heldin = masked_batch.copy()
        
        # Handle heldout and forward spikes in final batch
        if heldout_spikes is not None:
            # Heldout spikes are all masked (set to zero)
            zeros_heldout = jnp.zeros_like(heldout_spikes)
            masked_batch = jnp.concatenate([masked_batch, zeros_heldout], axis=-1)
            labels = jnp.concatenate([labels, heldout_spikes], axis=-1)
        
        if forward_spikes is not None:
            # Forward spikes are all masked (set to zero)
            zeros_forward = jnp.zeros_like(forward_spikes)
            masked_batch = jnp.concatenate([masked_batch, zeros_forward], axis=1)
            labels = jnp.concatenate([labels, forward_spikes], axis=1)
        
        return (masked_batch, labels, masked_heldin, 
                masked_batch_full, labels_full, spikes_full)


def create_contrastive_masks(
    batch: jnp.ndarray,
    config: Dict[str, Any],
    key: Optional[jax.random.PRNGKey] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Create two different masked views for contrastive learning
    
    Args:
        batch: Input batch (N, T, H)
        config: Configuration dictionary
        key: JAX random key
        
    Returns:
        Tuple of (masked_batch1, labels1, masked_batch2, labels2) for contrastive learning
    """
    if key is None:
        key = jr.PRNGKey(0)
    
    masker = JAXMasker(config)
    
    # Create two different masked views using contrastive masking mode
    key1, key2 = jr.split(key)
    
    # First contrastive view
    results1 = masker.mask_batch(
        batch, contrast_mode=True, key=key1
    )
    masked_batch1, labels1 = results1[0], results1[1]
    
    # Second contrastive view (different random mask)
    results2 = masker.mask_batch(
        batch, contrast_mode=True, key=key2
    )
    masked_batch2, labels2 = results2[0], results2[1]
    
    return masked_batch1, labels1, masked_batch2, labels2
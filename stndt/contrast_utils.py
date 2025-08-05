#!/usr/bin/env python3
"""
PyTorch STNDT-style contrast pair creation
"""
import jax.numpy as jnp
import jax.random as jr

def create_contrast_pairs_pytorch_style(batch_data, key):
    """
    Create contrast pairs exactly like PyTorch STNDT
    
    Args:
        batch_data: (batch_size, time_steps, num_neurons)
        key: JAX random key
    
    Returns:
        contrast_src1, contrast_src2: Two differently masked versions of the same data
    """
    
    # PyTorch STNDT contrast config
    config = {
        'CONTRAST_MASK_RATIO': 0.2,        # 20% of timesteps masked
        'CONTRAST_MASK_MODE': 'timestep',   # Mask entire timesteps
        'CONTRAST_MASK_TOKEN_RATIO': 1.0,   # Always use mask token
        'CONTRAST_MASK_RANDOM_RATIO': 0.5,  # 50% chance of random token
        'MAX_SPIKES': 5,                    # Maximum spike count
        'USE_ZERO_MASK': True,              # Use 0 as mask token
    }
    
    # Split key for two different maskings
    key1, key2 = jr.split(key)
    
    # Create two different masked versions of the same data
    contrast_src1 = apply_contrast_masking(batch_data, key1, config)
    contrast_src2 = apply_contrast_masking(batch_data, key2, config)
    
    return contrast_src1, contrast_src2

def apply_contrast_masking(data, key, config):
    """Apply timestep-level contrast masking like PyTorch STNDT"""
    batch_size, time_steps, num_neurons = data.shape
    
    # Create timestep-level masks
    mask = create_timestep_mask(
        batch_size, time_steps, num_neurons, 
        config['CONTRAST_MASK_RATIO'], 
        key
    )
    
    # Apply masking strategy
    masked_data = apply_masking_strategy(data, mask, config, key)
    
    return masked_data

def create_timestep_mask(batch_size, time_steps, num_neurons, mask_ratio, key):
    """Create timestep-level masks (mask entire timesteps across all neurons)"""
    num_masked_timesteps = int(time_steps * mask_ratio)
    
    masks = []
    for b in range(batch_size):
        key, subkey = jr.split(key)
        
        # Random timesteps to mask
        masked_timesteps = jr.choice(subkey, time_steps, (num_masked_timesteps,), replace=False)
        
        # Create mask for this batch
        timestep_mask = jnp.zeros(time_steps, dtype=bool)
        timestep_mask = timestep_mask.at[masked_timesteps].set(True)
        
        # Expand to all neurons
        batch_mask = jnp.tile(timestep_mask[:, None], (1, num_neurons))
        masks.append(batch_mask)
    
    return jnp.array(masks)

def apply_masking_strategy(data, mask, config, key):
    """Apply the PyTorch STNDT masking strategy"""
    key1, key2 = jr.split(key)
    
    # Create masking decisions
    mask_token_prob = config['CONTRAST_MASK_TOKEN_RATIO']
    random_token_prob = config['CONTRAST_MASK_RANDOM_RATIO']
    
    # For each masked position, decide: mask token or random token
    decision = jr.uniform(key1, mask.shape)
    use_mask_token = decision < mask_token_prob
    use_random_token = (decision >= mask_token_prob) & mask
    
    # Apply masking
    masked_data = data.copy()
    
    # Apply mask tokens (0)
    mask_value = 0 if config['USE_ZERO_MASK'] else config['MAX_SPIKES'] + 1
    masked_data = jnp.where(mask & use_mask_token, mask_value, masked_data)
    
    # Apply random tokens
    random_values = jr.randint(key2, data.shape, 0, config['MAX_SPIKES'] + 1)
    masked_data = jnp.where(mask & use_random_token, random_values, masked_data)
    
    return masked_data
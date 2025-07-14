import jax.numpy as jnp
import jax.random as jr

def create_reconstruction_mask(batch_data, mask_ratio=0.25, mask_token_ratio=0.5, key=None):
    """
    Create BERT-style random masking for reconstruction training
    Similar to PyTorch STNDT implementation
    """
    if key is None:
        key = jr.PRNGKey(42)
    
    batch_size, seq_len, num_neurons = batch_data.shape
    
    # Create random mask - True means position will be masked
    key, mask_key = jr.split(key)
    random_values = jr.uniform(mask_key, (batch_size, seq_len, num_neurons))
    mask_positions = random_values < mask_ratio
    
    # Create input data with masking
    input_data = batch_data.copy()
    
    # Create labels - only masked positions have targets
    mask_labels = jnp.full_like(batch_data, -1)  # -1 means ignore in loss
    mask_labels = jnp.where(mask_positions, batch_data, -1)
    
    # Apply masking strategy to input
    key, token_key, random_key = jr.split(key, 3)
    
    # For masked positions, decide what to replace with
    token_choice = jr.uniform(token_key, mask_positions.shape)
    
    # 50% of masked positions → set to 0 (zero masking)
    zero_mask = mask_positions & (token_choice < mask_token_ratio)
    input_data = jnp.where(zero_mask, 0, input_data)
    
    # 50% of masked positions → set to random spike counts
    random_spikes = jr.randint(random_key, mask_positions.shape, 0, 4)  # 0-3 spikes
    random_mask = mask_positions & (token_choice >= mask_token_ratio)
    input_data = jnp.where(random_mask, random_spikes, input_data)
    
    return input_data, mask_labels

def create_forward_prediction_mask(batch_data, num_forward_steps=5):
    """
    Create forward prediction masking for evaluation
    """
    batch_size, seq_len, num_neurons = batch_data.shape
    context_length = seq_len - num_forward_steps
    
    # Input: zero out the last num_forward_steps
    input_data = batch_data.copy()
    input_data = input_data.at[:, context_length:, :].set(0)
    
    # Labels only for forward positions
    mask_labels = jnp.full_like(batch_data, -1)
    mask_labels = mask_labels.at[:, context_length:, :].set(batch_data[:, context_length:, :])
    
    return input_data, mask_labels

def create_contrastive_masks(batch_data, mask_ratio=0.05, key=None):
    """
    Create two different random masks for contrastive learning
    """
    if key is None:
        key = jr.PRNGKey(42)
    
    key1, key2 = jr.split(key)
    
    # Create two different random masks
    input_1, labels_1 = create_reconstruction_mask(batch_data, mask_ratio, key=key1)
    input_2, labels_2 = create_reconstruction_mask(batch_data, mask_ratio, key=key2)
    
    return (input_1, labels_1), (input_2, labels_2)

def create_hybrid_batch(batch_data, mode='reconstruction', mask_ratio=0.25, 
                       num_forward_steps=5, contrastive=False, key=None):
    """
    Create batch for hybrid training/evaluation
    
    Args:
        batch_data: (batch_size, seq_len, num_neurons)
        mode: 'reconstruction' for training, 'forward' for evaluation
        mask_ratio: Fraction of positions to mask for reconstruction
        num_forward_steps: Number of future steps to predict
        contrastive: Whether to create contrastive pairs
        key: Random key
    
    Returns:
        If mode='reconstruction' and contrastive=False:
            (input_data, mask_labels)
        If mode='reconstruction' and contrastive=True:
            ((input_1, labels_1), (input_2, labels_2))
        If mode='forward':
            (input_data, mask_labels)
    """
    if key is None:
        key = jr.PRNGKey(42)
    
    if mode == 'reconstruction':
        if contrastive:
            return create_contrastive_masks(batch_data, mask_ratio, key)
        else:
            return create_reconstruction_mask(batch_data, mask_ratio, key=key)
    
    elif mode == 'forward':
        return create_forward_prediction_mask(batch_data, num_forward_steps)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

# Test functions
def test_masking_strategies():
    """Test different masking strategies"""
    # Create test data
    key = jr.PRNGKey(42)
    batch_data = jr.randint(key, (2, 10, 3), 0, 4)  # 2 samples, 10 timesteps, 3 neurons
    
    print("Original data:")
    print(batch_data[0])
    print()
    
    # Test reconstruction masking
    key, subkey = jr.split(key)
    input_recon, labels_recon = create_hybrid_batch(
        batch_data, mode='reconstruction', mask_ratio=0.3, key=subkey
    )
    
    print("Reconstruction masking:")
    print("Input (masked):")
    print(input_recon[0])
    print("Labels (only for masked positions):")
    print(labels_recon[0])
    print()
    
    # Test forward prediction masking
    input_forward, labels_forward = create_hybrid_batch(
        batch_data, mode='forward', num_forward_steps=3
    )
    
    print("Forward prediction masking:")
    print("Input (future zeroed):")
    print(input_forward[0])
    print("Labels (only for future positions):")
    print(labels_forward[0])
    print()
    
    # Test contrastive masking
    key, subkey = jr.split(key)
    (input_1, labels_1), (input_2, labels_2) = create_hybrid_batch(
        batch_data, mode='reconstruction', contrastive=True, mask_ratio=0.3, key=subkey
    )
    
    print("Contrastive masking:")
    print("View 1 input:")
    print(input_1[0])
    print("View 2 input:")
    print(input_2[0])

if __name__ == "__main__":
    test_masking_strategies()
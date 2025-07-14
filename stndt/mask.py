import jax.numpy as jnp

def create_forward_prediction_mask(batch_data, num_forward_steps=1, mask_strategy='all'):
    # batch_data: (batch, 125, 11)
    context_length = batch_data.shape[1] - num_forward_steps 
    input_data = batch_data.copy() 
    input_data = input_data.at[:, context_length:, :].set(0)
    
    # Labels only for the positions we want to predict  
    mask_labels = jnp.full_like(batch_data, -1)
    forward_data = batch_data[:, context_length:, :]
    
    if mask_strategy == 'all':
        # Original behavior - predict all positions
        mask_labels = mask_labels.at[:, context_length:, :].set(forward_data)
    
    elif mask_strategy == 'only_spikes':
        # Only set labels where there are actual spikes (non-zero)
        non_zero_mask = forward_data > 0
        mask_labels = mask_labels.at[:, context_length:, :].set(
            jnp.where(non_zero_mask, forward_data, -1)
        )
    
    elif mask_strategy == 'balanced':
        # Simplified balanced strategy: subsample non-spike positions
        # to roughly match the number of spike positions
        spike_mask = forward_data > 0
        non_spike_mask = forward_data == 0
        
        # Count spikes vs non-spikes across the batch
        spike_ratio = jnp.sum(spike_mask) / jnp.sum(non_spike_mask)
        
        # If we have way more non-spikes than spikes, randomly mask some non-spikes
        if spike_ratio < 0.5:  # Less than 50% are spikes
            import jax.random as jr
            # Use a fixed key for reproducibility (this is a limitation)
            key = jr.PRNGKey(42)
            # Create random mask for non-spike positions
            random_vals = jr.uniform(key, shape=forward_data.shape)
            # Keep non-spike positions with probability equal to spike_ratio * 2
            keep_prob = jnp.minimum(spike_ratio * 2, 1.0)
            keep_non_spike_mask = random_vals < keep_prob
            
            # Final mask: keep all spikes + sampled non-spikes
            final_mask = spike_mask | (non_spike_mask & keep_non_spike_mask)
            mask_labels = mask_labels.at[:, context_length:, :].set(
                jnp.where(final_mask, forward_data, -1)
            )
        else:
            # If relatively balanced, use all positions
            mask_labels = mask_labels.at[:, context_length:, :].set(forward_data)
    
    elif mask_strategy == 'weighted':
        # Use all positions but weight spike positions more in loss computation
        # This is handled in the loss function, not here
        mask_labels = mask_labels.at[:, context_length:, :].set(forward_data)
    
    return input_data, mask_labels
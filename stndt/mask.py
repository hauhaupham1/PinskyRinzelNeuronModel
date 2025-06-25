import jax.numpy as jnp

def create_forward_prediction_mask(batch_data, num_forward_steps=1):
    # batch_data: (batch, 125, 11)
    context_length = batch_data.shape[1] - num_forward_steps 
    input_data = batch_data.copy() 
    input_data = input_data.at[:, context_length:, :].set(0)
    
    # Labels only for the positions we want to predict  
    mask_labels = jnp.full_like(batch_data, -1)
    mask_labels = mask_labels.at[:, context_length:, :].set(batch_data[:, context_length:, :])
    
    return input_data, mask_labels
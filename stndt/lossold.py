#Poisson loss function
import jax.numpy as jnp

def poisson_nll_loss(predictions, targets, log_input=False, eps=1e-8):
    """
    Poisson negative log-likelihood loss
    
    Args:
        predictions: Predicted rates (batch_size, time, neurons)
        targets: Target spike counts (batch_size, time, neurons) 
        log_input: Whether predictions are in log space
        eps: Small epsilon for numerical stability
        
    Returns:
        Loss values (same shape as inputs)
    """
    if log_input:
        # predictions are log rates
        log_rates = predictions
        rates = jnp.exp(log_rates)
    else:
        # predictions are rates, ensure positive
        rates = jnp.maximum(predictions, eps)
        log_rates = jnp.log(rates + eps) 
    
    # Poisson NLL: rate - target * log(rate) + log_factorial(target)
    # We ignore the factorial term since it doesn't depend on predictions
    loss = rates - targets * log_rates
    
    return loss

def compute_forecasting_loss(predictions, mask_labels, config=None):
    """
    Compute loss only on masked (non -1) positions for forecasting
    
    Args:
        predictions: Model predictions (batch_size, time, neurons)
        mask_labels: Mask labels, -1 = ignore, real values = compute loss
        log_input: Whether predictions are in log space
        
    Returns:
        Scalar loss value
    """
    # Create mask for valid positions (not -1)
    valid_mask = mask_labels != -1
    
    # Get valid predictions and targets
    #using jax compatibility
    valid_predictions = jnp.where(valid_mask, predictions, 0.0)
    valid_targets = jnp.where(valid_mask, mask_labels, 0.0)
    
    # Compute Poisson loss on valid positions
    log_input = config.get('LOGRATE', False) if config is not None else False
    losses = poisson_nll_loss(valid_predictions, valid_targets, log_input=log_input)
    
    # Return mean loss (jnp.mean handles empty arrays gracefully)
    masked_losses = jnp.where(valid_mask, losses, 0.0)
    return jnp.sum(masked_losses) / jnp.sum(valid_mask)

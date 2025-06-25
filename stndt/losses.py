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
        config: Configuration dict containing LOGRATE and optionally MASK_STRATEGY
        
    Returns:
        Scalar loss value
    """
    # Create mask for valid positions (not -1)
    valid_mask = mask_labels != -1
    
    # Compute Poisson loss on all positions
    log_input = config.get('LOGRATE', False) if config is not None else False
    
    # Use jnp.where to handle invalid positions - set targets to 0 where mask is False
    safe_targets = jnp.where(valid_mask, mask_labels, 0.0)
    losses = poisson_nll_loss(predictions, safe_targets, log_input=log_input)
    
    # Apply weighting if using weighted strategy
    if config is not None and config.get('MASK_STRATEGY') == 'weighted':
        # Weight spike positions more heavily
        spike_weight = config.get('SPIKE_WEIGHT', 5.0)  # Default 5x weight for spikes
        weights = jnp.where(safe_targets > 0, spike_weight, 1.0)
        weighted_losses = losses * weights * valid_mask
        
        # Compute weighted mean
        total_loss = jnp.sum(weighted_losses)
        total_weight = jnp.sum(weights * valid_mask)
        mean_loss = jnp.where(total_weight > 0, total_loss / total_weight, 0.0)
    else:
        # Original unweighted loss
        masked_losses = losses * valid_mask
        total_loss = jnp.sum(masked_losses)
        valid_count = jnp.sum(valid_mask)
        mean_loss = jnp.where(valid_count > 0, total_loss / valid_count, 0.0)
    
    return mean_loss

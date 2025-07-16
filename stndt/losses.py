#Poisson loss function
import jax.numpy as jnp
import equinox as eqx
import jax


def poisson_nll_loss(predictions, targets, log_input=False, mask=None):
    """
    JAX equivalent of PyTorch's PoissonNLLLoss
    
    Args:
        predictions: predicted rates (or log-rates if log_input=True)
        targets: actual spike counts (may contain -100 for invalid positions)
        log_input: if True, predictions are log-rates
        mask: optional mask for valid positions (if None, computes on all positions)
    """
    eps = 1e-8
    if log_input:
        # If predictions are log-rates, convert to rates
        log_rates = predictions
        rates = jnp.exp(log_rates)
        # Poisson NLL: -log(P(target|rate)) = rate - target*log_rate
        # For invalid targets (-100), this will produce invalid values but we handle it downstream
        loss = rates - targets * log_rates
    else:
        # If predictions are rates directly
        rates = jnp.maximum(predictions, eps)
        # Poisson NLL: rate - target*log(rate)
        # For invalid targets (-100), this will produce invalid values but we handle it downstream
        loss = rates - targets * jnp.log(rates)
    
    # Return element-wise loss (like PyTorch) - filtering happens in calling code
    return loss

def discrete_spike_loss(logits, targets, mask=None):
    targets = targets.astype(jnp.int32)
    max_spike_class = logits.shape[-1] - 1
    targets = jnp.clip(targets, 0, max_spike_class)
    
    # Compute log probabilities
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    
    # Create indices for gathering
    B, T, N = targets.shape
    batch_idx = jnp.arange(B)[:, None, None]
    time_idx = jnp.arange(T)[None, :, None]
    neuron_idx = jnp.arange(N)[None, None, :]
    
    # Select log probability of true classes
    selected_log_probs = log_probs[batch_idx, time_idx, neuron_idx, targets]
    
    # Negative log likelihood
    loss = -selected_log_probs
    
    if mask is not None:
        loss = loss * mask
        valid_count = jnp.sum(mask)
        return jnp.where(valid_count > 0, jnp.sum(loss) / valid_count, 0.0)
    
    return jnp.mean(loss)

def compute_forecasting_loss(predictions, mask_labels, config = None, loss_type='discrete'):

    valid_mask = (mask_labels != -1).astype(jnp.float32)
    # if loss_type == 'discrete':
    safe_targets = jnp.where(mask_labels != -1, mask_labels, 0)
    loss = discrete_spike_loss(predictions, safe_targets, mask=valid_mask)
    return loss
    # elif loss_type == 'poisson':
    #     #TODO: Implement poisson loss
    #     pass


    

def discrete_aware_regression_loss(predictions, targets, mask=None):
    """Regression loss that handles discrete spike nature better"""
    
    # Handle masking first
    if mask is not None:
        valid_mask = (mask != -1).astype(jnp.float32)
        # Only compute loss on valid positions
        predictions = jnp.where(valid_mask, predictions, 0.0)
        targets = jnp.where(valid_mask, targets, 0.0)
    else:
        valid_mask = jnp.ones_like(predictions)
    
    # Round predictions for discrete-aware loss
    rounded_preds = jnp.round(predictions)
    
    # MSE on rounded predictions (element-wise)
    mse_loss = (rounded_preds - targets) ** 2
    
    # Add penalty for being far from integers (element-wise)
    rounding_penalty = (predictions - rounded_preds) ** 2
    
    # Poisson NLL for spike count modeling (element-wise)
    eps = 1e-8
    poisson_loss = predictions - targets * jnp.log(predictions + eps)
    
    # Combined loss (element-wise)
    total_loss = poisson_loss + 0.1 * rounding_penalty
    
    # Apply mask and average
    if mask is not None:
        total_loss = total_loss * valid_mask
        return jnp.sum(total_loss) / jnp.sum(valid_mask)
    else:
        return jnp.mean(total_loss)

def weighted_mse_loss(predictions, targets, mask=None, spike_weight=5.0):
    """MSE that weights spike positions more heavily"""
    
    # Square error
    sq_error = (predictions - targets) ** 2
    
    # Weight errors on spike positions more
    weights = jnp.where(targets > 0, spike_weight, 1.0)
    weighted_error = sq_error * weights
    
    if mask is not None:
        valid_mask = (mask != -1).astype(jnp.float32)
        return jnp.sum(weighted_error * valid_mask) / jnp.sum(weights * valid_mask)
    
    return jnp.sum(weighted_error) / jnp.sum(weights)
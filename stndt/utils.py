# Author: Trung Le
# Original file available at https://github.com/trungle93/STNDT
# Adapted by Hau Pham
"""
JAX/Equinox utilities for STNDT
Translated from PyTorch implementation
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Callable
import optax


def binary_mask_to_attn_mask(x: jnp.ndarray) -> jnp.ndarray:
    """
    Convert binary mask to attention mask
    
    Args:
        x: Binary mask where 1 = attend, 0 = mask
        
    Returns:
        Attention mask where 0 = attend, -inf = mask
    """
    return jnp.where(x == 0, -jnp.inf, 0.0)


def get_inverse_sqrt_schedule(
    warmup_steps: int = 1000,
    lr_init: float = 1e-8,
    lr_max: float = 5e-4
) -> optax.Schedule:
    """
    Inverse square root learning rate schedule with warmup
    
    Args:
        warmup_steps: Number of warmup steps
        lr_init: Initial learning rate
        lr_max: Maximum learning rate after warmup
        
    Returns:
        Optax schedule function
    """
    def schedule_fn(step):
        # Warmup phase: linear increase
        if step < warmup_steps:
            lr_step = (lr_max - lr_init) / warmup_steps
            return lr_init + step * lr_step
        else:
            # Decay phase: inverse square root
            decay_factor = lr_max * (warmup_steps ** 0.5)
            return decay_factor * (step ** -0.5)
    
    return schedule_fn


def create_cosine_schedule(
    peak_value: float,
    warmup_steps: int,
    total_steps: int,
    end_value: float = 0.0
) -> optax.Schedule:
    """
    Create cosine learning rate schedule with warmup
    
    Args:
        peak_value: Peak learning rate
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        end_value: Final learning rate value
        
    Returns:
        Optax schedule function
    """
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=peak_value,
        transition_steps=warmup_steps
    )
    
    cosine = optax.cosine_decay_schedule(
        init_value=peak_value,
        decay_steps=total_steps - warmup_steps,
        alpha=end_value / peak_value
    )
    
    return optax.join_schedules(
        schedules=[warmup, cosine],
        boundaries=[warmup_steps]
    )


def create_optimizer(
    config: Dict[str, Any],
    total_steps: Optional[int] = None
) -> optax.GradientTransformation:
    """
    Create optimizer based on configuration
    
    Args:
        config: Configuration dictionary
        total_steps: Total training steps (needed for some schedulers)
        
    Returns:
        Optax optimizer
    """
    lr_config = config.get('LR', {})
    init_lr = lr_config.get('INIT', 1e-3)
    use_schedule = lr_config.get('SCHEDULE', True)
    scheduler_type = lr_config.get('SCHEDULER', 'cosine')
    warmup_steps = lr_config.get('WARMUP', 1000)
    weight_decay = config.get('WEIGHT_DECAY', 0.0)
    eps = config.get('EPS', 1e-8)
    
    # Create learning rate schedule
    if use_schedule and total_steps is not None:
        if scheduler_type == 'cosine':
            schedule = create_cosine_schedule(
                peak_value=init_lr,
                warmup_steps=warmup_steps,
                total_steps=total_steps
            )
        elif scheduler_type == 'invsqrt':
            schedule = get_inverse_sqrt_schedule(
                warmup_steps=warmup_steps,
                lr_init=1e-8,
                lr_max=init_lr
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    else:
        schedule = optax.constant_schedule(init_lr)
    
    # Create optimizer
    if weight_decay > 0:
        optimizer = optax.adamw(
            learning_rate=schedule,
            weight_decay=weight_decay,
            eps=eps
        )
    else:
        optimizer = optax.adam(
            learning_rate=schedule,
            eps=eps
        )
    
    # Add gradient clipping if specified
    max_grad_norm = config.get('MAX_GRAD_NORM', 0.0)
    if max_grad_norm > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optimizer
        )
    
    return optimizer


def poisson_nll_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    log_input: bool = False,
    eps: float = 1e-8
) -> jnp.ndarray:
    """
    Poisson negative log-likelihood loss
    
    Args:
        predictions: Predicted rates
        targets: Target spike counts
        log_input: Whether predictions are in log space
        eps: Small epsilon for numerical stability
        
    Returns:
        Loss values
    """
    if log_input:
        # predictions are log rates
        log_rates = predictions
        rates = jnp.exp(log_rates)
    else:
        # predictions are rates
        rates = jnp.maximum(predictions, eps)
        log_rates = jnp.log(rates)
    
    # Poisson NLL: rate - target * log(rate) + log_factorial(target)
    # We ignore the factorial term since it doesn't depend on predictions
    loss = rates - targets * log_rates
    
    return loss


def compute_r2_score(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    axis: Optional[int] = None
) -> jnp.ndarray:
    """
    Compute R² coefficient of determination
    
    Args:
        predictions: Predicted values
        targets: Target values  
        axis: Axis along which to compute R² (None for global)
        
    Returns:
        R² scores
    """
    # Sum of squares of residuals
    ss_res = jnp.sum((targets - predictions) ** 2, axis=axis)
    
    # Total sum of squares
    target_mean = jnp.mean(targets, axis=axis, keepdims=True)
    ss_tot = jnp.sum((targets - target_mean) ** 2, axis=axis)
    
    # R² = 1 - (SS_res / SS_tot)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add epsilon to avoid division by zero
    
    return r2


def bits_per_spike(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    log_input: bool = False,
    eps: float = 1e-8
) -> jnp.ndarray:
    """
    Compute bits per spike metric
    
    Args:
        predictions: Predicted rates
        targets: Target spike counts
        log_input: Whether predictions are in log space
        eps: Small epsilon for numerical stability
        
    Returns:
        Bits per spike values
    """
    if log_input:
        log_rates = predictions
    else:
        rates = jnp.maximum(predictions, eps)
        log_rates = jnp.log(rates)
    
    # Mean firing rate for null model
    mean_rate = jnp.mean(targets, axis=0, keepdims=True)
    log_mean_rate = jnp.log(jnp.maximum(mean_rate, eps))
    
    # Log-likelihood under model
    ll_model = targets * log_rates - jnp.exp(log_rates)
    
    # Log-likelihood under null model (constant rate)
    ll_null = targets * log_mean_rate - mean_rate
    
    # Bits per spike = (LL_model - LL_null) / (total_spikes * log(2))
    total_spikes = jnp.sum(targets, axis=0, keepdims=True)
    bits_per_spike = (ll_model - ll_null) / (total_spikes * jnp.log(2) + eps)
    
    return bits_per_spike


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for STNDT training"""
    return {
        # Model configuration
        'NUM_LAYERS': 6,
        'NUM_HEADS': 8,
        'HIDDEN_SIZE': 128,
        'DROPOUT': 0.1,
        'DROPOUT_RATES': 0.2,
        'DROPOUT_EMBEDDING': 0.2,
        'EMBED_DIM': 0,
        'LINEAR_EMBEDDER': False,
        'LEARNABLE_POSITION': False,
        'MAX_SPIKE_COUNT': 20,
        'LOGRATE': True,
        'PRE_NORM': False,
        'SCALE_NORM': False,
        'FULL_CONTEXT': False,
        'CONTEXT_FORWARD': 4,
        'CONTEXT_BACKWARD': 8,
        
        # Contrastive learning
        'USE_CONTRAST_PROJECTOR': False,
        'LINEAR_PROJECTOR': False,
        'CONTRAST_LAYER': 'decoder',
        'TEMPERATURE': 0.07,
        'LAMBDA': 1e-8,
        
        # Loss configuration
        'LOSS': {
            'TYPE': 'poisson'
        },
        'DECODER': {
            'LAYERS': 1
        },
        
        # Training configuration
        'BATCH_SIZE': 64,
        'NUM_UPDATES': 10000,
        'MAX_GRAD_NORM': 200.0,
        'WEIGHT_DECAY': 0.0,
        'EPS': 1e-8,
        
        # Learning rate
        'LR': {
            'INIT': 1e-3,
            'SCHEDULE': True,
            'SCHEDULER': 'cosine',
            'WARMUP': 1000
        },
        
        # Masking
        'USE_ZERO_MASK': True,
        'MASK_RATIO': 0.2,
        'MASK_TOKEN_RATIO': 1.0,
        'MASK_RANDOM_RATIO': 0.5,
        'MASK_MODE': 'timestep',
        'MASK_MAX_SPAN': 1,
        
        # Contrastive masking
        'DO_CONTRAST': True,
        'CONTRAST_MASK_RATIO': 0.2,
        'CONTRAST_MASK_TOKEN_RATIO': 1.0,
        'CONTRAST_MASK_RANDOM_RATIO': 0.5,
        'CONTRAST_MASK_MODE': 'timestep',
        'CONTRAST_MASK_MAX_SPAN': 1,
        
        # Logging and checkpointing
        'LOG_INTERVAL': 50,
        'VAL_INTERVAL': 10,
        'CHECKPOINT_INTERVAL': 1000,
        'PATIENCE': 750,
    }


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.is_better = lambda new, best: new > best + min_delta
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop early
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten a dictionary with dot-separated keys
    
    Args:
        d: Flattened dictionary
        sep: Separator used in keys
        
    Returns:
        Nested dictionary
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result
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
        loss = rates - targets * log_rates
    else:
        # If predictions are rates directly
        rates = jnp.maximum(predictions, eps)
        loss = rates - targets * jnp.log(rates)

    return loss



def bits_per_spike_jax(rates, spikes):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = poisson_nll_loss(rates, spikes)
    null_rates = jnp.tile(
        jnp.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = poisson_nll_loss(null_rates, spikes)
    return (nll_null - nll_model) / jnp.nansum(spikes) / jnp.log(2)
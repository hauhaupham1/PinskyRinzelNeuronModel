import jax.numpy as jnp
import jax
import signax

def signature_mse_loss(pred_signatures, target_signatures):
    """Simple MSE loss between predicted and target signatures."""
    return jnp.mean((pred_signatures - target_signatures) ** 2)

def hybrid_loss(lift_pred: jax.Array, 
                lift_target: jax.Array, 
                signature_depth = 3):
    signature_fn = lambda x: signax.signature(x, depth=signature_depth)
    sig_pre = signature_fn(lift_pred)
    sig_tar = signature_fn(lift_target)
    return jnp.mean((sig_pre - sig_tar) ** 2)
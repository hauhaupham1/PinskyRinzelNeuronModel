import jax
import jax.numpy as jnp
import signax
def lift_flatten(lift: jax.Array) -> jax.Array:
    return lift.reshape(-1)

def lift_unflatten(lift_flat: jax.Array, num_neurons : int) -> jax.Array:
    num_cols = num_neurons + 1
    num_rows = lift_flat.shape[0] // num_cols
    return lift_flat.reshape((num_rows, num_cols))

def lift_flatten_dimenstion(lift: jax.Array) -> int:
    return len(lift_flatten(lift))



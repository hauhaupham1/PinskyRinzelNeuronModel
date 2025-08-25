import jax.numpy as jnp


def create_context_mask(
    context_forward: int, context_backward: int, sequence_length: int
):
    forward_array = jnp.tril(
        jnp.ones((sequence_length, sequence_length), dtype=bool), k=context_forward
    )
    # print(forward_array)
    backward_array = jnp.triu(
        jnp.ones((sequence_length, sequence_length), dtype=bool), k=-context_backward
    )
    # print(backward_array)
    combined = forward_array & backward_array
    return combined

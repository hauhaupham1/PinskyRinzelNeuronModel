import equinox as eqx
import jax.numpy as jnp
import jax



class TransformerBlock(eqx.Module):


    attention: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP


    def __init__(self, d_model, num_heads, d_ff, dropout, key):
        keys = jax.random.split(key, 2)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=d_model,
            key_size=d_model,
            value_size=d_model,
            output_size=d_model,
            key=keys[0]
        )

        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.norm2 = eqx.nn.LayerNorm(d_model)

        self.mlp = eqx.nn.MLP(
            in_size=d_model,
            out_size=d_model,
            width_size=d_ff,
            depth=2,
            key=keys[1],
        )


    def __call__(self, x, mask=None):
        attn_output = jax.vmap(self.attention)(x, x, x)
        x = jax.vmap(jax.vmap(self.norm1))(x + attn_output)
        mlp_output = jax.vmap(jax.vmap(self.mlp))(x)
        normed2 = jax.vmap(jax.vmap(self.norm2))(x + mlp_output)
        return normed2
    


class SimpleTransformer(eqx.Module):
    blocks: list
    input_proj: eqx.nn.Linear
    final_linear: eqx.nn.Linear
    future_steps: int  # Store as parameter
    def __init__(self, input_dim, output_dim, d_model, num_layers, num_heads, d_ff, future_steps, key):
        keys = jax.random.split(key, num_layers + 2)
        self.future_steps = future_steps
        # Input projection
        self.input_proj = eqx.nn.Linear(input_dim, d_model, key=keys[0])
        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, 0.1, keys[i+1])
            for i in range(num_layers)
        ]
        # Output projection: d_model -> (future_steps * output_dim)
        self.final_linear = eqx.nn.Linear(d_model, future_steps * output_dim, key=keys[-1])
    def __call__(self, x):
        x = jax.vmap(jax.vmap(self.input_proj))(x)  # (batch, context_length, d_model)
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        # Take last timestep
        last_hidden = x[:, -1, :]  # (batch, d_model)
        # Project to output
        output = jax.vmap(self.final_linear)(last_hidden)  # (batch, future_steps*output_dim)
        # Reshape to (batch, future_steps, output_dim)
        batch_size = output.shape[0]
        output_dim = output.shape[1] // self.future_steps
        return output.reshape(batch_size, self.future_steps, output_dim)
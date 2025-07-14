import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class FeedForward(eqx.Module):
    linear1: eqx.nn.Linear
    activation: eqx.nn.Lambda
    linear2: eqx.nn.Linear
    dropout: eqx.nn.Dropout


    def __init__(self, n_embed, dropout=0.1, expansion_factor = 4, hidden_size=None, key = None):
        if key is None:
            key = jr.PRNGKey(4)

        key1, key2 = jr.split(key)
        # If hidden_size is provided, use it directly. Otherwise use expansion_factor
        if hidden_size is not None:
            hidden_dim = hidden_size
        else:
            hidden_dim = expansion_factor * n_embed
            
        self.linear1 = eqx.nn.Linear(in_features=n_embed, out_features=hidden_dim, use_bias=True, key=key1)
        self.activation = eqx.nn.PReLU(init_alpha=0.01)
        self.linear2 = eqx.nn.Linear(in_features=hidden_dim, out_features=n_embed, use_bias=True, key=key2)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x, key=None):
        if key is not None:
            keys = jr.split(key, 2)
        else:
            keys = [None, None]
        x = jax.vmap(jax.vmap(self.linear1))(x)
        x = jax.vmap(jax.vmap(self.activation))(x)
        if key is not None:
            x = self.dropout(x, key=keys[0])
        else:
            x = self.dropout(x, inference=True)
        x = jax.vmap(jax.vmap(self.linear2))(x)
        if key is not None:
            x = self.dropout(x, key=keys[1])
        else:
            x = self.dropout(x, inference=True)

        return x
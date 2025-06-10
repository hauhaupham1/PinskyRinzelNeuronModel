import equinox as eqx
import jax
import jax.numpy as jnp



#Attention mechanism
class Head(eqx.Module):
    k : eqx.nn.Linear
    q : eqx.nn.Linear
    v : eqx.nn.Linear
    mask : jnp.ndarray
    dropout : eqx.nn.Dropout
    
    def __init__(self, n_embed, head_size, max_length = 100, dropout = 0.2, key = None):
        if key is None:
            key = jax.random.PRNGKey(0)
        k_key, q_key, v_key, dropout_key = jax.random.split(key, 4)
        self.k = eqx.nn.Linear(in_features=n_embed, out_features=head_size, use_bias=False, key=k_key)
        self.q = eqx.nn.Linear(in_features=n_embed, out_features=head_size, use_bias=False, key=q_key)
        self.v = eqx.nn.Linear(in_features=n_embed, out_features=head_size, use_bias=False, key=v_key)
        self.mask = jnp.tril(jnp.ones((max_length, max_length), dtype=jnp.float32))
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x):
        #B = batch size, T = sequence length, C = n_embed
        B, T, C = x.shape
        k = jax.vmap(jax.vmap(self.k))(x)  # (B, T, head_size)
        q = jax.vmap(jax.vmap(self.q))(x)  # (B, T, head_size)
        v = jax.vmap(jax.vmap(self.v))(x)  # (B, T, head_size)

        k_transposed = jnp.transpose(k, axes = (0, 2, 1))
        # compute attention scores ("affinities")
        # scores = jnp.einsum('bth,bsh->bts', q, k) * (k.shape[-1] ** -0.5)  # (B, T, T)
        scores = q @ k_transposed * k.shape[-1]**-0.5 # (B, T, T)
        scores = jnp.where(self.mask[:T, :T] == 0, -jnp.inf, scores)
        # print("Scores:" , scores[:5, :5])  # Print first 5 scores for debugging
        scores = jax.nn.softmax(scores, axis=-1) # (B, T, T)
        # print("Scores:" , scores[:5, :5])       
        scores = self.dropout(scores, inference=True)

        out = scores @ v # (B, T, head_size)
        return out



class MultiHeadAttention(eqx.Module):
    
    def __init__(self, n_embed, num_heads, max_length = 100, dropout=0.2, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key_heads, key_linear = jax.random.split(key)
        head_size = n_embed // num_heads
        self.heads = [Head(n_embed=n_embed, head_size=head_size, max_length=max_length, dropout=dropout, key=jax.random.fold_in(key_heads, i)) for i in range(num_heads)]
        self.linear = eqx.nn.Linear(in_features=(num_heads * head_size), out_features=n_embed, use_bias=False, key=jax.random.fold_in(key_linear, num_heads))
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x):
        head_out = [head(x) for head in self.heads]  
        out = jnp.concatenate(head_out, axis=-1)
        out = jax.vmap(jax.vmap(self.linear))(out)
        out = self.dropout(out, inference=True)
        return out
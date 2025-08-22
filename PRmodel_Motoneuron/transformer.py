import equinox as eqx
import jax
import jax.numpy as jnp


# Attention mechanism
class Head(eqx.Module):
    k: eqx.nn.Linear
    q: eqx.nn.Linear
    v: eqx.nn.Linear
    mask: jnp.ndarray
    dropout: eqx.nn.Dropout

    def __init__(self, n_embed, head_size, max_length=100, dropout=0.2, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        k_key, q_key, v_key, dropout_key = jax.random.split(key, 4)
        self.k = eqx.nn.Linear(
            in_features=n_embed, out_features=head_size, use_bias=False, key=k_key
        )
        self.q = eqx.nn.Linear(
            in_features=n_embed, out_features=head_size, use_bias=False, key=q_key
        )
        self.v = eqx.nn.Linear(
            in_features=n_embed, out_features=head_size, use_bias=False, key=v_key
        )
        self.mask = jnp.tril(jnp.ones((max_length, max_length), dtype=jnp.float32))
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x, key=None):
        # B = batch size, T = sequence length, C = n_embed
        B, T, C = x.shape
        k = jax.vmap(jax.vmap(self.k))(x)  # (B, T, head_size)
        q = jax.vmap(jax.vmap(self.q))(x)  # (B, T, head_size)
        v = jax.vmap(jax.vmap(self.v))(x)  # (B, T, head_size)

        k_transposed = jnp.transpose(k, axes=(0, 2, 1))
        # compute attention scores ("affinities")
        # scores = jnp.einsum('bth,bsh->bts', q, k) * (k.shape[-1] ** -0.5)  # (B, T, T)
        scores = q @ k_transposed * k.shape[-1] ** -0.5  # (B, T, T)
        scores = jnp.where(self.mask[:T, :T] == 0, -jnp.inf, scores)
        # print("Scores:" , scores[:5, :5])  # Print first 5 scores for debugging
        scores = jax.nn.softmax(scores, axis=-1)  # (B, T, T)
        # print("Scores:" , scores[:5, :5])
        scores = (
            self.dropout(scores, key=key)
            if key is not None
            else self.dropout(scores, inference=True)
        )

        out = scores @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(eqx.Module):
    heads: list
    linear: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, n_embed, num_heads, max_length=100, dropout=0.2, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key_heads, key_linear = jax.random.split(key)
        head_size = n_embed // num_heads
        self.heads = [
            Head(
                n_embed=n_embed,
                head_size=head_size,
                max_length=max_length,
                dropout=dropout,
                key=jax.random.fold_in(key_heads, i),
            )
            for i in range(num_heads)
        ]
        self.linear = eqx.nn.Linear(
            in_features=(num_heads * head_size),
            out_features=n_embed,
            use_bias=False,
            key=jax.random.fold_in(key_linear, num_heads),
        )
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x, key=None):
        head_out = [head(x, key=key) for head in self.heads]
        out = jnp.concatenate(head_out, axis=-1)
        out = jax.vmap(jax.vmap(self.linear))(out)
        out = (
            self.dropout(out, key=key)
            if key is not None
            else self.dropout(out, inference=True)
        )
        return out


# Feedforward network
class FeedForward(eqx.Module):
    linear1: eqx.nn.Linear
    activation: eqx.nn.Lambda
    linear2: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, n_embed, dropout=0.2, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)

        self.linear1 = eqx.nn.Linear(
            in_features=n_embed, out_features=4 * n_embed, use_bias=False, key=key1
        )
        self.activation = eqx.nn.Lambda(jax.nn.relu)
        self.linear2 = eqx.nn.Linear(
            in_features=4 * n_embed, out_features=n_embed, use_bias=False, key=key2
        )
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x, key=None):
        x = jax.vmap(jax.vmap(self.linear1))(x)  # (B, T, 4 * n_embed)
        x = jax.vmap(jax.vmap(self.activation))(x)  # (B, T, 4 * n_embed)
        x = jax.vmap(jax.vmap(self.linear2))(x)  # (B, T, n_embed)
        x = (
            self.dropout(x, key=key)
            if key is not None
            else self.dropout(x, inference=True)
        )
        return x  # (B, T, n_embed)


# Transformer block
class Block(eqx.Module):
    mha: MultiHeadAttention
    ffwd: FeedForward
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(self, n_embed, n_head, max_length=100, dropout=0.2, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key_mha, key_ffwd = jax.random.split(key, 2)
        self.mha = MultiHeadAttention(
            n_embed=n_embed,
            num_heads=n_head,
            max_length=max_length,
            dropout=dropout,
            key=key_mha,
        )
        self.ffwd = FeedForward(n_embed=n_embed, dropout=dropout, key=key_ffwd)
        self.ln1 = eqx.nn.LayerNorm(shape=(n_embed,))
        self.ln2 = eqx.nn.LayerNorm(shape=(n_embed,))

    def __call__(self, x, key=None):
        if key is not None:
            key1, key2 = jax.random.split(key, 2)
        else:
            key1, key2 = None, None

        x = x + self.mha(jax.vmap(jax.vmap(self.ln1))(x), key=key1)
        x = x + self.ffwd(jax.vmap(jax.vmap(self.ln2))(x), key=key2)
        return x


# Transformer model
class Transformer(eqx.Module):
    signature_embedding: eqx.nn.Linear
    position_embedding: jnp.ndarray
    blocks: eqx.nn.Sequential
    ln_f: eqx.nn.LayerNorm
    output_embedding: eqx.nn.Linear

    def __init__(
        self,
        signature_dim,
        n_embed,
        n_head,
        n_layer,
        max_length=100,
        dropout=0.2,
        key=None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)

        keys = jax.random.split(key, 3 + n_layer)
        key_sig_embedding, key_output_embedding, key_position_embedding = (
            keys[0],
            keys[1],
            keys[2],
        )
        key_blocks = keys[3:]

        # input embedding
        self.signature_embedding = eqx.nn.Linear(
            in_features=signature_dim,
            out_features=n_embed,
            use_bias=False,
            key=key_sig_embedding,
        )
        self.position_embedding = (
            jax.random.normal(key=key_position_embedding, shape=(max_length, n_embed))
            * 0.02
        )
        self.blocks = eqx.nn.Sequential(
            [
                Block(
                    n_embed=n_embed,
                    n_head=n_head,
                    max_length=max_length,
                    dropout=dropout,
                    key=key_blocks[i],
                )
                for i in range(n_layer)
            ]
        )
        self.ln_f = eqx.nn.LayerNorm(shape=(n_embed,))
        self.output_embedding = eqx.nn.Linear(
            in_features=n_embed,
            out_features=signature_dim,
            use_bias=False,
            key=key_output_embedding,
        )

    def __call__(self, x):
        B, T, _ = x.shape
        # Apply embedding
        sig_embedding = jax.vmap(jax.vmap(self.signature_embedding))(x)
        pos_embedding = self.position_embedding[:T, :]
        combined_embedding = sig_embedding + pos_embedding
        # Pass through transformer blocks
        combined_embedding = self.blocks(combined_embedding)
        # Final layer normalization
        combined_embedding = jax.vmap(jax.vmap(self.ln_f))(combined_embedding)
        output = jax.vmap(jax.vmap(self.output_embedding))(combined_embedding)
        return output


# Example usage:
transformer = Transformer(
    signature_dim=39,  # From your 2-neuron, depth-3 signatures
    n_embed=64,  # Model dimension
    n_head=8,  # Number of attention heads
    n_layer=4,  # Number of transformer layers
    max_length=50,  # Max sequence length
    key=jax.random.PRNGKey(42),
)
# Test with dummy data
batch_size, seq_len = 2, 10
dummy_signatures = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, 39))
output = transformer(dummy_signatures)
print(f"Output shape: {output.shape}")  # Should be (2, 10, 39)

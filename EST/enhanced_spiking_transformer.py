import jax
from jaxtyping import Float, Int, Array
import equinox.nn as nn
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray
from equinox._module import field



class ConvolutionalSelfAttention(eqx.Module):
    conv_q: nn.Conv1d
    conv_k: nn.Conv1d
    conv_v: nn.Conv1d
    out_proj: nn.Linear
    dropout: nn.Dropout
    
    num_heads: int = field(static=True)
    head_dim: int = field(static=True)
    scale: float = field(static=True)


    def __init__(self, num_heads: int, head_dim: int, dropout_rate: float, key: PRNGKeyArray, use_bias: bool = True):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        d_model = num_heads * head_dim
        
        key = jr.split(key, 4)
        self.conv_q = nn.Conv1d(
            in_channels=d_model,
            out_channels=num_heads * head_dim,
            kernel_size=5,
            padding=((4, 0),),
            use_bias=use_bias,
            key=key[0]
        )
        self.conv_k = nn.Conv1d(
            in_channels=d_model,
            out_channels=num_heads * head_dim,
            kernel_size=5,
            padding=((4, 0),),
            use_bias=use_bias,
            key=key[1]
        )
        self.conv_v = nn.Conv1d(
            in_channels=d_model,
            out_channels=num_heads * head_dim,
            kernel_size=5,
            padding='same',
            use_bias=use_bias,
            key=key[2]
        )
        
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, use_bias=use_bias, key=key[3])
        self.dropout = nn.Dropout(dropout_rate)

    def __call__(self, x: Float[Array, "seq_len features"], key: PRNGKeyArray, inference = False) -> Float[Array, "seq_len features"]:
        # Transpose for Conv1d: (seq_len, features) → (features, seq_len)
        q = self.conv_q(x.T).T
        k = self.conv_k(x.T).T
        v = self.conv_v(x.T).T

        # Reshape for multi-head attention
        seq_len = q.shape[0]
        
        # Reshape to (seq_len, num_heads, head_dim)
        q = q.reshape(seq_len, self.num_heads, self.head_dim)
        k = k.reshape(seq_len, self.num_heads, self.head_dim)
        v = v.reshape(seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (num_heads, seq_len, head_dim) for easier computation
        q = q.transpose(1, 0, 2)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)
        
        # Compute attention scores: Q @ K^T
        scores = jnp.matmul(q, k.transpose(0, 2, 1)) * self.scale  # (num_heads, seq_len, seq_len)
        
        # Apply causal mask (prevent looking into future)
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        scores = jnp.where(mask, scores, -jnp.inf)
        # print("Scores:", scores)

        # Apply softmax to get attention weights
        attn_weights = jax.nn.softmax(scores, axis=-1)
        # print("Attention Weights:", attn_weights)
        
        # Apply attention to values
        out = jnp.matmul(attn_weights, v)  # (num_heads, seq_len, head_dim)
        
        # Transpose back to (seq_len, num_heads, head_dim)
        out = out.transpose(1, 0, 2)
        
        # Reshape to (seq_len, num_heads * head_dim)
        out = out.reshape(seq_len, self.num_heads * self.head_dim)
        
        # Apply output projection
        out = jax.vmap(self.out_proj)(out)
        out = self.dropout(out, inference= not inference, key=key)
        
        return out
    

class TemporalProcessingUnit(eqx.Module):
    cross_timestep: nn.Conv1d
    temporal_integration: nn.Linear
    spike_activation: nn.Lambda


    def __init__(self, n_features: int, kernel_size: int = 3, *, key: PRNGKeyArray, use_bias: bool = True):
        keys = jr.split(key, 2)
        self.cross_timestep = nn.Conv1d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=kernel_size,
            padding=((kernel_size - 1, 0),),
            use_bias=use_bias,
            key=keys[0]
        )
        self.temporal_integration = nn.Linear(n_features, n_features, use_bias=use_bias, key=keys[1])
        self.spike_activation = nn.Lambda(jax.nn.relu)


    def __call__(self, x: Float[Array, "seq_len features"]) -> Float[Array, "seq_len features"]:

        conv_out = self.cross_timestep(x.T).T  # Transpose for Conv1d: (seq_len, features) → (features, seq_len)
        integrated = jax.vmap(self.temporal_integration)(conv_out)
        # output = jax.vmap(self.spike_activation)(integrated)
        output = integrated

        return output
    

#TESTING#
# input = jnp.ones((5, 5))  # Example input with 5 timesteps and 5 features
# tpu = TemporalProcessingUnit(n_features=5, key=jr.PRNGKey(0))
# output = tpu(input)
# print("TPU Output Shape:", output.shape)  # Should be (5, 5)
# print("TPU Output:", output)

class FeedForwardBlock(eqx.Module):
    mlp: eqx.nn.Linear
    output: eqx.nn.Linear
    # layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
            self, 
            hidden_size: int,
            intermediate_size: int,
            dropout_rate: float,
            key: jax.random.PRNGKey,
    ):
        mlp_key, output_key = jr.split(key)
        self.mlp = eqx.nn.Linear(in_features=hidden_size, out_features=intermediate_size, key=mlp_key)
        self.output = eqx.nn.Linear(in_features=intermediate_size, out_features=hidden_size, key=output_key)
        # self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
            self,
            inputs: Float[Array, "hidden_size"],
            inference: bool = True,
            key: "jax.random.PRNGKey | None" = None,
            ) -> Float[Array, "hidden_size"]:
        hidden = jax.vmap(self.mlp)(inputs)
        hidden = jax.nn.gelu(hidden)

        output = jax.vmap(self.output)(hidden)
        output = self.dropout(output, inference=not inference, key=key)

        # output += inputs
        # output = jax.vmap(self.layernorm)(output)
        return output
    

class ESTLayer(eqx.Module):
    conv_attention: ConvolutionalSelfAttention
    tpu: TemporalProcessingUnit
    norm1: nn.LayerNorm
    norm2: nn.LayerNorm
    norm3: nn.LayerNorm
    dropout: nn.Dropout

    ffn: FeedForwardBlock

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float, key: PRNGKeyArray, use_bias: bool=True):
        keys = jr.split(key, 3)
        self.conv_attention = ConvolutionalSelfAttention(
            num_heads=num_heads,
            head_dim = d_model // num_heads,
            dropout_rate=dropout_rate,
            key=keys[0],
            use_bias=use_bias
        )
        self.tpu = TemporalProcessingUnit(
            n_features=d_model,
            key=keys[1],
            use_bias=use_bias
        )
        self.norm1 = nn.LayerNorm(d_model, use_bias=use_bias)
        self.norm2 = nn.LayerNorm(d_model, use_bias=use_bias)
        self.norm3 = nn.LayerNorm(d_model, use_bias=use_bias)
        self.dropout = nn.Dropout(dropout_rate)

        self.ffn = FeedForwardBlock(
            hidden_size=d_model,
            intermediate_size=d_model * 4,  
            dropout_rate=dropout_rate,
            key=keys[2]
        )

    def __call__(self, x: Float[Array, "seq_len features"], *, key: PRNGKeyArray = None, inference: bool = True) -> Float[Array, "seq_len features"]:
        if key is not None:
            key1, key2, key3, key4, key5 = jr.split(key, 5)
        else:
            key1 = key2 = key3 = key4 = key5 = None

        # Convolutional Self-Attention
        x_normed = jax.vmap(self.norm1)(x)
        attn_out = self.conv_attention(x_normed, key=key1, inference=not inference)
        x = x + self.dropout(attn_out, inference=not inference, key=key2)
        

        # Temporal Processing Unit
        x_normed = jax.vmap(self.norm2)(x)
        tpu_out = self.tpu(x_normed)
        x = x + self.dropout(tpu_out, inference=not inference, key=key3)

        # Feed Forward Network
        x_normed = jax.vmap(self.norm3)(x)
        ffn_out = self.ffn(x_normed, inference=not inference, key=key4)
        x = x + self.dropout(ffn_out, inference=not inference, key=key5)

        return x

class EST(eqx.Module):
    input_proj: nn.Linear
    layers: list[ESTLayer]
    output_proj: nn.Linear
    output_activation: nn.Lambda
    dropout: nn.Dropout
    num_layers: int = field(static=True)


    def __init__(self,
                 input_dim: int,
                 d_model: int,
                 num_heads: int,
                 num_layers: int,
                 output_dim: int,
                 dropout_rate: float = 0.1,
                 *,
                 use_bias: bool = True,
                 key: PRNGKeyArray):
        self.num_layers = num_layers
        keys = jr.split(key, num_layers + 3)

        self.input_proj = nn.Linear(input_dim, d_model, use_bias=use_bias, key=keys[0])

        self.layers = [
            ESTLayer(
                d_model=d_model,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                key=keys[i + 1],
                use_bias=use_bias
            ) for i in range(num_layers)
        ]
        self.output_proj = nn.Linear(d_model, output_dim, use_bias=use_bias, key=keys[-2])
        self.output_activation = nn.Lambda(jax.nn.relu)
        self.dropout = nn.Dropout(dropout_rate)


    def __call__(self, x: Float[Array, 'batch seq_len features'], *, key: PRNGKeyArray = None, inference: bool 
    = True) -> Float[Array, 'batch seq_len output_dim']:
        batch_size = x.shape[0]

        if key is not None:
            # Split keys for each batch element and each layer
            layer_keys = jr.split(key, self.num_layers)
        else:
            layer_keys = [None] * self.num_layers

        # Input Projection
        x = jax.vmap(jax.vmap(self.input_proj))(x)

        # Pass through layers
        for layer, layer_key in zip(self.layers, layer_keys):
            x = jax.vmap(lambda x_i: layer(x_i, key=layer_key, inference=inference))(x)

        # Output Projection
        output = jax.vmap(jax.vmap(self.output_proj))(x)
        output = jax.vmap(jax.vmap(self.output_activation))(output)
        return output
    
#TESTING#
# input_data = jnp.zeros((2, 2, 2))  # Example
# est = EST(
#     input_dim=2,
#     d_model=128,
#     num_heads=8,
#     num_layers=6,
#     output_dim=2,
#     dropout_rate=0.1,
#     key=jr.PRNGKey(0)
# )
# output = est(input_data, key=jr.PRNGKey(80), inference=False)
# print("EST Output Shape:", output.shape)  # Should be (2, 2, 2)
# print("EST Output:", output)
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax
from jaxtyping import Float, Int, Array

class FeedForwardBlock(eqx.Module):
    mlp: eqx.nn.Linear
    output: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
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
        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
            self,
            inputs: Float[Array, "hidden_size"],
            enable_dropout: bool = True,
            key: "jax.random.PRNGKey | None" = None,
            ) -> Float[Array, "hidden_size"]:
        hidden = self.mlp(inputs)
        hidden = jax.nn.gelu(hidden)

        output = self.output(hidden)
        output = self.dropout(output, inference=not enable_dropout, key=key)

        output += inputs
        output = self.layernorm(output)
        return output
        


class AttentionBlock(eqx.Module):
    """A single transformer attention block."""

    attention: eqx.nn.MultiheadAttention
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    rope: eqx.nn.RotaryPositionalEmbedding
    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        self.num_heads = num_heads
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=attention_dropout_rate,
            key=key,
        )
        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.rope = eqx.nn.RotaryPositionalEmbedding(
            embedding_size=int(hidden_size / num_heads),
        )

    def __call__(
        self,
        inputs: Float[Array, "seq_len hidden_size"],
        mask: Int[Array, " seq_len"] | None,
        enable_dropout: bool = False,
        key: "jax.random.PRNGKey" = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        def process_heads(query_heads: Float[Array, 'seq_len num_heads qk_size'],
                          key_heads: Float[Array, 'seq_len num_heads qk_size'],
                          value_heads: Float[Array, 'seq_len num_heads vo_size']) -> tuple[
                              Float[Array, 'seq_len num_heads qk_size'],
                                Float[Array, 'seq_len num_heads qk_size'],
                                Float[Array, 'seq_len num_heads vo_size'],
                          ]:
            query_heads = jax.vmap(self.rope, in_axes=1, out_axes=1)(query_heads)
            key_heads = jax.vmap(self.rope, in_axes=1, out_axes=1)(key_heads)
            return query_heads, key_heads, value_heads

        if mask is not None:
            mask = self.make_self_attention_mask(mask)
        attention_key, dropout_key = (
            (None, None) if key is None else jax.random.split(key)
        )

        attention_output = self.attention(
            query=inputs,
            key_=inputs,
            value=inputs,
            mask=mask,
            inference=not enable_dropout,
            key=attention_key,
            process_heads=process_heads,
        )

        result = attention_output
        result = self.dropout(result, inference=not enable_dropout, key=dropout_key)
        result = result + inputs
        result = jax.vmap(self.layernorm)(result)
        return result

    def make_self_attention_mask(
        self, mask: Int[Array, " seq_len"]
    ) -> Float[Array, "num_heads seq_len seq_len"]:
        """Create self-attention mask from sequence-level mask."""
        mask = jnp.multiply(
            jnp.expand_dims(mask, axis=-1), jnp.expand_dims(mask, axis=-2)
        )
        mask = jnp.expand_dims(mask, axis=-3)
        mask = jnp.repeat(mask, repeats=self.num_heads, axis=-3)
        return mask.astype(jnp.float32)



class TransformerLayer(eqx.Module):
    """ A single transformer layer"""
    
    attention_block: AttentionBlock
    ff_block: FeedForwardBlock

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            num_heads: int,
            dropout_rate: float,
            attention_dropout_rate: float,
            key: jax.random.PRNGKey,
    ):
        
        attention_key, ff_key = jr.split(key)
        self.attention_block = AttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            key=attention_key,
        )
        self.ff_block = FeedForwardBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
            key=ff_key,
        )


    def __call__(
            self, 
            inputs: Float[Array, "seq_len hidden_size"],
            mask: Int[Array, "seq_len"] | None = None,
            *,
            enable_dropout: bool = False,
            key: "jax.random.PRNGKey | None" = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        attn_key, ff_key = (None, None) if key is None else jr.split(key)
        attention_output = self.attention_block(
            inputs=inputs,
            mask=mask,
            enable_dropout=enable_dropout,
            key=attn_key,
        )
        seq_len = inputs.shape[0]
        ff_keys = None if key is None else jr.split(ff_key, num=seq_len)
        output = jax.vmap(self.ff_block, in_axes=(0, None, 0))(
            attention_output, enable_dropout, ff_keys
        )
        return output
    

class EncoderBlock(eqx.Module):
    layers: list[TransformerLayer]

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            num_heads: int,
            num_layers: int,
            dropout_rate: float,
            attention_dropout_rate: float,
            key: jax.random.PRNGKey,
    ):
        keys = jr.split(key, num=num_layers)
        self.layers = [
            TransformerLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                key=keys[i],
            )
            for i in range(num_layers)
        ]


    def __call__(
        self, 
        inputs: Float[Array, "seq_len hidden_size"],
        mask: Int[Array, "seq_len"] | None = None,
        *,
        enable_dropout: bool = False,
        key: "jax.random.PRNGKey | None" = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        keys = None if key is None else jr.split(key, num=len(self.layers))

        for i, layer in enumerate(self.layers):
            inputs = layer(
                inputs=inputs,
                mask=mask,
                enable_dropout=enable_dropout,
                key=None if keys is None else keys[i],
            )
        return inputs
    


class Transformer(eqx.Module):
    input_proj : eqx.nn.Linear
    encoder: EncoderBlock
    output_proj: eqx.nn.Linear
    activation: eqx.nn.Lambda

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        input_proj_key, encoder_key, output_proj_key = jr.split(key, 3)
        self.input_proj = eqx.nn.Linear(
            in_features=input_dim,
            out_features=hidden_size,
            key=input_proj_key,
            )
        
        self.encoder = EncoderBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            key=encoder_key,
        )

        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=input_dim,
            key=output_proj_key,
        )
        
        self.activation = eqx.nn.Lambda(jax.nn.relu)


    def __call__(
        self,
        inputs: Float[Array, "batch seq_len input_dim"],
        mask : Int[Array, "seq_len"] | None = None,
        *,
        enable_dropout: bool = False,
        key: "jax.random.PRNGKey | None" = None,
    ) -> Float[Array, "batch seq_len input_dim"]:
        
        input_dim = inputs.shape[-1]
        batch_size = inputs.shape[0]
        inputs = jax.vmap(jax.vmap(self.input_proj))(inputs)
        if key is not None:
            key = jr.split(key, num=batch_size)
            outputs = jax.vmap(lambda x, k: self.encoder(x, mask=mask, enable_dropout=enable_dropout, key=k), in_axes=(0, 0))(inputs, key)
        else:
            outputs = jax.vmap(lambda x: self.encoder(x, mask=mask, enable_dropout=enable_dropout, key=None))(inputs)
        
        
        outputs = jax.vmap(jax.vmap(self.output_proj))(outputs)
        outputs = jax.vmap(jax.vmap(self.activation))(outputs)
        return outputs
    


#######TESTING CODE#######

# transformer = Transformer(
#     input_dim=5,
#     hidden_size=64,
#     intermediate_size=128,
#     num_heads=8,
#     num_layers=4,
#     dropout_rate=0.1,
#     attention_dropout_rate=0.1,
#     key=jax.random.PRNGKey(42)
# )

# #dummny data
# input = jr.uniform(jr.PRNGKey(0), (2, 5, 5))  # batch_size=2, seq_len=5, input_dim=5
# output = transformer(input, enable_dropout=False, key=jax.random.PRNGKey(1))
# print("Output shape:", output.shape)  # Should be (2, 5, 5)
# print("Output:", output)  # Should print the output tensor
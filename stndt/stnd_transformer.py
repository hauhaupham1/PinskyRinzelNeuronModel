#!/usr/bin/env python3
# Author: Trung Le
# Original file available at https://github.com/trungle93/STNDT
# Adapted by Hau Pham
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import math
from typing import Optional, Tuple, Dict, Any
import equinox.nn as nn
from functools import partial
from equinox.nn._attention import dot_product_attention_weights
import optax
from losses import poisson_nll_loss
from mask import UNMASKED_LABEL


class PositionalEncoding(eqx.Module):
    pe: jnp.ndarray
    dropout: eqx.nn.Dropout
    learnable: bool
    pos_embedding: Optional[eqx.nn.Embedding]

    def __init__(self, config, trial_length, d_model, key=None):
        self.dropout = eqx.nn.Dropout(config.get("DROPOUT_EMBEDDING", 0.1))
        self.learnable = config.get("LEARNABLE_POSITION", True)

        position = jnp.arange(0, trial_length, dtype=jnp.float32).reshape(-1, 1)
        if config.get("POSITION", {}).get("OFFSET"):
            position = position + 1

        if self.learnable:
            self.pe = position.astype(jnp.int32).squeeze()
            embed_key = jr.split(key)[0] if key is not None else None
            self.pos_embedding = eqx.nn.Embedding(trial_length, d_model, key=embed_key)
        else:
            pe = jnp.zeros((trial_length, d_model))
            div_term = jnp.exp(
                jnp.arange(0, d_model, 2).astype(jnp.float32)
                * (-math.log(10000.0) / d_model)
            )
            pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
            if d_model % 2 == 0:
                pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
            else:
                pe = pe.at[:, 1::2].set(jnp.cos(position * div_term[:-1]))
            # Keep as (seq_len, d_model) for our batch-first format
            self.pe = pe
            self.pos_embedding = None

    def __call__(self, x, key=None):
        if self.learnable:
            seq_len = x.shape[0]
            pos_indices = self.pe[:seq_len]
            x = x + jax.vmap(self.pos_embedding)(pos_indices)
        else:
            x = x + self.pe[: x.shape[0]]

        if key is not None:
            x = self.dropout(x, key=key)
        else:
            x = self.dropout(x, inference=True)
        return x


class MultiheadAttentionWithWeights(eqx.nn.MultiheadAttention):
    """Extended MultiheadAttention that always returns both output and attention weights"""

    def __call__(
        self,
        query,
        key_,
        value,
        mask=None,
        *,
        key=None,
        inference=None,
        deterministic=None,
        process_heads=None,
    ):
        """Same as parent but always returns (output, weights)"""

        if deterministic is not None:
            inference = deterministic

        query_seq_length, _ = query.shape
        kv_seq_length, _ = key_.shape
        kv_seq_length2, _ = value.shape
        if kv_seq_length != kv_seq_length2:
            raise ValueError("key and value must both be sequences of equal length.")

        query_heads = self._project(self.query_proj, query)
        key_heads = self._project(self.key_proj, key_)
        value_heads = self._project(self.value_proj, value)

        if process_heads is not None:
            q_shape, k_shape, v_shape = (
                query_heads.shape,
                key_heads.shape,
                value_heads.shape,
            )
            query_heads, key_heads, value_heads = process_heads(
                query_heads, key_heads, value_heads
            )

            if (
                query_heads.shape != q_shape
                or key_heads.shape != k_shape
                or value_heads.shape != v_shape
            ):
                raise ValueError(
                    "process_heads must not change the shape of the heads."
                )

        # Use Equinox's built-in functions
        def attention_with_weights(query_h, key_h, value_h, mask=None, key=None):
            weights = dot_product_attention_weights(query_h, key_h, mask)
            if self.dropout is not None:
                weights = self.dropout(weights, key=key, inference=inference)
            attn = jnp.einsum("sS,Sd->sd", weights, value_h)
            return attn, weights

        keys = None if key is None else jax.random.split(key, query_heads.shape[1])

        if mask is not None and mask.ndim == 3:
            results = jax.vmap(attention_with_weights, in_axes=1, out_axes=1)(
                query_heads, key_heads, value_heads, mask=mask, key=keys
            )
        else:
            results = jax.vmap(
                partial(attention_with_weights, mask=mask), in_axes=1, out_axes=1
            )(query_heads, key_heads, value_heads, key=keys)

        attn, weights = results
        attn = attn.reshape(query_seq_length, -1)
        output = jax.vmap(self.output_proj)(attn)
        return output, weights.mean(axis=1)  # Always return both output and weights


class SpikeProjection(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, num_neurons, n_embed, key=None):
        if key is None:
            key = jr.PRNGKey(1)
        self.linear = eqx.nn.Linear(
            in_features=num_neurons, out_features=n_embed, use_bias=False, key=key
        )

    def __call__(self, spike_counts):
        return jax.vmap(self.linear)(spike_counts)  # (B, T, N) -> (B, T, n_embed)


# Fusion Mechanism
class STNDTEncoder(eqx.nn.MultiheadAttention):
    num_input: int
    trial_length: int
    spatial_self_attn: MultiheadAttentionWithWeights
    spatial_norm_1: eqx.nn.LayerNorm
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    activation: eqx.nn.PReLU
    linear1: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    linear2: eqx.nn.Linear
    ts_norm1: eqx.nn.LayerNorm
    ts_norm2: eqx.nn.LayerNorm
    ts_linear1: eqx.nn.Linear
    ts_linear2: eqx.nn.Linear
    ts_dropout1: eqx.nn.Dropout
    ts_dropout2: eqx.nn.Dropout
    ts_dropout3: eqx.nn.Dropout
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    ts_dropout1: eqx.nn.Dropout
    ts_dropout2: eqx.nn.Dropout
    ts_dropout3: eqx.nn.Dropout
    config: Dict[str, Any] = eqx.static_field()

    def __init__(self, config, d_model, trial_length, key=None):
        if key is None:
            key = jr.PRNGKey(5)

        keys = jr.split(key, 3)
        super().__init__(
            num_heads=config.get("NUM_HEADS", 8),
            query_size=d_model,
            key_size=d_model,
            value_size=d_model,
            output_size=d_model,
            dropout_p=config.get("DROPOUT", 0.1),
            key=keys[0],
        )

        self.config = config
        self.num_input = d_model
        self.trial_length = trial_length
        self.spatial_self_attn = MultiheadAttentionWithWeights(
            num_heads=config.get("NUM_HEADS", 8),
            query_size=trial_length,
            key_size=trial_length,
            value_size=trial_length,
            output_size=trial_length,
            dropout_p=config.get("DROPOUT", 0.1),
            key=keys[1],
        )
        self.spatial_norm_1 = eqx.nn.LayerNorm(self.trial_length)

        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.norm2 = eqx.nn.LayerNorm(d_model)

        self.activation = eqx.nn.PReLU()

        key_linear1, key_linear2, key3 = jr.split(keys[2], 3)

        self.linear1 = eqx.nn.Linear(
            d_model, self.config.get("HIDDEN_SIZE"), key=key_linear1
        )
        self.dropout = eqx.nn.Dropout(self.config.get("DROPOUT"))
        self.linear2 = eqx.nn.Linear(
            self.config.get("HIDDEN_SIZE"), d_model, key=key_linear2
        )

        self.ts_norm1 = eqx.nn.LayerNorm(d_model)
        self.ts_norm2 = eqx.nn.LayerNorm(d_model)
        key_ts_linear1, key_ts_linear2 = jr.split(key3, 2)
        self.ts_linear1 = eqx.nn.Linear(
            d_model, config.get("HIDDEN_SIZE"), key=key_ts_linear1
        )
        self.ts_linear2 = eqx.nn.Linear(
            config.get("HIDDEN_SIZE"), d_model, key=key_ts_linear2
        )
        self.ts_dropout1 = eqx.nn.Dropout(config.get("DROPOUT"))
        self.ts_dropout2 = eqx.nn.Dropout(config.get("DROPOUT"))
        self.ts_dropout3 = eqx.nn.Dropout(config.get("DROPOUT"))

        self.dropout1 = eqx.nn.Dropout(config.get("DROPOUT"))
        self.dropout2 = eqx.nn.Dropout(config.get("DROPOUT"))
        self.ts_dropout1 = eqx.nn.Dropout(config.get("DROPOUT"))
        self.ts_dropout2 = eqx.nn.Dropout(config.get("DROPOUT"))
        self.ts_dropout3 = eqx.nn.Dropout(config.get("DROPOUT"))

    def get_input_size(self):
        return self.num_input

    def attend(self, src, context_mask=None, **kwargs):
        # src shape: (B, T, N) - batch first format
        # MultiheadAttention expects (T, N), so we vmap over batch

        if "key" in kwargs:
            key = kwargs.pop("key")
            if key is not None:
                keys = jr.split(key, src.shape[0])  # Split for each batch
            else:
                keys = [None] * src.shape[0]

            def single_attend(src_single, key_single):
                return super(STNDTEncoder, self).__call__(
                    query=src_single,
                    key_=src_single,
                    value=src_single,
                    mask=context_mask,
                    key=key_single,
                )

            attn_res = jax.vmap(single_attend, in_axes=(0, 0), out_axes=0)(src, keys)
        else:
            attn_res = jax.vmap(
                lambda src_single: super(STNDTEncoder, self).__call__(
                    query=src_single,
                    key_=src_single,
                    value=src_single,
                    mask=context_mask,
                    inference=True,
                ),
                in_axes=0,
                out_axes=0,
            )(src)

        return attn_res

    def spatial_attend(self, src, context_mask=None, key=None, **kwargs):
        # src shape: (B, N, T) for spatial attention
        if key is not None:
            keys = jr.split(key, src.shape[0])  # Split for each batch
        else:
            keys = [None] * src.shape[0]

        def single_spatial_attend(src_single, key_single):
            return self.spatial_self_attn(
                query=src_single,
                key_=src_single,
                value=src_single,
                mask=context_mask,
                key=key_single,
            )

        results = jax.vmap(single_spatial_attend, in_axes=(0, 0), out_axes=(0, 0))(
            src, keys
        )
        output, weights = results
        return output, weights

    def __call__(
        self,
        src,
        spatial_src,
        src_mask=None,
        spatial_src_mask=None,
        key=None,
        prenorm=None,
    ):
        if key is None:
            key = jr.PRNGKey(6)

        # Get prenorm from config if not provided
        if prenorm is None:
            prenorm = self.config.get("PRE_NORM", False)

        # Split keys properly for all dropout operations
        keys = jr.split(key, 8)  # Need 8 different keys
        (
            key_t,
            key_drop_1,
            key_drop_ff1,
            key_drop_ff2,
            key_spatial,
            key_ts_drop1,
            key_ts_drop2,
            key_ts_drop3,
        ) = keys

        residual = src
        if prenorm:
            # For src shape (B, T, N), we need to apply norm over the N dimension
            # vmap over B, then vmap over T
            src = jax.vmap(jax.vmap(self.norm1))(src)

        # Apply temporal attention
        t_out = self.attend(src, context_mask=src_mask, key=key_t)
        src = residual + (
            self.dropout1(t_out, key=key_drop_1)
            if key is not None
            else self.dropout1(t_out, inference=True)
        )

        if not prenorm:
            src = jax.vmap(jax.vmap(self.norm1))(src)
        residual = src
        if prenorm:
            src = jax.vmap(jax.vmap(self.norm2))(src)

        src2 = jax.vmap(jax.vmap(self.linear1))(src)
        src2 = jax.vmap(jax.vmap(self.activation))(src2)
        src2 = (
            self.dropout(src2, key=key_drop_ff1)
            if key is not None
            else self.dropout(src2, inference=True)
        )
        src2 = jax.vmap(jax.vmap(self.linear2))(src2)

        src = residual + (
            self.dropout(src2, key=key_drop_ff2)
            if key is not None
            else self.dropout(src2, inference=True)
        )

        if not prenorm:
            src = jax.vmap(jax.vmap(self.norm2))(src)

        spatial_residual = spatial_src
        if prenorm:
            spatial_src = jax.vmap(jax.vmap(self.spatial_norm_1))(spatial_src)
        spatial_out, spatial_weights = self.spatial_attend(
            src=spatial_src, context_mask=spatial_src_mask, key=key_spatial
        )

        ts_residual = src
        if prenorm:
            src = jax.vmap(jax.vmap(self.ts_norm1))(src)

        # Fusion: spatial_weights (B, N, N) @ src.transpose (B, N, T) -> (B, N, T) -> (B, T, N)
        ts_out = jnp.matmul(spatial_weights, src.transpose(0, 2, 1)).transpose(0, 2, 1)
        ts_out = ts_residual + (
            self.ts_dropout1(ts_out, key=key_ts_drop1)
            if key is not None
            else self.ts_dropout1(ts_out, inference=True)
        )
        if not prenorm:
            ts_out = jax.vmap(jax.vmap(self.ts_norm1))(ts_out)

        ts_residual = ts_out
        if prenorm:
            ts_out = jax.vmap(jax.vmap(self.ts_norm2))(ts_out)

        ts_out = jax.vmap(jax.vmap(self.ts_linear1))(ts_out)
        ts_out = jax.vmap(jax.vmap(self.activation))(ts_out)
        ts_out = (
            self.ts_dropout2(ts_out, key=key_ts_drop2)
            if key is not None
            else self.ts_dropout2(ts_out, inference=True)
        )
        ts_out = jax.vmap(jax.vmap(self.ts_linear2))(ts_out)

        ts_out = ts_residual + (
            self.ts_dropout3(ts_out, key=key_ts_drop3)
            if key is not None
            else self.ts_dropout3(ts_out, inference=True)
        )
        if not prenorm:
            ts_out = jax.vmap(jax.vmap(self.ts_norm2))(ts_out)

        return (
            ts_out,
            spatial_weights,
            0.0,
        )  # Return dummy cost since we removed loss computation


class STNDTEncoderStack(eqx.Module):
    layers: list
    norm: Optional[eqx.Module]
    config: Dict[str, Any] = eqx.static_field()

    def __init__(
        self,
        encoder_layer: STNDTEncoder,
        config=None,
        num_layers=None,
        d_model=None,
        trial_length=None,
        norm=None,
        key=None,
    ):
        keys = jr.split(key, num_layers) if key is not None else [None] * num_layers

        self.layers = [
            encoder_layer(
                config=config, d_model=d_model, trial_length=trial_length, key=keys[i]
            )
            for i in range(num_layers)
        ]

        self.config = config
        self.norm = norm

    def split_src(self, src):
        r"""More useful in inherited classes"""
        return src

    def extract_return_src(self, src):
        r"""More useful in inherited classes"""
        return src

    def __call__(
        self,
        src,
        spatial_src,
        mask=None,
        spatial_mask=None,
        return_outputs=False,
        return_weights=False,
        **kwargs,
    ):
        value = src
        src = self.split_src(src)
        layer_outputs = []
        layer_weights = []
        layer_costs = []
        prenorm = self.config.get("PRE_NORM", False)
        for i, mod in enumerate(self.layers):
            if i == 0:
                src, weights, layer_cost = mod(
                    src,
                    spatial_src,
                    src_mask=mask,
                    spatial_src_mask=spatial_mask,
                    prenorm=prenorm,
                    **kwargs,
                )
            else:
                # For subsequent layers, spatial_src is the transposed src: (B, T, N) -> (B, N, T)
                src, weights, layer_cost = mod(
                    src,
                    src.transpose(0, 2, 1),
                    src_mask=mask,
                    spatial_src_mask=spatial_mask,
                    prenorm=prenorm,
                    **kwargs,
                )
            if return_outputs:
                layer_outputs.append(src)  # Already in (B, T, N) format
            layer_weights.append(weights)
            layer_costs.append(layer_cost)
        total_layer_cost = jnp.sum(jnp.array(layer_costs), axis=0)

        if not return_weights:
            layer_weights = None
        if not return_outputs:
            layer_outputs = None
        else:
            layer_outputs = jnp.stack(layer_outputs, axis=-1)

        return_src = self.extract_return_src(src)
        if self.norm is not None:
            return_src = jax.vmap(jax.vmap(self.norm))(return_src)

        return return_src, layer_outputs, layer_weights, total_layer_cost


class STNDT(eqx.Module):
    """Spatiotemporal Neural Data Transformer for JAX/Equinox"""

    # Configuration
    trial_length: int
    num_neurons: int
    num_input: int
    num_spatial_input: int
    scale: float
    spatial_scale: float
    n_views: int
    temperature: float
    constrast_lambda: float

    # Model components
    embedder: eqx.Module
    spatial_embedder: eqx.Module
    src_pos_encoder: PositionalEncoding
    spatial_pos_encoder: PositionalEncoding
    projector: eqx.Module
    spatial_projector: eqx.Module
    rate_dropout: eqx.nn.Dropout
    encoder: STNDTEncoderStack
    src_decoder: eqx.Module

    # Masks (static)
    src_mask: Optional[jnp.ndarray] = eqx.static_field()
    spatial_src_mask: Optional[jnp.ndarray] = eqx.static_field()
    config: Dict[str, Any] = eqx.static_field()

    def __init__(self, config, trial_length, num_neurons, max_spikes, key=None):
        if key is None:
            key = jr.PRNGKey(0)

        self.config = config
        self.trial_length = trial_length
        self.num_neurons = num_neurons

        # TODO buffer
        if config.get("FULL_CONTEXT", False):
            self.src_mask = None
            self.spatial_src_mask = None
        else:
            self.src_mask = {}  # multi-GPU masks
            self.spatial_src_mask = {}  # multi-GPU masks

        self.num_input = self.num_neurons
        self.num_spatial_input = self.trial_length

        assert config.get("EMBED_DIM") in [0, 1], "EMBED_DIM must be 0 or 1 for STNDT"

        # Split keys for different components
        keys = jr.split(key, 10)  # We need several keys

        if config.get("LINEAR_EMBEDDER"):
            self.embedder = eqx.nn.Sequential(
                [eqx.nn.Linear(self.num_neurons, self.num_input, key=keys[0])]
            )
            self.spatial_embedder = eqx.nn.Sequential(
                [eqx.nn.Linear(self.trial_length, self.num_spatial_input, key=keys[1])]
            )
        elif config.get("EMBED_DIM") == 0:
            self.embedder = eqx.nn.Identity()
            self.spatial_embedder = eqx.nn.Identity()
        else:  # config.EMBED_DIM == 1
            # Create embeddings with custom initialization if SPIKE_LOG_INIT is True
            num_embeddings = max_spikes + 2
            embed_dim = config.get("EMBED_DIM")

            if config.get("SPIKE_LOG_INIT", False):
                # Use log-scale initialization for spike semantics
                embed_key1, embed_key2 = jr.split(keys[0])
                embed_weights = self.spike_log_embedding_init(
                    num_embeddings, embed_dim, embed_key1
                )
                spatial_embed_weights = self.spike_log_embedding_init(
                    num_embeddings, embed_dim, embed_key2
                )

                # Create embeddings and replace weights
                embedder_layer = eqx.nn.Embedding(
                    num_embeddings, embed_dim, key=keys[0]
                )
                embedder_layer = eqx.tree_at(
                    lambda m: m.weight, embedder_layer, embed_weights
                )

                spatial_embedder_layer = eqx.nn.Embedding(
                    num_embeddings, embed_dim, key=keys[1]
                )
                spatial_embedder_layer = eqx.tree_at(
                    lambda m: m.weight, spatial_embedder_layer, spatial_embed_weights
                )
            else:
                # Standard uniform initialization
                embedder_layer = eqx.nn.Embedding(
                    num_embeddings, embed_dim, key=keys[0]
                )
                spatial_embedder_layer = eqx.nn.Embedding(
                    num_embeddings, embed_dim, key=keys[1]
                )

            self.embedder = eqx.nn.Sequential(
                [
                    embedder_layer,
                    eqx.nn.Lambda(
                        lambda x: x.reshape(*x.shape[:-2], -1)
                    ),  # Flatten last 2 dims
                ]
            )
            self.spatial_embedder = eqx.nn.Sequential(
                [
                    spatial_embedder_layer,
                    eqx.nn.Lambda(
                        lambda x: x.reshape(*x.shape[:-2], -1)
                    ),  # Flatten last 2 dims
                ]
            )

        self.scale = math.sqrt(self.get_factor_size())
        self.spatial_scale = math.sqrt(self.get_factor_size(spatial=True))
        self.src_pos_encoder = PositionalEncoding(
            config, self.trial_length, self.get_factor_size(), key=keys[2]
        )
        self.spatial_pos_encoder = PositionalEncoding(
            config, self.num_neurons, self.trial_length, key=keys[3]
        )

        if config.get("USE_CONTRAST_PROJECTOR"):
            if config.get("LINEAR_PROJECTOR"):
                self.projector = eqx.nn.Linear(
                    self.get_factor_size(), self.get_factor_size(), key=keys[4]
                )
                self.spatial_projector = eqx.nn.Linear(
                    self.get_factor_size(spatial=True),
                    self.get_factor_size(spatial=True),
                    key=keys[5],
                )
            else:
                proj_keys = jr.split(keys[4], 4)
                self.projector = eqx.nn.Sequential(
                    [
                        eqx.nn.Linear(self.get_factor_size(), 1024, key=proj_keys[0]),
                        eqx.nn.Lambda(eqx.nn.PReLU(init_alpha=0.01)),
                        eqx.nn.Linear(1024, self.get_factor_size(), key=proj_keys[1]),
                    ]
                )
                self.spatial_projector = eqx.nn.Sequential(
                    [
                        eqx.nn.Linear(
                            self.get_factor_size(spatial=True), 1024, key=proj_keys[2]
                        ),
                        eqx.nn.Lambda(eqx.nn.PReLU(init_alpha=0.01)),
                        eqx.nn.Linear(
                            1024, self.get_factor_size(spatial=True), key=proj_keys[3]
                        ),
                    ]
                )
        else:
            self.projector = eqx.nn.Identity()
            self.spatial_projector = eqx.nn.Identity()

        # Rate dropout
        self.n_views = 2
        self.temperature = self.config.get("TEMPERATURE")
        self.constrast_lambda = self.config.get("LAMBDA")

        # Initialize encoder stack
        encoder_layer = self.get_encoder_layer()
        if self.config.get("SCALE_NORM", False):
            norm = ScaleNorm(self.get_factor_size() ** 0.5)
        else:
            norm = eqx.nn.LayerNorm(self.get_factor_size())
        self.encoder = STNDTEncoderStack(
            encoder_layer=encoder_layer,
            config=config,
            num_layers=config.get("NUM_LAYERS", 6),
            d_model=self.get_factor_size(),
            trial_length=self.trial_length,
            norm=norm,
            key=keys[6],
        )

        self.rate_dropout = eqx.nn.Dropout(config.get("DROPOUT_RATES", 0.0))

        ##LOSS
        src_decoder_layers_keys = jr.split(keys[7], 2)
        if config.get("LOSS").get("TYPE") == "poisson":

            if config.get("DECODER", {}).get("LAYERS", 1) == 1:
                # Single layer decoder - custom initialization
                weight_shape = (self.get_factor_size(), self.num_neurons)
                bias_shape = (self.num_neurons,)
                weight, bias = self.spike_aware_decoder_init(
                    weight_shape, bias_shape, src_decoder_layers_keys[0]
                )

                # Create decoder with custom weights
                decoder = eqx.nn.Linear(
                    self.get_factor_size(),
                    self.num_neurons,
                    key=src_decoder_layers_keys[0],
                )
                decoder = eqx.tree_at(
                    lambda m: (m.weight, m.bias), decoder, (weight, bias)
                )
                self.src_decoder = decoder
            else:
                # Multi-layer decoder
                src_decoder_layers = []

                # Apply custom initialization to FIRST layer (matching PyTorch)
                first_layer = eqx.nn.Linear(
                    self.get_factor_size(), 16, key=src_decoder_layers_keys[0]
                )
                weight_shape = (self.get_factor_size(), 16)
                bias_shape = (16,)
                weight, bias = self.spike_aware_decoder_init(
                    weight_shape, bias_shape, src_decoder_layers_keys[0]
                )
                first_layer = eqx.tree_at(
                    lambda m: (m.weight, m.bias), first_layer, (weight, bias)
                )

                src_decoder_layers.append(first_layer)
                src_decoder_layers.append(eqx.nn.Lambda(jax.nn.relu))
                src_decoder_layers.append(
                    eqx.nn.Linear(16, self.num_neurons, key=src_decoder_layers_keys[1])
                )

                self.src_decoder = eqx.nn.Sequential(src_decoder_layers)

            # Add activation only if not using log-rates
            if not config.get('LOGRATE', True):
                self.src_decoder = eqx.nn.Sequential([self.src_decoder, eqx.nn.Lambda(jax.nn.relu)])
        elif config.get('LOSS').get('TYPE') == 'cel':
            self.src_decoder = eqx.nn.Sequential(eqx.nn.Linear(self.get_factor_size(), config.get('MAX_SPIKE_COUNT') * self.num_neurons, key=src_decoder_layers_keys[0])) #log-likelihood

        else:
            raise ValueError(f"Unsupported loss type: {config.get('LOSS').get('TYPE')}")

    def spike_aware_decoder_init(self, weight_shape, bias_shape, key, initrange=0.1):
        weight = jax.random.uniform(
            key, weight_shape, minval=-initrange, maxval=initrange
        )
        bias = jnp.zeros(bias_shape)
        return weight, bias

    def spike_log_embedding_init(self, num_embeddings, embed_dim, key, initrange=0.1):
        """Initialize embeddings with log scale for spike semantics (PyTorch STNDT style)"""
        # Use a log scale, since we expect spike semantics to follow compressive distribution
        log_scale = jnp.log(
            jnp.arange(1, num_embeddings + 1, dtype=jnp.float32)
        )  # [0, ln(2), ln(3), ...]
        log_scale = (log_scale - log_scale.mean()) / (log_scale[-1] - log_scale[0])
        log_scale = log_scale * initrange

        # Add some noise
        noise = jax.random.uniform(
            key,
            (num_embeddings, embed_dim),
            minval=-initrange / 10,
            maxval=initrange / 10,
        )

        # Expand log_scale to match embedding shape and add noise
        weights = log_scale[:, None] + noise
        return weights

    def info_nce_loss(self, features):
        batch_size = features.shape[0] // 2
        labels = jnp.concatenate(
            [jnp.arange(batch_size) for i in range(self.n_views)], axis=0
        )
        labels = (jnp.expand_dims(labels, 0) == jnp.expand_dims(labels, 1)).astype(
            jnp.float32
        )

        features = jax_normalize(features, axis=1)
        similarity_matrix = jnp.matmul(features, features.T)

        mask = jnp.eye(labels.shape[0], dtype=jnp.bool)
        labels = labels[~mask].reshape(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].reshape(
            similarity_matrix.shape[0], -1
        )

        positives = similarity_matrix[labels == 1].reshape(labels.shape[0], -1)
        negatives = similarity_matrix[labels == 0].reshape(
            similarity_matrix.shape[0], -1
        )

        logits = jnp.concatenate([positives, negatives], axis=1)
        labels = jnp.zeros(logits.shape[0], dtype=jnp.int64)

        logits = logits / self.temperature
        return logits, labels

    def get_factor_size(self, spatial=False):
        if spatial:
            return self.num_spatial_input
        else:
            return self.num_input

    def get_hidden_size(self):
        return self.num_input

    def get_encoder_layer(self):
        return STNDTEncoder

    def get_encoder(self):
        return STNDTEncoderStack

    def _generate_context_mask(
        self, size: int, expose_ic: bool = True, spatial=False
    ) -> jnp.ndarray:
        """Generate context mask for attention"""
        if self.config.get("FULL_CONTEXT", False):
            return None

        context_forward = self.config.get("CONTEXT_FORWARD", 0)
        context_backward = self.config.get("CONTEXT_BACKWARD", -1)
        if context_forward < 0:
            context_forward = size

        # Create forward-looking mask (boolean)
        mask = jnp.triu(jnp.ones((size, size), dtype=bool), k=-context_forward).T

        # Apply backward context if specified
        if context_backward > 0:
            back_mask = jnp.triu(
                jnp.ones((size, size), dtype=bool), k=-context_backward
            )
            mask = mask & back_mask

            # Expose initial segment for initial conditions
            if expose_ic and self.config.get("CONTEXT_WRAP_INITIAL", False):
                initial_mask = jnp.triu(
                    jnp.ones((context_backward, context_backward), dtype=bool)
                )
                mask = mask.at[:context_backward, :context_backward].set(
                    mask[:context_backward, :context_backward] | initial_mask
                )

        # Convert binary mask to attention mask (False -> -inf, True -> 0)
        return jnp.where(mask, 0.0, -jnp.inf)

    # def __call__(self, src, key=None, return_weights=False, return_outputs=False):
    #     # src shape: (B, T, N)
    #     if key is None:
    #         key = jr.PRNGKey(1)

    #     key, subkey1, subkey2, subkey3, subkey4 = jr.split(key, 5)

    #     # For spatial attention: (B, N, T)
    #     spatial_src = src.transpose(0, 2, 1)
    #     # print(f"Spatial src shape after transpose: {spatial_src.shape}")
    #     if self.config.get('LINEAR_EMBEDDER'):
    #         print(f"Spatial embedder input size: {self.trial_length}, output size: {self.num_spatial_input}")
    #         spatial_src = jax.vmap(jax.vmap(self.spatial_embedder))(spatial_src)
    #     elif self.config.get('EMBED_DIM') == 0:
    #         spatial_src = jax.vmap(self.spatial_embedder)(spatial_src)
    #     else: # config.EMBED_DIM == 1
    #         spatial_src = jax.vmap(jax.vmap(jax.vmap(self.spatial_embedder)))(spatial_src)
    #         spatial_src = spatial_src.squeeze(-1)

    #     spatial_src = spatial_src * self.spatial_scale

    #     # Apply spatial embedder correctly based on type
    #     # Split key for each batch
    #     if subkey1 is not None:
    #         spatial_pos_keys = jr.split(subkey1, spatial_src.shape[0])
    #     else:
    #         spatial_pos_keys = [None] * spatial_src.shape[0]
    #     spatial_src = jax.vmap(self.spatial_pos_encoder)(spatial_src, spatial_pos_keys)

    #     # Keep temporal in (B, T, N) format
    #     if self.config.get('LINEAR_EMBEDDER'):
    #         src = jax.vmap(jax.vmap(self.embedder))(src)
    #     elif self.config.get('EMBED_DIM') == 0:
    #         src = jax.vmap(self.embedder)(src)
    #     else:  # config.EMBED_DIM == 1
    #         src = jax.vmap(jax.vmap(jax.vmap(self.embedder)))(src)
    #         src = src.squeeze(-1)

    #     src = src * self.scale

    #     # Split key for each batch
    #     if subkey2 is not None:
    #         pos_keys = jr.split(subkey2, src.shape[0])
    #     else:
    #         pos_keys = [None] * src.shape[0]
    #     src = jax.vmap(self.src_pos_encoder)(src, pos_keys)
    #     src_mask = self._generate_context_mask(src.shape[1])  # T dimension
    #     spatial_src_mask = None

    #     (
    #         encoder_output,
    #         layer_outputs,
    #         layer_weights,
    #         _
    #     ) = self.encoder(src, spatial_src,
    #                      mask=src_mask,
    #                      spatial_mask=spatial_src_mask,
    #                      return_outputs=return_outputs,
    #                      return_weights=return_weights,)
    #     encoder_output = self.rate_dropout(encoder_output, key=subkey4) if key is not None else self.rate_dropout(encoder_output, inference=True)
    #     # decoder_output = jax.vmap(jax.vmap(self.decoder))(encoder_output)  # (B, T, D) -> (B, T, N)
    #     # decoder_rates = decoder_output  # Already in (B, T, N) format

    #     # if return_weights and return_outputs:
    #     #     return decoder_rates, layer_weights, layer_outputs
    #     # elif return_weights:
    #     #     return decoder_rates, layer_weights
    #     # elif return_outputs:
    #     #     return decoder_rates, layer_outputs
    #     # else:
    #     #     return decoder_rates
    #     decoder_output = jax.vmap(jax.vmap(self.src_decoder))(encoder_output)

    #     #TODO: Add continuous rate prediction decoder

    #     if return_weights and return_outputs:
    #         return decoder_output, layer_weights, layer_outputs
    #     elif return_weights:
    #         return decoder_output, layer_weights
    #     elif return_outputs:
    #         return decoder_output, layer_outputs
    #     else:
    #         return decoder_output

    def forward(
        self,
        src,
        mask_labels,
        contrast_src1=None,
        contrast_src2=None,
        val_phase=False,
        key=None,
        **kwargs,
    ):
        # src = src.astype(jnp.float32)

        # if contrast_src1 is not None and contrast_src2 is not None:
        #     contrast_src1 = contrast_src1.astype(jnp.float32)
        #     contrast_src2 = contrast_src2.astype(jnp.float32)

        spatial_src = src.transpose(0, 2, 1)

        # spatial_src = jax.vmap(jax.vmap(self.spatial_embedder))(spatial_src) * self.spatial_scale
        # spatial_src = self.spatial_pos_encoder(spatial_src)
        if self.config.get("LINEAR_EMBEDDER"):
            spatial_src = (
                jax.vmap(jax.vmap(self.spatial_embedder))(spatial_src)
                * self.spatial_scale
            )
        elif self.config.get("EMBED_DIM") == 0:
            spatial_src = (
                jax.vmap(self.spatial_embedder)(spatial_src) * self.spatial_scale
            )
        else:  # config.EMBED_DIM == 1
            spatial_src = jax.vmap(jax.vmap(jax.vmap(self.spatial_embedder)))(
                spatial_src
            )
            spatial_src = spatial_src.astype(jnp.float32) * self.spatial_scale
            spatial_src = spatial_src.squeeze(-1)
        spatial_src = jax.vmap(self.spatial_pos_encoder)(spatial_src)

        # src = self.embedder(src) * self.scale
        # src = self.src_pos_encoder(src)
        if self.config.get("LINEAR_EMBEDDER"):
            src = jax.vmap(jax.vmap(self.embedder))(src) * self.scale
        elif self.config.get("EMBED_DIM") == 0:
            src = jax.vmap(self.embedder)(src) * self.scale
        else:  # config.EMBED_DIM == 1
            src = jax.vmap(jax.vmap(jax.vmap(self.embedder)))(src) * self.scale
            src = src.squeeze(-1)
        src = jax.vmap(self.src_pos_encoder)(src)

        src_mask = self._generate_context_mask(src.shape[1])
        spatial_src_mask = None
        (encoder_output, layer_outputs, layer_weights, _) = self.encoder(
            src=src,
            spatial_src=spatial_src,
            mask=src_mask,
            spatial_mask=spatial_src_mask,
            **kwargs,
        )
        if key is None:
            key = jr.PRNGKey(1)

        key, subkey1, subkey2, subkey3, subkey4 = jr.split(key, 5)

        encoder_output = (
            self.rate_dropout(encoder_output, key=subkey1)
            if not val_phase
            else self.rate_dropout(encoder_output, inference=True)
        )
        decoder_output = jax.vmap(jax.vmap(self.src_decoder))(encoder_output)

        if contrast_src1 is not None and contrast_src2 is not None:
            spatial_contrast1 = contrast_src1.transpose(2, 0, 1)  # BxTxN to NxBxT
            spatial_contrast_embedded1 = (
                self.spatial_embedder(spatial_contrast1) * self.spatial_scale
            )
            spatial_contrast1 = self.spatial_pos_encoder(spatial_contrast_embedded1)

            contrast_src1 = contrast_src1  # keep BxTxN format
            contrast_src_embedded1 = self.embedder(contrast_src1) * self.scale
            contrast_src1 = self.src_pos_encoder(contrast_src_embedded1)
            contrast_mask1 = self._generate_context_mask(contrast_src1.shape[1])
            spatial_contrast_mask1 = None

            (encoder_output_contrast1, layer_outputs_contrast1, _, _) = self.encoder(
                contrast_src1,
                spatial_contrast1,
                contrast_mask1,
                spatial_contrast_mask1,
                **kwargs,
            )
            encoder_output_contrast1 = (
                self.rate_dropout(encoder_output_contrast1, key=subkey2)
                if not val_phase
                else self.rate_dropout(encoder_output_contrast1, inference=True)
            )

            spatial_contrast2 = contrast_src2.transpose(2, 0, 1)  # BxTxN to NxBxT
            spatial_contrast_embedded2 = (
                self.spatial_embedder(spatial_contrast2) * self.spatial_scale
            )
            spatial_contrast2 = self.spatial_pos_encoder(spatial_contrast_embedded2)

            contrast_src2 = contrast_src2  # keep BxTxN format
            contrast_src_embedded2 = self.embedder(contrast_src2) * self.scale
            contrast_src2 = self.src_pos_encoder(contrast_src_embedded2)
            contrast_mask2 = self._generate_context_mask(contrast_src2.shape[1])
            spatial_contrast_mask2 = None
            (encoder_output_contrast2, layer_outputs_contrast2, _, _) = self.encoder(
                contrast_src2,
                spatial_contrast2,
                contrast_mask2,
                spatial_contrast_mask2,
                **kwargs,
            )
            encoder_output_contrast2 = (
                self.rate_dropout(encoder_output_contrast2, key=subkey3)
                if not val_phase
                else self.rate_dropout(encoder_output_contrast2, inference=True)
            )

            decoder_output_contrast1 = self.src_decoder(encoder_output_contrast1)
            decoder_output_contrast2 = self.src_decoder(encoder_output_contrast2)

            if self.config.get("CONTRAST_LAYER") == "embedder":
                out1 = contrast_src_embedded1
                out2 = contrast_src_embedded2
            elif self.config.get("CONTRAST_LAYER") == "encoder":
                out1 = decoder_output_contrast1
                out2 = decoder_output_contrast2
            else:
                out1 = layer_outputs_contrast1[self.config.get("CONTRAST_LAYER")]
                out2 = layer_outputs_contrast2[self.config.get("CONTRAST_LAYER")]

            # Apply projector and flatten (no permute needed - keep BxTxN format)
            out1 = self.projector(out1)  # Still (B, T, N)
            out1 = out1.reshape(out1.shape[0], -1)  # (B, T, N) → (B, T*N)
            out2 = self.projector(out2)  # Still (B, T, N)
            out2 = out2.reshape(out2.shape[0], -1)  # (B, T, N) → (B, T*N)

            # Combine for InfoNCE loss
            features = jnp.concatenate([out1, out2], axis=0)  # (2*B, T*N)
            logits, labels = self.info_nce_loss(features)

            if not val_phase:
                contrast_loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits, labels
                )
                contrast_loss = jnp.mean(contrast_loss) * self.constrast_lambda
            else:
                contrast_loss = (
                    optax.softmax_cross_entropy_with_integer_labels(logits, labels)
                    * self.constrast_lambda
                )

            if self.config.get("CONTRAST_LAYER") == "embedder":
                out1 = spatial_contrast_embedded1
                out2 = spatial_contrast_embedded2
                out1 = self.projector(out1)
                out1 = out1.reshape(out1.shape[0], -1)
                out2 = self.projector(out2)
                out2 = out2.reshape(out2.shape[0], -1)

                features = jnp.concatenate([out1, out2], axis=0)  # (2*B, T*N)
                logits, labels = self.info_nce_loss(features)
                if not val_phase:
                    contrast_loss = (
                        contrast_loss
                        + jnp.mean(
                            optax.softmax_cross_entropy_with_integer_labels(
                                logits, labels
                            )
                        )
                        * self.constrast_lambda
                    )

                else:
                    contrast_loss = (
                        contrast_loss
                        + optax.softmax_cross_entropy_with_integer_labels(
                            logits, labels
                        )
                        * self.constrast_lambda
                    )
        else:
            contrast_loss = 0.0

        if self.config.get("LOSS").get("TYPE") == "poisson":
            decoder_rates = decoder_output
            # Match PyTorch exactly: compute loss on ALL positions first (including invalid -100s)
            # This will produce invalid values for -100 targets, but we filter them out next
            decoder_loss = poisson_nll_loss(
                decoder_rates, mask_labels, log_input=self.config.get("LOGRATE", True)
            )

        # Extract losses only for masked positions (like PyTorch boolean indexing)
        valid_mask = mask_labels != UNMASKED_LABEL
        # Filter out invalid losses (matching PyTorch approach)
        valid_losses = jnp.where(valid_mask, decoder_loss, 0.0)

        if not val_phase:
            # Take mean of valid losses only (matching PyTorch .mean())
            num_valid = jnp.sum(valid_mask)
            masked_decoder_loss = jnp.sum(valid_losses) / num_valid
            loss = masked_decoder_loss + contrast_loss
        else:
            # For validation, return filtered losses (matching PyTorch shape)
            masked_decoder_loss = (
                valid_losses  # Keep original shape with zeros for invalid
            )
            loss = masked_decoder_loss

        if not val_phase:
            return (
                loss,  # Must be scalar for gradient computation
                masked_decoder_loss,
                contrast_loss,
                decoder_rates,
            )
        else:
            return (
                loss,
                masked_decoder_loss,
                contrast_loss,
                decoder_rates,
                layer_weights,
                layer_outputs,
            )


class ScaleNorm(eqx.Module):
    """ScaleNorm for JAX/Equinox"""

    scale: jnp.ndarray
    eps: float

    def __init__(self, scale, eps=1e-5):
        self.scale = jnp.array(scale)
        self.eps = eps

    def __call__(self, x):
        norm = self.scale / jnp.linalg.norm(x, axis=-1, keepdims=True).clip(
            min=self.eps
        )
        return x * norm


def jax_normalize(x, dim=-1, eps=1e-12):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=dim, keepdims=True), eps)

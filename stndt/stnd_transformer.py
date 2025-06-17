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

class PositionalEncoding(eqx.Module):
    pe: jnp.ndarray
    dropout: eqx.nn.Dropout
    learnable: bool
    pos_embedding: Optional[eqx.nn.Embedding]
    
    def __init__(self, trial_length, d_model, dropout=0.1, learnable=False, offset=False, key=None):
        if key is None:
            key = jr.PRNGKey(0)
        
        self.dropout = eqx.nn.Dropout(dropout)
        self.learnable = learnable
        
        position = jnp.arange(0, trial_length, dtype=jnp.float32).reshape(-1, 1)
        if offset:
            position = position + 1
            
        if learnable:
            self.pe = position.astype(jnp.int32).squeeze()
            self.pos_embedding = eqx.nn.Embedding(trial_length, d_model, key=key)
        else:
            pe = jnp.zeros((trial_length, d_model))
            div_term = jnp.exp(jnp.arange(0, d_model, 2).astype(jnp.float32) * 
                              (-math.log(10000.0) / d_model))
            pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
            if d_model % 2 == 0:
                pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
            else:
                pe = pe.at[:, 1::2].set(jnp.cos(position * div_term[:-1]))
            # Transpose to match PyTorch format (seq_len, 1, d_model)
            self.pe = pe[jnp.newaxis, :, :].transpose(1, 0, 2)
            self.pos_embedding = None
    
    def __call__(self, x, key=None):
        if self.learnable:
            seq_len = x.shape[0]
            pos_indices = self.pe[:seq_len]
            x = x + jax.vmap(self.pos_embedding)(pos_indices)
        else:
            x = x + self.pe[:x.shape[0], :]
        
        if key is not None:
            x = self.dropout(x, key=key)
        else:
            x = self.dropout(x, inference=True)
        return x


class TemporalHead(eqx.Module):
    k : eqx.nn.Linear
    q : eqx.nn.Linear
    v : eqx.nn.Linear
    mask : jnp.ndarray
    dropout : eqx.nn.Dropout


    def __init__(self, n_embed, head_size, max_length, dropout=0.1, key=None):
        if key is None:
            key = jr.PRNGKey(0)
    
        key_k, key_q, key_v = jr.split(key, 3)

        self.k = eqx.nn.Linear(in_features=n_embed, out_features=head_size, use_bias=False, key=key_k)
        self.q = eqx.nn.Linear(in_features=n_embed, out_features=head_size, use_bias=False, key=key_q)
        self.v = eqx.nn.Linear(in_features=n_embed, out_features=head_size, use_bias=False, key=key_v)
        self.mask = jnp.tril(jnp.ones((max_length, max_length), dtype=jnp.float32))
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x, key=None):
        B, T, C = x.shape
        k = jax.vmap(jax.vmap(self.k))(x)  # (B, T, head_size)
        q = jax.vmap(jax.vmap(self.q))(x)  
        v = jax.vmap(jax.vmap(self.v))(x)  
        k_transpose = jnp.transpose(k, (0, 2, 1))
        scores = jnp.matmul(q, k_transpose) / jnp.sqrt(k.shape[-1])
        scores = jnp.where(self.mask[:T, :T] == 0, -jnp.inf, scores)
        scores = jax.nn.softmax(scores, axis=-1)
        if key is not None:
            scores = self.dropout(scores, key=key)
        else:
            scores = self.dropout(scores, inference=True)
        out = jnp.matmul(scores, v)
        return out


class SpatialHead(eqx.Module):
    k: eqx.nn.Linear
    q: eqx.nn.Linear
    dropout: eqx.nn.Dropout


    def __init__(self, n_embed, head_size, dropout, key=None):
        if key is None:
            key = jr.PRNGKey(1)

        key_k, key_q = jr.split(key, 2)

        self.k = eqx.nn.Linear(in_features=n_embed, out_features=head_size, use_bias=False, key=key_k)
        self.q = eqx.nn.Linear(in_features=n_embed, out_features=head_size, use_bias=False, key=key_q)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x, key=None):
        # x has shape (B, N, T) where T is trial_length
        # We want to produce attention weights of shape (B, N, N)
        k = jax.vmap(jax.vmap(self.k))(x)  # (B, N, head_size)
        q = jax.vmap(jax.vmap(self.q))(x)  # (B, N, head_size)
        
        # Compute attention scores over spatial dimension
        scores = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) / jnp.sqrt(k.shape[-1])  # (B, N, N)
        scores = jax.nn.softmax(scores, axis=-1)
        
        if key is not None:
            scores = self.dropout(scores, key=key)
        else:
            scores = self.dropout(scores, inference=True)
        return scores


#Multihead attention
class MultiheadTemporalAttention(eqx.Module):
    heads: list
    linear: eqx.nn.Linear
    dropout: eqx.nn.Dropout


    def __init__(self, n_embed, num_heads, max_length=100, dropout=0.1, key=None):
        if key is None:
            key = jr.PRNGKey(2)

        head_size = n_embed // num_heads
        keys = jr.split(key, num_heads + 1)
        self.heads = [TemporalHead(n_embed=n_embed, head_size=head_size, max_length=max_length, dropout=dropout, key=keys[i]) for i in range(num_heads)]
        #ouput projection, 
        self.linear = eqx.nn.Linear(in_features=(num_heads * head_size), out_features=n_embed, use_bias=False, key=keys[-1])
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x, key=None):
        head_out = [head(x, key=key) for head in self.heads]
        out = jnp.concatenate(head_out, axis=-1)
        #apply output projection
        out = jax.vmap(jax.vmap(self.linear))(out)
        if key is not None:
            out = self.dropout(out, key=key)
        else:
            out = self.dropout(out, inference=True)
        return out


class MultiheadSpatialAttention(eqx.Module):
    heads: list

    def __init__(self, trial_length, num_heads, dropout = 0.1, key = None):
        if key is None:
            key= jr.PRNGKey(3)

        keys = jr.split(key, num_heads)
        head_size = trial_length // num_heads
        self.heads = [SpatialHead(n_embed=trial_length, head_size=head_size, dropout=dropout, key=keys[i]) for i in range(num_heads)]

    def __call__(self, x, key=None):
        head_out = [head(x, key=key) for head in self.heads]
        out = jnp.stack(head_out, axis=0)   #(num_heads, B, spatial_dim, spatial_dim)
        out = jnp.mean(out, axis=0)
        return out


class FeedForward(eqx.Module):
    linear1: eqx.nn.Linear
    activation: eqx.nn.Lambda
    linear2: eqx.nn.Linear
    dropout: eqx.nn.Dropout


    def __init__(self, n_embed, dropout=0.1, expansion_factor = 4, key = None):
        if key is None:
            key = jr.PRNGKey(4)

        key1, key2 = jr.split(key)
        self.linear1 = eqx.nn.Linear(in_features=n_embed, out_features=expansion_factor * n_embed, use_bias=True, key=key1)
        self.activation = eqx.nn.Lambda(jax.nn.relu)
        self.linear2 = eqx.nn.Linear(in_features=expansion_factor * n_embed, out_features=n_embed, use_bias=True, key=key2)
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


class SpikeProjection(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, num_neurons, n_embed, key=None):
        if key is None:
            key = jr.PRNGKey(1)
        self.linear = eqx.nn.Linear(in_features=num_neurons, out_features=n_embed, use_bias=False, key=key)

    def __call__(self, spike_counts):
        return jax.vmap(self.linear)(spike_counts)  # (B, T, N) -> (B, T, n_embed)


#Fusion Mechanism
class STNDTEncoder(eqx.Module):
    temporal_attention: MultiheadTemporalAttention
    spatial_attention: MultiheadSpatialAttention
    feed_forward_1: FeedForward
    feed_forward_2: FeedForward
    t_ln1: eqx.nn.LayerNorm
    t_ln2: eqx.nn.LayerNorm
    s_ln1: eqx.nn.LayerNorm
    fusion_ln: eqx.nn.LayerNorm

    def __init__(self, n_embed, num_heads, max_length=100, dropout=0.1, key=None):
        if key is None:
            key = jr.PRNGKey(5)

        keys = jr.split(key, 3)
        self.temporal_attention = MultiheadTemporalAttention(n_embed=n_embed, num_heads=num_heads, max_length=max_length, dropout=dropout, key=keys[0])
        self.spatial_attention = MultiheadSpatialAttention(trial_length=max_length, num_heads=num_heads, dropout=dropout, key=keys[1])
        key_ff1, key_ff2 = jr.split(keys[2], 2)
        self.feed_forward_1 = FeedForward(n_embed=n_embed, dropout=dropout, key=key_ff1)
        self.feed_forward_2 = FeedForward(n_embed=n_embed, dropout=dropout, key=key_ff2)
        self.t_ln1 = eqx.nn.LayerNorm(n_embed, eps=1e-6)
        self.t_ln2 = eqx.nn.LayerNorm(n_embed, eps=1e-6)
        self.s_ln1 = eqx.nn.LayerNorm(max_length, eps=1e-6)  # Normalize along T dimension for spatial
        self.fusion_ln = eqx.nn.LayerNorm(n_embed, eps=1e-6)

    def __call__(self, x_T, x_S, key = None):
        #Temporal attention
        if key is None:
            key = jr.PRNGKey(6)
        key_temporal, key_spatial, key_ff1, key_ff2 = jr.split(key, 4)
        ln_x_T = jax.vmap(jax.vmap(self.t_ln1))(x_T)
        temporal_attn_out = self.temporal_attention(ln_x_T, key=key_temporal)
        #Residual connection
        x = x_T + temporal_attn_out

        #Feed forward for temporal features
        ln_x = jax.vmap(jax.vmap(self.t_ln2))(x)
        ff_out = self.feed_forward_1(ln_x, key=key_ff1)
        temporal_out = x + ff_out     #Z_T

        #Spatial attention
        # x_S has shape (B, N, T), normalize along T dimension
        ln_x_S = jax.vmap(jax.vmap(self.s_ln1))(x_S)
        spatial_weights = self.spatial_attention(ln_x_S, key=key_spatial)

        t_permuted = jnp.transpose(temporal_out, (0, 2, 1))  # (B, N, T)

        #fusion mechanism
        fused = jnp.matmul(spatial_weights, t_permuted)  # (B, N, T)
        fused_permuted = jnp.transpose(fused, (0, 2, 1))  # (B, T, N)

        x = temporal_out + fused_permuted  # (B, T, N)

        ln_x = jax.vmap(jax.vmap(self.fusion_ln))(x)
        ff_out = self.feed_forward_2(ln_x, key=key_ff2)
        out = x + ff_out
        return out, spatial_weights


class ScaleNorm(eqx.Module):
    """ScaleNorm for JAX/Equinox"""
    scale: jnp.ndarray
    eps: float
    
    def __init__(self, scale, eps=1e-5):
        self.scale = jnp.array(scale)
        self.eps = eps
    
    def __call__(self, x):
        norm = self.scale / jnp.linalg.norm(x, axis=-1, keepdims=True).clip(min=self.eps)
        return x * norm


class STNDT(eqx.Module):
    """Spatiotemporal Neural Data Transformer for JAX/Equinox"""
    
    # Model components
    embedder: eqx.Module
    spatial_embedder: eqx.Module
    src_pos_encoder: PositionalEncoding
    spatial_pos_encoder: PositionalEncoding
    encoder_layers: list
    norm: Optional[eqx.Module]
    projector: eqx.Module
    spatial_projector: eqx.Module
    src_decoder: eqx.nn.Sequential
    rate_dropout: eqx.nn.Dropout
    
    # Model parameters
    num_neurons: int
    trial_length: int
    scale: float
    spatial_scale: float
    n_views: int = 2
    temperature: float
    contrast_lambda: float
    
    # Configuration
    config: Dict[str, Any]
    
    def __init__(
        self,
        config: Dict[str, Any],
        trial_length: int,
        num_neurons: int,
        max_spikes: int,
        key=None
    ):
        if key is None:
            key = jr.PRNGKey(0)
            
        self.config = config
        self.trial_length = trial_length
        self.num_neurons = num_neurons
        self.temperature = config.get('TEMPERATURE', 0.5)
        self.contrast_lambda = config.get('LAMBDA', 1.0)
        
        # Initialize embedders
        embed_dim = config.get('EMBED_DIM', 0)
        if config.get('LINEAR_EMBEDDER', False):
            key1, key2, key = jr.split(key, 3)
            self.embedder = eqx.nn.Linear(num_neurons, num_neurons, key=key1)
            self.spatial_embedder = eqx.nn.Linear(trial_length, trial_length, key=key2)
        elif embed_dim == 0:
            self.embedder = eqx.nn.Identity()
            self.spatial_embedder = eqx.nn.Identity()
        else:  # embed_dim == 1
            key1, key2, key = jr.split(key, 3)
            self.embedder = eqx.nn.Sequential([
                eqx.nn.Embedding(max_spikes + 2, embed_dim, key=key1),
                eqx.nn.Lambda(lambda x: x.reshape(x.shape[0], -1))  # Flatten
            ])
            self.spatial_embedder = eqx.nn.Sequential([
                eqx.nn.Embedding(max_spikes + 2, embed_dim, key=key2),
                eqx.nn.Lambda(lambda x: x.reshape(x.shape[0], -1))  # Flatten
            ])
        
        self.scale = math.sqrt(num_neurons)
        self.spatial_scale = math.sqrt(trial_length)
        
        # Positional encoders
        key1, key2, key = jr.split(key, 3)
        self.src_pos_encoder = PositionalEncoding(
            trial_length, num_neurons, 
            dropout=config.get('DROPOUT_EMBEDDING', 0.1),
            learnable=config.get('LEARNABLE_POSITION', False),
            offset=config.get('POSITION', {}).get('OFFSET', False),
            key=key1
        )
        self.spatial_pos_encoder = PositionalEncoding(
            num_neurons, trial_length,
            dropout=config.get('DROPOUT_EMBEDDING', 0.1),
            learnable=config.get('LEARNABLE_POSITION', False),
            offset=config.get('POSITION', {}).get('OFFSET', False),
            key=key2
        )
        
        # Initialize transformer encoder layers
        num_layers = config.get('NUM_LAYERS', 6)
        num_heads = config.get('NUM_HEADS', 8)
        dropout = config.get('DROPOUT', 0.1)
        keys = jr.split(key, num_layers + 1)
        
        self.encoder_layers = [
            STNDTEncoder(
                n_embed=num_neurons,
                num_heads=num_heads,
                max_length=trial_length,
                dropout=dropout,
                key=keys[i]
            )
            for i in range(num_layers)
        ]
        
        # Layer norm
        if config.get('SCALE_NORM', False):
            self.norm = ScaleNorm(num_neurons ** 0.5)
        else:
            self.norm = eqx.nn.LayerNorm(num_neurons)
        
        # Projectors for contrastive learning
        key1, key2, key = jr.split(keys[-1], 3)
        if config.get('USE_CONTRAST_PROJECTOR', False):
            if config.get('LINEAR_PROJECTOR', False):
                self.projector = eqx.nn.Linear(num_neurons, num_neurons, key=key1)
                self.spatial_projector = eqx.nn.Linear(trial_length, trial_length, key=key2)
            else:
                self.projector = eqx.nn.Sequential([
                    eqx.nn.Linear(num_neurons, 1024, key=key1),
                    eqx.nn.Lambda(jax.nn.relu),
                    eqx.nn.Linear(1024, num_neurons, key=jr.split(key1)[0])
                ])
                self.spatial_projector = eqx.nn.Sequential([
                    eqx.nn.Linear(trial_length, 1024, key=key2),
                    eqx.nn.Lambda(jax.nn.relu),
                    eqx.nn.Linear(1024, trial_length, key=jr.split(key2)[0])
                ])
        else:
            self.projector = eqx.nn.Identity()
            self.spatial_projector = eqx.nn.Identity()
        
        # Rate dropout
        self.rate_dropout = eqx.nn.Dropout(config.get('DROPOUT_RATES', 0.0))
        
        # Decoder
        key1, key2, key = jr.split(key, 3)
        if config.get('LOSS', {}).get('TYPE') == "poisson":
            decoder_layers = []
            if config.get('DECODER', {}).get('LAYERS', 1) == 1:
                decoder_layers.append(eqx.nn.Linear(num_neurons, num_neurons, key=key1))
            else:
                decoder_layers.extend([
                    eqx.nn.Linear(num_neurons, 16, key=key1),
                    eqx.nn.Lambda(jax.nn.relu),
                    eqx.nn.Linear(16, num_neurons, key=key2)
                ])
            if not config.get('LOGRATE', False):
                decoder_layers.append(eqx.nn.Lambda(jax.nn.relu))
            self.src_decoder = eqx.nn.Sequential(decoder_layers)
    
    def _generate_context_mask(self, size: int, context_forward: int = -1, 
                             context_backward: int = 0, expose_ic: bool = True) -> jnp.ndarray:
        """Generate context mask for attention"""
        if self.config.get('FULL_CONTEXT', False):
            return None
            
        if context_forward < 0:
            context_forward = size
            
        # Create forward-looking mask
        mask = jnp.triu(jnp.ones((size, size)), k=-context_forward).T
        
        # Apply backward context if specified
        if context_backward > 0:
            back_mask = jnp.triu(jnp.ones((size, size)), k=-context_backward)
            mask = mask & back_mask
            
            # Expose initial segment for initial conditions
            if expose_ic and self.config.get('CONTEXT_WRAP_INITIAL', False):
                initial_mask = jnp.triu(jnp.ones((context_backward, context_backward)))
                mask = mask.at[:context_backward, :context_backward].set(
                    mask[:context_backward, :context_backward] | initial_mask
                )
        
        # Convert binary mask to attention mask (0 -> -inf, 1 -> 0)
        return jnp.where(mask == 0, -jnp.inf, 0.0)
    
    def info_nce_loss(self, features: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute InfoNCE loss for contrastive learning"""
        batch_size = features.shape[0] // 2
        
        # Create labels
        labels = jnp.concatenate([jnp.arange(batch_size) for _ in range(self.n_views)])
        labels = (labels[:, None] == labels[None, :]).astype(jnp.float32)
        
        # Normalize features
        features = features / jnp.linalg.norm(features, axis=1, keepdims=True)
        
        # Compute similarity matrix
        similarity_matrix = jnp.matmul(features, features.T)
        
        # Remove diagonal
        mask = jnp.eye(labels.shape[0], dtype=bool)
        labels = labels[~mask].reshape(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].reshape(similarity_matrix.shape[0], -1)
        
        # Extract positives and negatives
        positives = similarity_matrix[labels.astype(bool)].reshape(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.astype(bool)].reshape(similarity_matrix.shape[0], -1)
        
        # Compute logits
        logits = jnp.concatenate([positives, negatives], axis=1)
        labels = jnp.zeros(logits.shape[0], dtype=jnp.int32)
        
        logits = logits / self.temperature
        return logits, labels
    
    def __call__(
        self,
        src: jnp.ndarray,
        mask_labels: jnp.ndarray,
        contrast_src1: Optional[jnp.ndarray] = None,
        contrast_src2: Optional[jnp.ndarray] = None,
        val_phase: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
        return_weights: bool = False,
        return_outputs: bool = False
    ) -> Tuple:
        """Forward pass of STNDT"""
        
        if key is None:
            key = jr.PRNGKey(0)
            
        # Prepare spatial source (N, B, T)
        spatial_src = src.transpose(2, 0, 1)
        
        # Embed spatial features
        # Note: spatial_embedder is identity when LINEAR_EMBEDDER=False and EMBED_DIM=0
        spatial_src = spatial_src * self.spatial_scale
        key, subkey = jr.split(key)
        spatial_src = self.spatial_pos_encoder(spatial_src, key=subkey)
        
        # Prepare temporal source (T, B, N)
        src = src.transpose(1, 0, 2)
        
        # Embed temporal features
        src = jax.vmap(self.embedder)(src) * self.scale
        key, subkey = jr.split(key)
        src = self.src_pos_encoder(src, key=subkey)
        
        # Generate masks
        src_mask = self._generate_context_mask(
            src.shape[0],
            self.config.get('CONTEXT_FORWARD', -1),
            self.config.get('CONTEXT_BACKWARD', 0)
        )
        
        # Pass through encoder layers
        layer_outputs = []
        layer_weights = []
        
        for i, encoder in enumerate(self.encoder_layers):
            key, subkey = jr.split(key)
            if i == 0:
                # First layer: x_T is temporal (T, B, N), x_S is spatial (N, B, T)
                src, spatial_weights = encoder(src.transpose(1, 0, 2), 
                                             spatial_src.transpose(1, 0, 2), 
                                             key=subkey)
                src = src.transpose(1, 0, 2)  # Back to (T, B, N)
            else:
                # Subsequent layers: both inputs are (B, T, N) -> convert to expected format
                src_for_layer = src.transpose(1, 0, 2)  # (T, B, N) -> (B, T, N)
                spatial_for_layer = src.transpose(1, 0, 2).transpose(0, 2, 1)  # (T, B, N) -> (B, T, N) -> (B, N, T)
                src_out, spatial_weights = encoder(src_for_layer, spatial_for_layer, key=subkey)
                src = src_out.transpose(1, 0, 2)  # Back to (T, B, N)
            
            if return_outputs:
                layer_outputs.append(src.transpose(1, 0, 2))  # Store as (B, T, N)
            if return_weights:
                layer_weights.append(spatial_weights)
        
        # Apply final norm
        if self.norm is not None:
            # src has shape (T, B, N), apply norm along N dimension
            src = jax.vmap(jax.vmap(self.norm))(src)
        
        # Apply rate dropout and decoder
        key, subkey = jr.split(key)
        if val_phase:
            encoder_output = self.rate_dropout(src, inference=True)
        else:
            encoder_output = self.rate_dropout(src, key=subkey)
        # encoder_output has shape (T, B, N), apply decoder along last dimension
        decoder_output = jax.vmap(jax.vmap(self.src_decoder))(encoder_output)
        
        # Compute decoder loss
        decoder_rates = decoder_output.transpose(1, 0, 2)  # (B, T, N)
        
        # Poisson loss computation
        if self.config.get('LOSS', {}).get('TYPE') == "poisson":
            if self.config.get('LOGRATE', False):
                # Log-space Poisson loss
                decoder_loss = mask_labels * jnp.exp(decoder_rates) - decoder_rates * mask_labels
            else:
                # Standard Poisson loss
                decoder_loss = decoder_rates - mask_labels * jnp.log(decoder_rates + 1e-8)
        
        # Masked decoder loss
        mask = mask_labels != -1  # Assuming -1 is UNMASKED_LABEL
        masked_decoder_loss = jnp.where(mask, decoder_loss, 0.0)
        
        # Handle contrastive loss
        contrast_loss = jnp.array(0.0)
        if contrast_src1 is not None and contrast_src2 is not None:
            # Process first contrast source
            spatial_contrast1 = contrast_src1.transpose(2, 0, 1) * self.spatial_scale
            key, subkey = jr.split(key)
            spatial_contrast1 = self.spatial_pos_encoder(spatial_contrast1, key=subkey)
            
            contrast_src1 = contrast_src1.transpose(1, 0, 2)
            contrast_src1 = jax.vmap(self.embedder)(contrast_src1) * self.scale
            key, subkey = jr.split(key)
            contrast_src1_embedded = self.src_pos_encoder(contrast_src1, key=subkey)
            
            # Pass through encoder layers for contrast1
            layer_outputs_contrast1 = []
            src1 = contrast_src1_embedded
            for i, encoder in enumerate(self.encoder_layers):
                key, subkey = jr.split(key)
                if i == 0:
                    src1, _ = encoder(src1.transpose(1, 0, 2), 
                                     spatial_contrast1.transpose(1, 0, 2), 
                                     key=subkey)
                    src1 = src1.transpose(1, 0, 2)
                else:
                    src1_for_layer = src1.transpose(1, 0, 2)
                    spatial1_for_layer = src1.transpose(1, 0, 2).transpose(0, 2, 1)
                    src1_out, _ = encoder(src1_for_layer, spatial1_for_layer, key=subkey)
                    src1 = src1_out.transpose(1, 0, 2)
                layer_outputs_contrast1.append(src1.transpose(1, 0, 2))
            
            if self.norm is not None:
                src1 = jax.vmap(jax.vmap(self.norm))(src1)
            
            key, subkey = jr.split(key)
            if val_phase:
                encoder_output_contrast1 = self.rate_dropout(src1, inference=True)
            else:
                encoder_output_contrast1 = self.rate_dropout(src1, key=subkey)
            decoder_output_contrast1 = jax.vmap(jax.vmap(self.src_decoder))(encoder_output_contrast1)
            
            # Process second contrast source
            spatial_contrast2 = contrast_src2.transpose(2, 0, 1) * self.spatial_scale
            key, subkey = jr.split(key)
            spatial_contrast2 = self.spatial_pos_encoder(spatial_contrast2, key=subkey)
            
            contrast_src2 = contrast_src2.transpose(1, 0, 2)
            contrast_src2 = jax.vmap(self.embedder)(contrast_src2) * self.scale
            key, subkey = jr.split(key)
            contrast_src2_embedded = self.src_pos_encoder(contrast_src2, key=subkey)
            
            # Pass through encoder layers for contrast2
            layer_outputs_contrast2 = []
            src2 = contrast_src2_embedded
            for i, encoder in enumerate(self.encoder_layers):
                key, subkey = jr.split(key)
                if i == 0:
                    src2, _ = encoder(src2.transpose(1, 0, 2), 
                                     spatial_contrast2.transpose(1, 0, 2), 
                                     key=subkey)
                    src2 = src2.transpose(1, 0, 2)
                else:
                    src2_for_layer = src2.transpose(1, 0, 2)
                    spatial2_for_layer = src2.transpose(1, 0, 2).transpose(0, 2, 1)
                    src2_out, _ = encoder(src2_for_layer, spatial2_for_layer, key=subkey)
                    src2 = src2_out.transpose(1, 0, 2)
                layer_outputs_contrast2.append(src2.transpose(1, 0, 2))
            
            if self.norm is not None:
                src2 = jax.vmap(jax.vmap(self.norm))(src2)
                
            key, subkey = jr.split(key)
            if val_phase:
                encoder_output_contrast2 = self.rate_dropout(src2, inference=True)
            else:
                encoder_output_contrast2 = self.rate_dropout(src2, key=subkey)
            decoder_output_contrast2 = jax.vmap(jax.vmap(self.src_decoder))(encoder_output_contrast2)
            
            # Select which layer to use for contrastive loss
            contrast_layer = self.config.get('CONTRAST_LAYER', 'decoder')
            if contrast_layer == 'embedder':
                out1 = contrast_src1_embedded.transpose(1, 0, 2)  # (B, T, N)
                out2 = contrast_src2_embedded.transpose(1, 0, 2)
            elif contrast_layer == 'decoder':
                out1 = decoder_output_contrast1.transpose(1, 0, 2)  # (B, T, N)
                out2 = decoder_output_contrast2.transpose(1, 0, 2)
            else:
                # Use specific encoder layer output
                layer_idx = int(contrast_layer) if isinstance(contrast_layer, (int, str)) and str(contrast_layer).isdigit() else -1
                out1 = layer_outputs_contrast1[layer_idx]
                out2 = layer_outputs_contrast2[layer_idx]
            
            # Apply projector and flatten
            out1 = jax.vmap(jax.vmap(self.projector))(out1)  # (B, T, N)
            out2 = jax.vmap(jax.vmap(self.projector))(out2)  # (B, T, N)
            out1 = out1.reshape(out1.shape[0], -1)  # (B, T*N)
            out2 = out2.reshape(out2.shape[0], -1)  # (B, T*N)
            
            # Concatenate features from both views
            features = jnp.concatenate([out1, out2], axis=0)  # (2*B, T*N)
            
            # Compute InfoNCE loss
            logits, labels = self.info_nce_loss(features)
            
            # Cross entropy loss
            log_probs = jax.nn.log_softmax(logits, axis=1)
            ce_loss = -log_probs[jnp.arange(logits.shape[0]), labels]
            
            if not val_phase:
                contrast_loss = jnp.mean(ce_loss) * self.contrast_lambda
            else:
                contrast_loss = ce_loss * self.contrast_lambda
                
            # If using embedder layer, also compute spatial contrastive loss
            if contrast_layer == 'embedder' and self.config.get('USE_SPATIAL_CONTRAST', True):
                # Use spatial embeddings
                spatial_embed1 = spatial_contrast1.transpose(1, 0, 2)  # (B, N, T)
                spatial_embed2 = spatial_contrast2.transpose(1, 0, 2)
                
                spatial_out1 = jax.vmap(jax.vmap(self.spatial_projector))(spatial_embed1)
                spatial_out2 = jax.vmap(jax.vmap(self.spatial_projector))(spatial_embed2)
                spatial_out1 = spatial_out1.reshape(spatial_out1.shape[0], -1)
                spatial_out2 = spatial_out2.reshape(spatial_out2.shape[0], -1)
                
                # Compute spatial contrastive loss
                spatial_features = jnp.concatenate([spatial_out1, spatial_out2], axis=0)
                spatial_logits, spatial_labels = self.info_nce_loss(spatial_features)
                
                spatial_log_probs = jax.nn.log_softmax(spatial_logits, axis=1)
                spatial_ce_loss = -spatial_log_probs[jnp.arange(spatial_logits.shape[0]), spatial_labels]
                
                if not val_phase:
                    contrast_loss = contrast_loss + jnp.mean(spatial_ce_loss) * self.contrast_lambda
                else:
                    contrast_loss = contrast_loss + spatial_ce_loss * self.contrast_lambda
        
        # Aggregate losses
        if not val_phase:
            masked_decoder_loss = jnp.mean(masked_decoder_loss)
            loss = masked_decoder_loss + contrast_loss
        else:
            loss = masked_decoder_loss
        
        # Return based on val_phase
        if not val_phase:
            return (
                jnp.expand_dims(loss, 0),
                jnp.expand_dims(masked_decoder_loss, 0),
                jnp.expand_dims(contrast_loss, 0),
                decoder_rates,
            )
        else:
            return (
                loss,
                masked_decoder_loss,
                contrast_loss,
                decoder_rates,
                layer_weights if return_weights else None,
                jnp.stack(layer_outputs, axis=-1) if return_outputs else None,
            )


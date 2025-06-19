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
    dropout : eqx.nn.Dropout


    def __init__(self, n_embed, head_size, dropout=0.1, key=None):
        if key is None:
            key = jr.PRNGKey(0)
    
        key_k, key_q, key_v = jr.split(key, 3)

        self.k = eqx.nn.Linear(in_features=n_embed, out_features=head_size, use_bias=False, key=key_k)
        self.q = eqx.nn.Linear(in_features=n_embed, out_features=head_size, use_bias=False, key=key_q)
        self.v = eqx.nn.Linear(in_features=n_embed, out_features=head_size, use_bias=False, key=key_v)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x, src_mask=None, key=None):
        B, T, C = x.shape
        k = jax.vmap(jax.vmap(self.k))(x)  # (B, T, head_size)
        q = jax.vmap(jax.vmap(self.q))(x)  
        v = jax.vmap(jax.vmap(self.v))(x)  
        k_transpose = jnp.transpose(k, (0, 2, 1))
        scores = jnp.matmul(q, k_transpose) / jnp.sqrt(k.shape[-1])
        
        # Apply context mask if provided (limits attention window)
        if src_mask is not None:
            scores = scores + src_mask[:T, :T]
        
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


    def __init__(self, n_embed, num_heads, dropout=0.1, key=None):
        if key is None:
            key = jr.PRNGKey(2)

        head_size = n_embed // num_heads
        keys = jr.split(key, num_heads + 1)
        self.heads = [TemporalHead(n_embed=n_embed, head_size=head_size, dropout=dropout, key=keys[i]) for i in range(num_heads)]
        #ouput projection, 
        self.linear = eqx.nn.Linear(in_features=(num_heads * head_size), out_features=n_embed, use_bias=False, key=keys[-1])
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x, src_mask=None, key=None):
        head_out = [head(x, src_mask=src_mask, key=key) for head in self.heads]
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
    activation: eqx.nn.PReLU
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    ts_linear1: eqx.nn.Linear
    ts_linear2: eqx.nn.Linear
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm 
    ts_norm1: eqx.nn.LayerNorm
    ts_norm2: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    dropout1: eqx.nn.Dropout
    ts_dropout1: eqx.nn.Dropout
    ts_dropout2: eqx.nn.Dropout
    ts_dropout3: eqx.nn.Dropout
    spatial_norm_1: eqx.nn.LayerNorm


    def __init__(self, n_embed, num_heads, max_length=100, dropout=0.1, hidden_size=None, key=None):
        if key is None:
            key = jr.PRNGKey(5)

        keys = jr.split(key, 3)
        self.temporal_attention = MultiheadTemporalAttention(n_embed=n_embed, num_heads=num_heads, dropout=dropout, key=keys[0])
        self.spatial_attention = MultiheadSpatialAttention(trial_length=max_length, num_heads=num_heads, dropout=dropout, key=keys[1])
        key_ln1, key_ln2, key_ts_linear1, key_ts_linear2 = jr.split(keys[2], 4)
        self.activation = eqx.nn.PReLU(init_alpha=0.01)
        self.linear1 = eqx.nn.Linear(in_features=n_embed, out_features=hidden_size, use_bias=True, key=key_ln1)
        self.linear2 = eqx.nn.Linear(in_features=hidden_size, out_features=n_embed, use_bias=True, key=key_ln2)
        self.ts_linear1 = eqx.nn.Linear(in_features=n_embed, out_features=hidden_size, use_bias=True, key=key_ts_linear1)
        self.ts_linear2 = eqx.nn.Linear(in_features=hidden_size, out_features=n_embed, use_bias=True, key=key_ts_linear2)
        self.norm1 = eqx.nn.LayerNorm(n_embed)
        self.norm2 = eqx.nn.LayerNorm(n_embed)
        self.ts_norm1 = eqx.nn.LayerNorm(n_embed)
        self.ts_norm2 = eqx.nn.LayerNorm(n_embed)
        self.dropout = eqx.nn.Dropout(dropout)
        self.dropout1 = eqx.nn.Dropout(dropout)
        self.ts_dropout1 = eqx.nn.Dropout(dropout)
        self.ts_dropout2 = eqx.nn.Dropout(dropout)
        self.ts_dropout3 = eqx.nn.Dropout(dropout)
        self.spatial_norm_1 = eqx.nn.LayerNorm(max_length)

    def __call__(self, src, spatial_src, src_mask=None, key=None, prenorm=True):
        if key is None:
            key = jr.PRNGKey(6)
        
        # Split keys properly for all dropout operations
        keys = jr.split(key, 8)  # Need 8 different keys
        key_t, key_drop_1, key_drop_ff1, key_drop_ff2, key_spatial, key_ts_drop1, key_ts_drop2, key_ts_drop3 = keys
        
        residual = src
        if prenorm:
            src = jax.vmap(jax.vmap(self.norm1))(src)

        # Apply temporal attention
        t_out = self.temporal_attention(src, src_mask=src_mask, key=key_t)
        src = residual + (self.dropout1(t_out, key=key_drop_1) if key is not None else self.dropout1(t_out, inference=True))

        if not prenorm:
            src = jax.vmap(jax.vmap(self.norm1))(src)
        residual = src
        if prenorm:
            src = jax.vmap(jax.vmap(self.norm2))(src)
        
        src2 = jax.vmap(jax.vmap(self.linear1))(src)
        src2 = jax.vmap(jax.vmap(self.activation))(src2)
        src2 = self.dropout(src2, key=key_drop_ff1) if key is not None else self.dropout(src2, inference=True)
        src2 = jax.vmap(jax.vmap(self.linear2))(src2)
        
        src = residual + (self.dropout(src2, key=key_drop_ff2) if key is not None else self.dropout(src2, inference=True))

        if not prenorm:
            src = jax.vmap(jax.vmap(self.norm2))(src)

        spatial_residual = spatial_src
        if prenorm:
            spatial_src = jax.vmap(jax.vmap(self.spatial_norm_1))(spatial_src)
        spatial_weights = self.spatial_attention(spatial_src, key=key_spatial)

        ts_residual = src
        if prenorm:
            src = jax.vmap(jax.vmap(self.ts_norm1))(src)
        
        ts_out = jnp.matmul(spatial_weights, src.transpose(0, 2, 1)).transpose(0, 2, 1)  # (B, T, N)
        ts_out = ts_residual + (self.ts_dropout1(ts_out, key=key_ts_drop1) if key is not None else self.ts_dropout1(ts_out, inference=True))
        if not prenorm:
            ts_out = jax.vmap(jax.vmap(self.ts_norm1))(ts_out)

        ts_residual = ts_out
        if prenorm:
            ts_out = jax.vmap(jax.vmap(self.ts_norm2))(ts_out)

        ts_out = jax.vmap(jax.vmap(self.ts_linear1))(ts_out)
        ts_out = jax.vmap(jax.vmap(self.activation))(ts_out)
        ts_out = self.ts_dropout2(ts_out, key=key_ts_drop2) if key is not None else self.ts_dropout2(ts_out, inference=True)
        ts_out = jax.vmap(jax.vmap(self.ts_linear2))(ts_out)

        ts_out = ts_residual + (self.ts_dropout3(ts_out, key=key_ts_drop3) if key is not None else self.ts_dropout3(ts_out, inference=True))
        if not prenorm:
            ts_out = jax.vmap(jax.vmap(self.ts_norm2))(ts_out)

        return ts_out, spatial_weights  

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
    projector: eqx.nn.Sequential
    spatial_projector: eqx.nn.Sequential
    rate_dropout: eqx.nn.Dropout
    
    # Model parameters
    num_neurons: int
    trial_length: int
    scale: float
    spatial_scale: float
    n_views: int
    temperature: float
    contrast_lambda: float
    max_spikes: int
    
    # Decoder (using Sequential)
    decoder: eqx.nn.Sequential
    
    # Configuration (static, not trainable)
    config: Dict[str, Any] = eqx.static_field()
    
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
        self.max_spikes = max_spikes
        self.n_views = 2
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
                hidden_size=config.get('HIDDEN_SIZE', 64),  # Pass hidden_size 
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
                key1_split1, key1_split2 = jr.split(key1)
                key2_split1, key2_split2 = jr.split(key2)
                # Use Sequential with PReLU wrapped in Lambda to handle key passing
                self.projector = eqx.nn.Sequential([
                    eqx.nn.Linear(num_neurons, 1024, key=key1_split1),
                    eqx.nn.Lambda(eqx.nn.PReLU(init_alpha=0.01)),
                    eqx.nn.Linear(1024, num_neurons, key=key1_split2)
                ])
                
                self.spatial_projector = eqx.nn.Sequential([
                    eqx.nn.Linear(trial_length, 1024, key=key2_split1),
                    eqx.nn.Lambda(eqx.nn.PReLU(init_alpha=0.01)),
                    eqx.nn.Linear(1024, trial_length, key=key2_split2)
                ])
        else:
            self.projector = eqx.nn.Identity()
            self.spatial_projector = eqx.nn.Identity()
        
        # Rate dropout
        self.rate_dropout = eqx.nn.Dropout(config.get('DROPOUT_RATES', 0.0))
        
        # Decoder
        key1, key2, key = jr.split(key, 3)
        initrange = config.get('INIT_RANGE', 0.1)
        if config.get('LOSS', {}).get('TYPE') == "poisson":
            decoder_layers = []
            
            if config.get('DECODER', {}).get('LAYERS', 1) == 1:
                # Single layer decoder with custom initialization
                temp_linear = eqx.nn.Linear(num_neurons, num_neurons, key=key1)
                key, weight_key = jr.split(key, 2)
                weights = jr.uniform(weight_key, temp_linear.weight.shape, minval=-initrange, maxval=initrange)
                bias = jnp.zeros_like(temp_linear.bias)
                first_linear = eqx.tree_at(lambda layer: (layer.weight, layer.bias),
                                          temp_linear, (weights, bias))
                decoder_layers.append(first_linear)
            else:
                # Two layer decoder with custom initialization for first layer
                temp_linear = eqx.nn.Linear(num_neurons, 16, key=key1)
                key, weight_key = jr.split(key, 2)
                weights = jr.uniform(weight_key, temp_linear.weight.shape, minval=-initrange, maxval=initrange)
                bias = jnp.zeros_like(temp_linear.bias)
                first_linear = eqx.tree_at(lambda layer: (layer.weight, layer.bias),
                                          temp_linear, (weights, bias))
                decoder_layers.extend([
                    first_linear,
                    eqx.nn.Lambda(eqx.nn.PReLU(init_alpha=0.01)),
                    eqx.nn.Linear(16, num_neurons, key=key2)
                ])
            
            # Add final activation if not using lograte
            if not config.get('LOGRATE', False):
                decoder_layers.append(eqx.nn.Lambda(jax.nn.softplus))
            
            self.decoder = eqx.nn.Sequential(decoder_layers)
    
    def _generate_context_mask(self, size: int, context_forward: int = -1, 
                             context_backward: int = 0, expose_ic: bool = True) -> jnp.ndarray:
        """Generate context mask for attention"""
        if self.config.get('FULL_CONTEXT', False):
            return None
            
        if context_forward < 0:
            context_forward = size
            
        # Create forward-looking mask (boolean)
        mask = jnp.triu(jnp.ones((size, size), dtype=bool), k=-context_forward).T
        
        # Apply backward context if specified
        if context_backward > 0:
            back_mask = jnp.triu(jnp.ones((size, size), dtype=bool), k=-context_backward)
            mask = mask & back_mask
            
            # Expose initial segment for initial conditions
            if expose_ic and self.config.get('CONTEXT_WRAP_INITIAL', False):
                initial_mask = jnp.triu(jnp.ones((context_backward, context_backward), dtype=bool))
                mask = mask.at[:context_backward, :context_backward].set(
                    mask[:context_backward, :context_backward] | initial_mask
                )
        
        # Convert binary mask to attention mask (False -> -inf, True -> 0)
        return jnp.where(mask, 0.0, -jnp.inf)
    
    
    
    def info_nce_loss(self, features: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute InfoNCE loss for contrastive learning
        Adapted from PyTorch implementation but using JAX-compatible operations
        """
        batch_size = features.shape[0] // 2
        
        # Create labels - same as PyTorch
        labels = jnp.concatenate([jnp.arange(batch_size) for _ in range(self.n_views)])
        labels = (labels[:, None] == labels[None, :]).astype(jnp.float32)
        
        # Normalize features
        features = features / jnp.linalg.norm(features, axis=1, keepdims=True)
        
        # Compute similarity matrix
        similarity_matrix = jnp.matmul(features, features.T)
        
        # Remove diagonal using masking (JAX-compatible way)
        eye_mask = jnp.eye(labels.shape[0])
        # Use where to remove diagonal elements
        labels_no_diag = jnp.where(eye_mask[..., None], 0, labels.reshape(labels.shape[0], labels.shape[1], 1))
        labels_no_diag = labels_no_diag.reshape(-1, labels.shape[1])
        
        sim_no_diag = jnp.where(eye_mask[..., None], 0, similarity_matrix.reshape(similarity_matrix.shape[0], similarity_matrix.shape[1], 1))
        sim_no_diag = sim_no_diag.reshape(-1, similarity_matrix.shape[1])
        
        # Extract positives and negatives using where instead of boolean indexing
        pos_mask = labels_no_diag.astype(bool)
        neg_mask = ~pos_mask
        
        # For each row, get positive and negative similarities
        positives = jnp.where(pos_mask, sim_no_diag, -jnp.inf)
        negatives = jnp.where(neg_mask, sim_no_diag, -jnp.inf)
        
        # Take max positive per row (should be only one)
        positives = jnp.max(positives, axis=1, keepdims=True)
        
        # Concatenate positives and negatives
        logits = jnp.concatenate([positives, negatives], axis=1)
        target_labels = jnp.zeros(logits.shape[0], dtype=jnp.int32)
        
        logits = logits / self.temperature
        return logits, target_labels
    
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
        
        # Generate context mask for attention
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
                                             src_mask=src_mask,
                                             key=subkey)
                src = src.transpose(1, 0, 2)  # Back to (T, B, N)
            else:
                # Subsequent layers: both inputs are (B, T, N) -> convert to expected format
                src_for_layer = src.transpose(1, 0, 2)  # (T, B, N) -> (B, T, N)
                spatial_for_layer = src.transpose(1, 0, 2).transpose(0, 2, 1)  # (T, B, N) -> (B, T, N) -> (B, N, T)
                src_out, spatial_weights = encoder(src_for_layer, spatial_for_layer, 
                                                 src_mask=src_mask, key=subkey)
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
        decoder_output = jax.vmap(jax.vmap(self.decoder))(encoder_output)
        
        # Compute decoder loss
        decoder_rates = decoder_output.transpose(1, 0, 2)  # (B, T, N)
        
        # Poisson loss computation
        if self.config.get('LOSS', {}).get('TYPE') == "poisson":
            if self.config.get('LOGRATE', False):
                # Log-space Poisson loss
                decoder_loss = jnp.exp(decoder_rates) - mask_labels * decoder_rates
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
                                     src_mask=src_mask,
                                     key=subkey)
                    src1 = src1.transpose(1, 0, 2)
                else:
                    src1_for_layer = src1.transpose(1, 0, 2)
                    spatial1_for_layer = src1.transpose(1, 0, 2).transpose(0, 2, 1)
                    src1_out, _ = encoder(src1_for_layer, spatial1_for_layer, 
                                        src_mask=src_mask, key=subkey)
                    src1 = src1_out.transpose(1, 0, 2)
                layer_outputs_contrast1.append(src1.transpose(1, 0, 2))
            
            if self.norm is not None:
                src1 = jax.vmap(jax.vmap(self.norm))(src1)
            
            key, subkey = jr.split(key)
            if val_phase:
                encoder_output_contrast1 = self.rate_dropout(src1, inference=True)
            else:
                encoder_output_contrast1 = self.rate_dropout(src1, key=subkey)
            decoder_output_contrast1 = jax.vmap(jax.vmap(self.decoder))(encoder_output_contrast1)
            
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
                                     src_mask=src_mask,
                                     key=subkey)
                    src2 = src2.transpose(1, 0, 2)
                else:
                    src2_for_layer = src2.transpose(1, 0, 2)
                    spatial2_for_layer = src2.transpose(1, 0, 2).transpose(0, 2, 1)
                    src2_out, _ = encoder(src2_for_layer, spatial2_for_layer, 
                                        src_mask=src_mask, key=subkey)
                    src2 = src2_out.transpose(1, 0, 2)
                layer_outputs_contrast2.append(src2.transpose(1, 0, 2))
            
            if self.norm is not None:
                src2 = jax.vmap(jax.vmap(self.norm))(src2)
                
            key, subkey = jr.split(key)
            if val_phase:
                encoder_output_contrast2 = self.rate_dropout(src2, inference=True)
            else:
                encoder_output_contrast2 = self.rate_dropout(src2, key=subkey)
            decoder_output_contrast2 = jax.vmap(jax.vmap(self.decoder))(encoder_output_contrast2)
            
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
                
                # Apply spatial projector
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


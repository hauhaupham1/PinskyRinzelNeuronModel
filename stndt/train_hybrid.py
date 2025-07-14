#!/usr/bin/env python3
"""
Hybrid STNDT Training: Reconstruction during training, Forward prediction during evaluation
"""

import jax
from get_data_S1 import load_s1_train, load_s1_test, data_loading_for_batch, process_sample_vectorized
from stnd_transformer import STNDT
import jax.numpy as jnp
import jax.random as jr
from losses import compute_forecasting_loss
import optax
import equinox as eqx
from mask_hybrid import create_hybrid_batch
import matplotlib.pyplot as plt
import numpy as np
import time

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_platform_name", "cpu")  

def config():
    return{
        'EMBED_DIM': 1,
        'LINEAR_EMBEDDER': False,
        'USE_CONTRAST_PROJECTOR': True,
        'LINEAR_PROJECTOR' : False,
        'DROPOUT_RATES': 0.1,
        'SCALE_NORM': True,
        'NUM_LAYERS': 6,
        'DECODER': {
            'LAYERS': 1,
        },
        'LOGRATE': True,
        'NUM_HEADS': 2,
        'DROPOUT': 0.1,
        'HIDDEN_SIZE': 64,
        'PRE_NORM': False,
        'FULL_CONTEXT': False,
        'CONTEXT_FORWARD': 0,
        'CONTEXT_BACKWARD': -1,
        # Hybrid training parameters
        'MASK_RATIO': 0.25,  # Reconstruction masking ratio
        'CONTRASTIVE_RATIO': 0.05,  # Contrastive masking ratio  
        'CONTRASTIVE_WEIGHT': 0.1,  # Lambda for contrastive loss
        'USE_CONTRASTIVE': True,  # Whether to use contrastive learning
    }

# Training parameters
batch_size = 128
num_epochs = 10
learning_rate = 1e-2
num_forward_steps = 5  # For evaluation

# Load data
train_data = load_s1_train()
test_data = load_s1_test()
num_batches = len(train_data) // batch_size

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Batches per epoch: {num_batches}")

# Initialize model
key = jr.PRNGKey(42)
model_key, key = jr.split(key)
model = STNDT(
    config=config(),
    trial_length=125, 
    num_neurons=11,
    max_spikes=50,
    key=model_key
)

total_steps = num_epochs * num_batches
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=learning_rate,
    warmup_steps=100,
    decay_steps=total_steps
)

optimizer = optax.chain(
    optax.clip_by_global_norm(200.0),
    optax.adam(learning_rate=schedule, eps=1e-8)
)
params, static = eqx.partition(model, eqx.is_inexact_array)
opt_state = optimizer.init(params)

# Loss functions for hybrid training
def reconstruction_loss_fn(params, batch_data, key):
    """Loss function for reconstruction training"""
    model = eqx.combine(params, static)
    # Create reconstruction mask
    input_data, mask_labels = create_hybrid_batch(
        batch_data, mode='reconstruction', 
        mask_ratio=config()['MASK_RATIO'], 
        key=key
    )
    predictions = model(input_data, key=key)
    return compute_forecasting_loss(predictions=predictions, mask_labels=mask_labels, config=config())

def contrastive_loss_fn(params, batch_data, key):
    """Loss function for contrastive learning"""
    model = eqx.combine(params, static)
    # Create two contrastive views
    (input_1, labels_1), (input_2, labels_2) = create_hybrid_batch(
        batch_data, mode='reconstruction', contrastive=True,
        mask_ratio=config()['CONTRASTIVE_RATIO'], 
        key=key
    )
    
    # Get embeddings for both views (before decoder)
    # We'll use the encoder output as embeddings
    key1, key2 = jr.split(key)
    
    # Forward pass through encoder only (need to modify model to expose encoder output)
    pred_1 = model(input_1, key=key1)
    pred_2 = model(input_2, key=key2)
    
    # Simple contrastive loss: encourage similar predictions for same data
    # This is a simplified version - in practice you'd want proper InfoNCE
    similarity = jnp.mean((pred_1 - pred_2) ** 2)
    return similarity

def forward_loss_fn(params, batch_data, key):
    """Loss function for forward prediction evaluation"""
    model = eqx.combine(params, static)
    input_data, mask_labels = create_hybrid_batch(
        batch_data, mode='forward', 
        num_forward_steps=num_forward_steps
    )
    predictions = model(input_data, key=key)
    return compute_forecasting_loss(predictions=predictions, mask_labels=mask_labels, config=config())

# Training steps
@eqx.filter_jit
def reconstruction_step(params, optimizer, opt_state, batch_data, key):
    loss_value, grads = eqx.filter_value_and_grad(reconstruction_loss_fn)(params, batch_data, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

@eqx.filter_jit  
def contrastive_step(params, optimizer, opt_state, batch_data, key):
    loss_value, grads = eqx.filter_value_and_grad(contrastive_loss_fn)(params, batch_data, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

@eqx.filter_jit
def evaluation_step(params, batch_data, key):
    """Evaluation using forward prediction"""
    model = eqx.combine(params, static)
    input_data, mask_labels = create_hybrid_batch(
        batch_data, mode='forward', 
        num_forward_steps=num_forward_steps
    )
    predictions = model(input_data, key=key)
    loss = compute_forecasting_loss(predictions=predictions, mask_labels=mask_labels, config=config())
    return loss, predictions, mask_labels

# Check backend
print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
print(f"Available devices: {jax.devices()}")
print("Starting hybrid training...")
print(f"Reconstruction mask ratio: {config()['MASK_RATIO']}")
print(f"Contrastive learning: {config()['USE_CONTRASTIVE']}")
print(f"Forward prediction steps: {num_forward_steps}")
print()

for epoch in range(num_epochs):
    epoch_start = time.time()
    # Shuffle data at start of epoch
    key, shuffle_key = jr.split(key)
    shuffled_data = [train_data[i] for i in jr.permutation(shuffle_key, len(train_data))]
    
    # Force garbage collection
    # import gc
    # gc.collect()
    
    epoch_recon_loss = 0.0
    epoch_contrast_loss = 0.0
    
    for batch_idx in range(num_batches):
        batch_data = data_loading_for_batch(shuffled_data, batch_size=batch_size, batch_idx=batch_idx)
        batch_data = batch_data.astype(jnp.int32)
        
        key, train_key = jr.split(key)
        
        # Primary training: Reconstruction
        params, opt_state, recon_loss = reconstruction_step(
            params, optimizer, opt_state, batch_data, train_key
        )
        epoch_recon_loss += recon_loss
        
        # Optional: Contrastive learning (every 3rd batch)
        if config()['USE_CONTRASTIVE'] and batch_idx % 3 == 0:
            key, contrast_key = jr.split(key)
            params, opt_state, contrast_loss = contrastive_step(
                params, optimizer, opt_state, batch_data, contrast_key
            )
            epoch_contrast_loss += contrast_loss
        
        # Print progress
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}, "
                  f"Recon Loss: {recon_loss:.4f}")
    
    avg_recon_loss = epoch_recon_loss / num_batches
    avg_contrast_loss = epoch_contrast_loss / (num_batches // 3) if config()['USE_CONTRASTIVE'] else 0.0
    
    # Evaluation using forward prediction
    key, val_key = jr.split(key)
    val_batch_size = min(batch_size, len(test_data))
    val_indices = jr.choice(val_key, len(test_data), (val_batch_size,), replace=False)
    val_samples = [test_data[i] for i in val_indices]
    
    # Process validation samples
    val_batch_binned = []
    for it, t in val_samples:
        sample_matrix = process_sample_vectorized(it, t)
        val_batch_binned.append(sample_matrix)
    val_batch_data = jnp.array(val_batch_binned).astype(jnp.int32)
    
    # Forward prediction evaluation
    val_loss, predictions, mask_labels = evaluation_step(params, val_batch_data, val_key)
    
    # Extract forward predictions for metrics
    context_length = val_batch_data.shape[1] - num_forward_steps
    ground_truth = val_batch_data[:, context_length:, :]
    pred_timesteps = predictions[:, context_length:, :]
    
    # Compute metrics
    if config()['LOGRATE']:
        pred_rates = jnp.exp(pred_timesteps)
    else:
        pred_rates = pred_timesteps
    
    mae = jnp.mean(jnp.abs(pred_rates - ground_truth))
    mse = jnp.mean((pred_rates - ground_truth) ** 2)
    poisson_nll = jnp.mean(pred_rates - ground_truth * jnp.log(pred_rates + 1e-8))
    
    epoch_time = time.time() - epoch_start
    
    print(f"Epoch {epoch+1}/{num_epochs} completed:")
    print(f"  Reconstruction Loss: {avg_recon_loss:.4f}")
    if config()['USE_CONTRASTIVE']:
        print(f"  Contrastive Loss: {avg_contrast_loss:.4f}")
    print(f"  Forward Prediction Loss: {val_loss:.4f}")
    print(f"  Forward MAE: {mae:.4f}, MSE: {mse:.4f}, Poisson NLL: {poisson_nll:.4f}")
    print(f"  Epoch time: {epoch_time:.1f}s")
    
    # Save predictions for comparison
    if epoch % 2 == 0 or epoch == num_epochs - 1:
        np.savez(f'hybrid_predictions_epoch_{epoch+1}.npz',
                 predictions=np.array(pred_rates),
                 ground_truth=np.array(ground_truth),
                 recon_loss=float(avg_recon_loss),
                 forward_loss=float(val_loss),
                 mae=float(mae),
                 mse=float(mse))
        
        print(f"  Saved predictions to hybrid_predictions_epoch_{epoch+1}.npz")
    
    print()

print("Hybrid training completed!")
print("Training used: Masked reconstruction (like PyTorch STNDT)")
print("Evaluation used: Forward prediction (practical forecasting)")
import jax.extend
from get_data_S1 import load_s1_train, load_s1_test, data_loading_for_batch, process_sample_vectorized
from stnd_transformer import STNDT
import jax
import jax.numpy as jnp
import jax.random as jr
from losses import compute_forecasting_loss
import optax
import equinox as eqx
from mask import create_forward_prediction_mask
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_platform_name", "cpu")  

# Training configuration
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
        'NUM_HEADS': 2,  # Reduced from 8 for faster training
        'DROPOUT': 0.1,
        'HIDDEN_SIZE': 64,
        'PRE_NORM': False,
        'FULL_CONTEXT': False,
        'CONTEXT_FORWARD': 0,
        'CONTEXT_BACKWARD': -1,
        'MASK_STRATEGY': 'all',  # 'all', 'only_spikes', 'balanced', or 'weighted'
        'SPIKE_WEIGHT': 5.0,  # Weight multiplier for spike positions when using 'weighted' strategy
    }



# Training parameters (epoch-based approach)
batch_size = 128  # Reduce for memory stability
num_epochs = 10  
learning_rate = 1e-2
num_forward_steps = 5

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
    warmup_steps=100,  # 100 warmup steps
    decay_steps=total_steps
)

optimizer = optax.chain(
    optax.clip_by_global_norm(200.0),  # Gradient clipping
    optax.adam(learning_rate=schedule, eps=1e-8)
)
params, static = eqx.partition(model, eqx.is_inexact_array)
opt_state = optimizer.init(params)

# Define loss function for forecasting
def loss_fn(params, batch_data, key):
    # Reconstruct model from params and static parts
    model = eqx.combine(params, static)
    mask_strategy = config().get('MASK_STRATEGY', 'all')
    input_data, mask_labels = create_forward_prediction_mask(batch_data, num_forward_steps, mask_strategy=mask_strategy)
    predictions = model(input_data, key=key)
    return compute_forecasting_loss(predictions=predictions, mask_labels=mask_labels, config=config())

# Training step with parameter filtering (adapted from train_transformer.py)
@eqx.filter_jit
def train_step_filtered(params, static, optimizer, opt_state, batch_data, key):
    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(params, batch_data, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

# Validation function that also returns predictions
@eqx.filter_jit
def validation_step(params, static, batch_data, key):
    model = eqx.combine(params, static)
    # Always use 'all' strategy for evaluation - no special masking
    input_data, mask_labels = create_forward_prediction_mask(batch_data, num_forward_steps, mask_strategy='all')
    predictions = model(input_data, key=key)
    # Create a config without weighted strategy for evaluation
    eval_config = config().copy()
    eval_config['MASK_STRATEGY'] = 'all'  # Ensure no weighting in evaluation loss
    loss = compute_forecasting_loss(predictions=predictions, mask_labels=mask_labels, config=eval_config)
    return loss, predictions, mask_labels

# Check if Metal is available
import jax
print(f"JAX backend: {jax.extend.backend.get_backend()}")
print(f"Available devices: {jax.devices()}")
print("Starting training...")
print(f"Training MASK_STRATEGY: {config().get('MASK_STRATEGY', 'all')}")
print(f"Evaluation: Always uses 'all' strategy (no special masking)")
print("  - 'all': Train on all positions")
print("  - 'weighted': All positions with 5x weight for spikes")
print()

for epoch in range(num_epochs):
    # Shuffle data at start of epoch
    key, shuffle_key = jr.split(key)
    shuffled_data = [train_data[i] for i in jr.permutation(shuffle_key, len(train_data))]
    
    # Force garbage collection at start of epoch
    import gc
    gc.collect()
    
    epoch_loss = 0.0
    for batch_idx in range(num_batches):
        # Use data_loading_for_batch with shuffled data
        batch_data = data_loading_for_batch(shuffled_data, batch_size=batch_size, batch_idx=batch_idx)
        batch_data = batch_data.astype(jnp.int32)
        
        # Training step
        key, train_key = jr.split(key)
        params, opt_state, loss_value = train_step_filtered(params, static, optimizer, opt_state, batch_data, train_key)
        epoch_loss += loss_value
        
        # Print progress every 50 batches
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}, Loss: {loss_value:.4f}")
    
    avg_train_loss = epoch_loss / num_batches
    
    # Validation at end of each epoch - use random test samples
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
    val_loss, predictions, mask_labels = validation_step(params, static, val_batch_data, val_key)
    
    # Extract predictions and ground truth for the forward timesteps
    # Get the timesteps we're predicting (last num_forward_steps)
    context_length = val_batch_data.shape[1] - num_forward_steps
    
    # Ground truth for predicted timesteps
    ground_truth = val_batch_data[:, context_length:, :]  # (batch, num_forward_steps, num_neurons)
    
    # Model predictions for those timesteps
    pred_timesteps = predictions[:, context_length:, :]  # (batch, num_forward_steps, num_neurons)
    
    # Compute metrics
    # With LOGRATE=True, predictions are log-rates, convert to rates for metrics
    if config()['LOGRATE']:
        pred_rates = jnp.exp(pred_timesteps)  # Convert log(λ) to λ
    else:
        pred_rates = pred_timesteps
    
    mae = jnp.mean(jnp.abs(pred_rates - ground_truth))
    mse = jnp.mean((pred_rates - ground_truth) ** 2)
    
    # Poisson NLL: E[λ - y*log(λ)]
    poisson_nll = jnp.mean(pred_rates - ground_truth * jnp.log(pred_rates + 1e-8))
    
    # Count how many positions have spikes vs no spikes
    num_spike_positions = jnp.sum(ground_truth > 0)
    total_positions = ground_truth.size
    spike_ratio = num_spike_positions / total_positions
    
    print(f"Epoch {epoch+1}/{num_epochs} completed:")
    print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"  MAE: {mae:.4f}, MSE: {mse:.4f}, Poisson NLL: {poisson_nll:.4f}")
    print(f"  Spike positions: {num_spike_positions}/{total_positions} ({spike_ratio:.2%})")
    
    # Visualize predictions vs ground truth for first sample and first few neurons
    if epoch % 2 == 0 or epoch == num_epochs - 1:  # Plot every 2 epochs and final
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f'Predictions vs Ground Truth - Epoch {epoch+1}')
        
        for i, ax in enumerate(axes.flat):
            if i < min(12, ground_truth.shape[2]):  # Show up to 12 neurons or max available
                # Plot ground truth
                gt = ground_truth[0, :, i]  # First sample, all timesteps, neuron i
                # Use converted rates for plotting
                if config()['LOGRATE']:
                    pred = jnp.exp(pred_timesteps[0, :, i])  # Convert log(λ) to λ
                else:
                    pred = pred_timesteps[0, :, i]
                
                timesteps = np.arange(num_forward_steps)
                ax.plot(timesteps, gt, 'ko-', label='Ground Truth', markersize=8)
                ax.plot(timesteps, pred, 'ro--', label='Prediction', markersize=6)
                ax.set_xlabel('Future Timestep')
                ax.set_ylabel('Spike Count')
                ax.set_title(f'Neuron {i}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Fix y-axis to always be 0 to 3
                ax.set_ylim(0, 3)
        
        plt.tight_layout()
        plt.savefig(f'predictions_epoch_{epoch+1}.png')
        plt.close()
        
        # Also save raw predictions for later analysis
        np.savez(f'predictions_epoch_{epoch+1}.npz',
                 predictions=np.array(pred_rates),  # Save converted rates, not log-rates
                 ground_truth=np.array(ground_truth),
                 val_loss=float(val_loss),
                 mae=float(mae),
                 mse=float(mse))
    
    print()

# Reconstruct final model
final_model = eqx.combine(params, static)
print("Training completed!")
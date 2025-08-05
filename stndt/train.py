import jax.extend
from get_data_S1 import load_s1_train, load_s1_test, data_loading_for_batch, process_sample_vectorized
from stnd_transformer import STNDT
import jax
import jax.numpy as jnp
import jax.random as jr
from losses import compute_forecasting_loss, discrete_aware_regression_loss
import optax
import equinox as eqx
from mask import create_forward_prediction_mask, get_mixed_batch, create_reconstruction_mask
from contrast_utils import create_contrast_pairs_pytorch_style
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append('..')

# Training parameters (epoch-based approach)
batch_size = 128
num_epochs = 10  
learning_rate = 1e-3
num_forward_steps = 2

# Load data
train_data = load_s1_train()
test_data = load_s1_test()
max_spikes = 7
num_batches = len(train_data) // batch_size


def config():
    return{
        'EMBED_DIM': 0,
        'LINEAR_EMBEDDER': False, 
        'USE_CONTRAST_PROJECTOR': False,
        'LINEAR_PROJECTOR' : True,
        'DROPOUT_RATES': 0.1,
        'SCALE_NORM': True,
        'NUM_LAYERS': 11,
        'DECODER': {
            'LAYERS': 1,
        },
        'LOSS': {
            'TYPE': 'poisson',  
        },
        'LOGRATE': False,  # Work in log-rate space
        'NUM_HEADS': 11,
        'DROPOUT': 0.1,
        'HIDDEN_SIZE': 256,
        'PRE_NORM': True,
        'FULL_CONTEXT': False,
        'CONTEXT_FORWARD': 0,
        'CONTEXT_BACKWARD': -1,
        'MAX_SPIKES': max_spikes,
        'CONTRAST_LAYER': 'encoder',  # Use encoder output for contrast learning
        'TEMPERATURE': 0.1,  # Temperature for InfoNCE loss
        'LAMBDA': 0.1,  # Weight for contrast loss
        'LEARNABLE_POSITION': False,
    }

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Batches per epoch: {num_batches}")

# Initialize model
key = jr.PRNGKey(42)
model_key, key = jr.split(key)
model = STNDT(
    config=config(),
    trial_length=25,  # Changed to even number for rotary encoding 
    num_neurons=11,
    max_spikes=10,
    key=model_key
)

total_steps = num_epochs * num_batches
warmup_steps = min(10, total_steps // 4)  # Use 10 or 1/4 of total steps
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=learning_rate,
    warmup_steps=warmup_steps,
    decay_steps=total_steps
)

optimizer = optax.chain(
    optax.clip_by_global_norm(200.0),  # Gradient clipping
    optax.adam(learning_rate=schedule, eps=1e-8)
)
params, static = eqx.partition(model, eqx.is_inexact_array)
opt_state = optimizer.init(params)

# Define loss function for mixed training (reconstruction + forecasting + contrast)
def loss_fn(params, batch_data, key, forecast_ratio=0.5):
    # Reconstruct model from params and static parts
    model = eqx.combine(params, static)
    
    # Split key for different operations
    key1, key2, key3 = jr.split(key, 3)
    
    # Create contrast pairs (PyTorch STNDT style) - temporarily disabled
    # contrast_src1, contrast_src2 = create_contrast_pairs_pytorch_style(batch_data, key1)
    contrast_src1, contrast_src2 = None, None
    
    # Get mixed batch: some samples for forecasting, others for reconstruction
    (forecast_input, forecast_labels), (recon_input, recon_labels) = get_mixed_batch(
        batch_data, forecast_ratio=forecast_ratio, num_forward_steps=num_forward_steps, key=key2
    )
    
    # Combine inputs and labels
    mixed_input = jnp.concatenate([forecast_input, recon_input], axis=0)
    mixed_labels = jnp.concatenate([forecast_labels, recon_labels], axis=0)
    
    # Forward pass WITH contrast learning - model computes loss internally
    # Returns: (loss, decoder_loss, contrast_loss) or just loss
    outputs = model.forward(mixed_input, mixed_labels, 
                           contrast_src1=contrast_src1, 
                           contrast_src2=contrast_src2, 
                           key=key3)
    
    # Handle different return formats from model
    if isinstance(outputs, tuple):
        loss = outputs[0]  # Main loss (includes contrast loss)
    else:
        loss = outputs
    
    return loss

# Training step with parameter filtering 
@eqx.filter_jit
def train_step_filtered(params, optimizer, opt_state, batch_data, key):
    # Create a partial function that fixes forecast_ratio
    def loss_partial(params):
        return loss_fn(params, batch_data, key, forecast_ratio=0.3)
    
    loss_value, grads = eqx.filter_value_and_grad(loss_partial)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

# Validation function that also returns predictions
@eqx.filter_jit
def validation_step(params, batch_data, key):
    model = eqx.combine(params, static)
    
    # Use forecasting task for validation
    input_data, mask_labels = create_forward_prediction_mask(batch_data, num_forward_steps)
    # input_data, mask_labels = create_reconstruction_mask(
    #     batch_data, key, mask_ratio=0.15, mode="full",
    #     mask_token_ratio=0.8, use_zero_mask=False, max_spikes=max_spikes
    # )
    
    # Model computes loss internally
    outputs = model.forward(input_data, mask_labels, key=key, val_phase=True)
    
    # Handle different return formats
    if isinstance(outputs, tuple):
        loss = outputs[0]
        predictions = outputs[3] if len(outputs) > 3 else None  # decoder_rates is at index 3
    else:
        loss = outputs
        predictions = None
    
    return loss, predictions, mask_labels


for epoch in range(num_epochs):
    # Shuffle data at start of epoch
    key, shuffle_key = jr.split(key)
    shuffled_data = [train_data[i] for i in jr.permutation(shuffle_key, len(train_data))]
    
    # Force garbage collection at start of epoch
    epoch_loss = 0.0
    for batch_idx in range(num_batches):
        # Use data_loading_for_batch with shuffled data
        batch_data = data_loading_for_batch(shuffled_data, batch_size=batch_size, batch_idx=batch_idx)
        batch_data = batch_data.astype(jnp.int32)
        
        # Training step
        key, train_key = jr.split(key)
        params, opt_state, loss_value = train_step_filtered(params, optimizer, opt_state, batch_data, train_key)
        epoch_loss += loss_value
        
        # Print progress every 50 batches
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}, Loss: {loss_value.item():.4f}")
    
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
    # Validation section
    val_loss, predictions, mask_labels = validation_step(params, val_batch_data, val_key)
    context_length = val_batch_data.shape[1] - num_forward_steps
    ground_truth = val_batch_data[:, context_length:, :]
    
    # Convert log-rates to rates, then to discrete spike counts
    if predictions is not None:
        if config()['LOGRATE']:
            # Convert log-rates to rates
            pred_rates = jnp.exp(predictions[:, context_length:, :])
        else:
            pred_rates = predictions[:, context_length:, :]
        
        # Convert rates to discrete spike counts (round to nearest integer)
        pred_timesteps = jnp.round(pred_rates).astype(jnp.int32)
        pred_timesteps = jnp.clip(pred_timesteps, 0, max_spikes)  # Clip to valid range
        
        #print out last forward steps of predictions and ground truth
        print(f"Raw log-rates: {predictions[:2, context_length:, :]}")  # Show first 2 samples, 3 neurons
        # print(f"Converted rates: {pred_rates[:2, :, :]}")  # Show 3 neurons instead of 2
        print(f"Final predictions: {pred_timesteps[:2, :, :]}")  # Show 3 neurons instead of 2
    else:
        pred_timesteps = None
        pred_rates = None
        print(f"Raw log-rates: None")
        print(f"Converted rates: None") 
        print(f"Final predictions: None")
    print(f"Ground truth: {ground_truth[:2, :, :]}")  # Show ground truth for 5 neurons

    
    # Discrete metrics
    accuracy = jnp.mean(pred_timesteps == ground_truth)
    valid_mask = mask_labels[:, context_length:, :] != -1
    masked_accuracy = jnp.sum((pred_timesteps == ground_truth) * valid_mask) / jnp.sum(valid_mask)
    
    # Confusion matrix (optional)
    confusion = jnp.zeros((max_spikes + 1, max_spikes + 1))
    for true_val in range(max_spikes + 1):
        for pred_val in range(max_spikes + 1):
            confusion = confusion.at[true_val, pred_val].set(
                jnp.sum((ground_truth == true_val) & (pred_timesteps == pred_val))
            )
    
    print(f"Epoch {epoch+1}/{num_epochs} completed:")
    # Compute mean val loss if it's an array
    val_loss_scalar = jnp.mean(val_loss) if val_loss.size > 1 else val_loss.item()
    print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss_scalar:.4f}")
    print(f"  Accuracy: {accuracy.item():.4f}, Masked Accuracy: {masked_accuracy.item():.4f}")
    
    # Visualize predictions vs ground truth for multiple samples
    if epoch % 2 == 0 or epoch == num_epochs - 1:  # Plot every 2 epochs and final
        plots_dir = 'training_plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        num_samples_to_plot = min(3, ground_truth.shape[0])  # Show up to 3
        timesteps = np.arange(num_forward_steps)
        
        for sample_idx in range(num_samples_to_plot):
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            fig.suptitle(f'Sample {sample_idx+1} - Predictions vs Ground Truth - Epoch {epoch+1}')
            
            for i, ax in enumerate(axes.flat):
                if i < min(12, ground_truth.shape[2]):
                    # Plot ground truth and predictions
                    gt = ground_truth[sample_idx, :, i]
                    pred_continuous = pred_rates[sample_idx, :, i]

                    # Plot ground truth as line
                    ax.plot(timesteps, gt, 'k-', linewidth=3, 
                           label='Ground Truth', marker='s', markersize=6)
                    
                    # Plot continuous predictions as line
                    ax.plot(timesteps, pred_continuous, 'r-', linewidth=2, 
                           label='Continuous Pred', marker='o', markersize=4)

                    # Set y-axis to show both continuous and discrete values
                    ax.set_ylim(-0.1, max(max_spikes + 0.5, np.max(pred_continuous) + 0.5))
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Spike Count / Rate')
                    ax.set_title(f'Neuron {i+1}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.axis('off')  # Hide unused subplots
            
            plt.tight_layout()
            # Save plot in the plots directory
            plot_filename = f'predictions_epoch_{epoch+1}_sample_{sample_idx+1}.png'
            plot_path = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
        
    

# Reconstruct final model
final_model = eqx.combine(params, static)
print("Training completed!")
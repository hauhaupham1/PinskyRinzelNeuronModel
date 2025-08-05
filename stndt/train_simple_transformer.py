import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
import os
from get_data_S1 import load_s1_train, load_s1_test, process_sample_vectorized
from mlp_style import SimpleTransformer

# Training parameters
train_batch_size = 30  # Train on 30 samples at a time
test_batch_size = 1    # Test on 1 sample
learning_rate = 1e-3
context_length = 23
future_steps = 2
num_epochs = 10        # Number of times to go through the entire dataset
test_every = 100       # Test every N training steps

# Model parameters
d_model = 256
num_layers = 4
num_heads = 8
d_ff = 512

# Load data
print("Loading data...")
train_data = load_s1_train()
# train_data = train_data[:1000]  # Limit for quick testing
test_data = load_s1_test()

# Process all training data
print("Processing training data...")
train_processed = []
for trial_idx, trial_times in train_data:
    sample_matrix = process_sample_vectorized(trial_idx, trial_times)
    train_processed.append(sample_matrix)

# Process test data
print("Processing test data...")
test_processed = []
for trial_idx, trial_times in test_data:
    sample_matrix = process_sample_vectorized(trial_idx, trial_times)
    test_processed.append(sample_matrix)

train_processed = np.array(train_processed)
test_processed = np.array(test_processed)

print(f"Train data shape: {train_processed.shape}")
print(f"Test data shape: {test_processed.shape}")

def create_forecasting_data(data, context_length=23, future_steps=2):
    """Create X, y pairs for forecasting"""
    X = data[:, :context_length, :]  # (batch, 23, neurons)
    y = data[:, context_length:context_length + future_steps, :]  # (batch, 2, neurons)
    return X, y

# Create train/test splits
X_train, y_train = create_forecasting_data(train_processed, context_length, future_steps)
X_test, y_test = create_forecasting_data(test_processed, context_length, future_steps)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Initialize model
key = jr.PRNGKey(42)
model_key, key = jr.split(key)

input_dim = X_train.shape[-1]  # number of neurons
output_dim = y_train.shape[-1]  # same as input_dim

model = SimpleTransformer(
    input_dim=input_dim,
    output_dim=output_dim,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    future_steps=future_steps,
    key=model_key
)

# Initialize optimizer
optimizer = optax.adam(learning_rate)
params, static = eqx.partition(model, eqx.is_inexact_array)
opt_state = optimizer.init(params)

# Loss function
def loss_fn(params, X, y):
    model = eqx.combine(params, static)
    predictions = model(X)  # (batch, future_steps, neurons)
    return jnp.mean((predictions - y) ** 2)  # MSE loss

# Training step
@eqx.filter_jit
def train_step(params, opt_state, X, y):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(params, X, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Validation step
@eqx.filter_jit
def val_step(params, X, y):
    model = eqx.combine(params, static)
    predictions = model(X)
    loss = jnp.mean((predictions - y) ** 2)
    return predictions, loss

# Training loop: Go through dataset multiple times, test periodically
train_losses = []
test_losses = []
test_predictions_list = []
test_ground_truth_list = []

num_batches_per_epoch = len(X_train) // train_batch_size
total_steps = num_epochs * num_batches_per_epoch

print(f"Starting training...")
print(f"Dataset size: {len(X_train)} samples")
print(f"Batches per epoch: {num_batches_per_epoch}")
print(f"Total epochs: {num_epochs}")
print(f"Total training steps: {total_steps}")
print(f"Will test every {test_every} steps")

step = 0
for epoch in range(num_epochs):
    # Shuffle training data each epoch
    key, shuffle_key = jr.split(key)
    perm = jr.permutation(shuffle_key, len(X_train))
    X_train_shuffled = X_train[perm]
    y_train_shuffled = y_train[perm]
    
    # Go through all batches in this epoch
    for batch_idx in range(num_batches_per_epoch):
        start_idx = batch_idx * train_batch_size
        end_idx = start_idx + train_batch_size
        
        X_train_batch = X_train_shuffled[start_idx:end_idx]
        y_train_batch = y_train_shuffled[start_idx:end_idx]
        
        # Training step
        params, opt_state, train_loss = train_step(params, opt_state, X_train_batch, y_train_batch)
        train_losses.append(train_loss)
        
        # Test periodically
        if step % test_every == 0:
            # Get random test sample (1 sample)
            key, test_key = jr.split(key)
            test_idx = jr.choice(test_key, len(X_test), (test_batch_size,))
            X_test_sample = X_test[test_idx]
            y_test_sample = y_test[test_idx]
            
            # Test step
            test_pred, test_loss = val_step(params, X_test_sample, y_test_sample)
            test_losses.append(test_loss)
            test_predictions_list.append(test_pred[0])  # Store the single prediction
            test_ground_truth_list.append(y_test_sample[0])  # Store the single ground truth
            
            print(f"Epoch {epoch+1:2d}/{num_epochs}, Step {step:4d}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        step += 1

print("Training completed!")

# Convert collected predictions to arrays
test_predictions = jnp.array(test_predictions_list)  # (num_iterations, 2, neurons)
test_ground_truth = jnp.array(test_ground_truth_list)  # (num_iterations, 2, neurons)
final_test_loss = jnp.mean((test_predictions - test_ground_truth) ** 2)
print(f"Final test loss: {final_test_loss:.6f}")

# Create plots directory
plots_dir = 'transformer_plots'
os.makedirs(plots_dir, exist_ok=True)

# Plot training curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', alpha=0.7)
plt.plot(test_losses, label='Test Loss', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Training Progress (Train on 30 → Test on 1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(plots_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot predictions vs ground truth for a few samples from our test iterations
num_samples_to_plot = 3
timesteps = np.arange(future_steps)

for sample_idx in range(min(num_samples_to_plot, len(test_predictions))):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Transformer Sample {sample_idx+1} - Predictions vs Ground Truth')
    
    for i, ax in enumerate(axes.flat):
        if i < min(12, output_dim):  # Show up to 12 neurons
            # Ground truth and predictions for this neuron
            gt = test_ground_truth[sample_idx, :, i]
            pred = test_predictions[sample_idx, :, i]
            
            # Plot as lines
            ax.plot(timesteps, gt, 'k-', linewidth=3, 
                   label='Ground Truth', marker='s', markersize=8)
            ax.plot(timesteps, pred, 'r-', linewidth=2, 
                   label='Prediction', marker='o', markersize=6)
            
            ax.set_ylim(-0.1, max(7, np.max(pred) + 0.5))
            ax.set_xlabel('Future Timestep')
            ax.set_ylabel('Spike Count')
            ax.set_title(f'Neuron {i+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plot_filename = f'transformer_predictions_sample_{sample_idx+1}.png'
    plt.savefig(os.path.join(plots_dir, plot_filename), dpi=150, bbox_inches='tight')
    plt.close()

print(f"Plots saved to {plots_dir}/")

# Print some statistics
print(f"\nModel Statistics:")
print(f"Mean prediction: {np.mean(test_predictions):.4f}")
print(f"Mean ground truth (from test samples): {np.mean(test_ground_truth):.4f}")
print(f"Prediction std: {np.std(test_predictions):.4f}")
print(f"Ground truth std (from test samples): {np.std(test_ground_truth):.4f}")
print(f"Correlation: {np.corrcoef(test_predictions.flatten(), test_ground_truth.flatten())[0,1]:.4f}")
print(f"Number of test samples collected: {len(test_predictions)}")
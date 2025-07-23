import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stndt.get_data_S1 import data_loading_for_batch, load_s1_test, load_s1_train
from enhanced_spiking_transformer import EST
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
import matplotlib.pyplot as plt
from stndt.mask import create_forward_prediction_mask

#HYPERPARAMETERS
input_dim = 11
d_model = 64
num_heads = 8
num_layers = 4
dropout_rate = 0.1
# attention_dropout_rate = 0.1
num_epochs = 10
batch_size = 64
num_batches = 30000 // batch_size
sample_to_plot = jax.random.randint(key=jr.PRNGKey(3), shape=(), minval=0, maxval=3)



#load data
train_data_original = load_s1_train()
test_data = load_s1_test()
#process data

model = EST(
    input_dim=input_dim,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    output_dim=input_dim,  
    dropout_rate=dropout_rate,
    key=jr.PRNGKey(0)
)
optimizer = optax.adam(learning_rate=0.001)
optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

import jax.scipy as jcp

def loss_fn(model, inputs, targets, mask_labels, key):
    predictions = model(inputs, key=key, inference=False)
    
    # Only compute loss where mask_labels != -100
    valid_mask = mask_labels != -100
    predictions = jnp.maximum(predictions, 1e-8)
    
    losses = -jax.scipy.stats.poisson.logpmf(mask_labels, predictions)

    masked_losses = jnp.where(valid_mask, losses, 0.0)
    loss = jnp.sum(masked_losses) / jnp.sum(valid_mask)
    return loss


@eqx.filter_value_and_grad
def compute_loss_and_grads(model, inputs, targets, mask_labels, key):
    loss = loss_fn(model, inputs, targets, mask_labels, key)
    return loss

def training_step(model, optimizer, inputs, targets, mask_labels, key, optimizer_state):
    loss, grads = compute_loss_and_grads(model, inputs, targets, mask_labels, key)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    model = eqx.apply_updates(model, updates)
    return model, optimizer_state, loss

def validate_step(model, inputs, targets, mask_labels, key):
    predictions = model(inputs, key=key, inference=True)
    
    # Only compute loss where mask_labels != -100
    valid_mask = mask_labels != -100
    predictions = jnp.maximum(predictions, 1e-8)
    
    # Compute loss only on valid positions using jax.where
    losses = -jax.scipy.stats.poisson.logpmf(mask_labels, predictions)
    masked_losses = jnp.where(valid_mask, losses, 0.0)
    loss = jnp.sum(masked_losses) / jnp.sum(valid_mask)
    return loss, predictions

def shuffle_data(data_list, key):
    """Shuffle list of data samples (not JAX array)"""
    n_samples = len(data_list)
    perm = jr.permutation(key, n_samples)
    shuffled_list = [data_list[i] for i in perm]
    return shuffled_list

def plot_validation_samples(targets, predictions, epoch, num_samples_to_plot=3, plots_dir="validation_plots"):
    os.makedirs(plots_dir, exist_ok=True)
    timesteps = jnp.arange(targets.shape[1])
    for sample_idx in range(num_samples_to_plot):
        sample_to_plot = jax.random.randint(key=jr.PRNGKey(sample_idx), shape=(), minval=0, maxval=targets.shape[0])
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f"Sample {sample_to_plot+1} - Predictions vs Targets (Epoch {epoch+1})")
        for i, ax in enumerate(axes.flat):
            if i < min(12, targets.shape[2]):
                gt = targets[sample_to_plot, :, i]
                pred_continuous = predictions[sample_to_plot, :, i]

                ax.plot(timesteps, gt, 'k-', linewidth=3, label='Ground Truth', marker='s', markersize=5)
                ax.plot(timesteps, pred_continuous, 'r-', linewidth=2, label='Predictions', marker='o', markersize=3)

                ax.set_ylim(0, 6)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Spike Count')
                ax.set_title(f"Neuron {i+1}")
                ax.legend()
                ax.grid(True, alpha=0.3)

                ax.set_xticks(timesteps)
                ax.set_xticklabels([f'T+{t+1}' for t in timesteps])

        plt.tight_layout()
        plot_file_name = f'predictions_epoch_{epoch+1}_sample_{sample_idx+1}.png'
        plot_path = os.path.join(plots_dir, plot_file_name)
        plt.savefig(plot_path)
        plt.close(fig)


for i in range(num_epochs):
    shuffle_key = jr.PRNGKey(i)

    train_data = shuffle_data(train_data_original, shuffle_key)

    print(f"Epoch {i+1}/{num_epochs}")
    for batch_idx in range(num_batches):
        batch_key = jr.PRNGKey(i * 10000 + batch_idx)
        batch_data = data_loading_for_batch(train_data, batch_size, batch_idx)
        
        # Use masking approach like STNDT
        inputs, mask_labels = create_forward_prediction_mask(batch_data, num_forward_steps=2)
        
        model, optimizer_state, loss = training_step(
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            targets=batch_data,  # Original full data as targets
            mask_labels=mask_labels,
            key=batch_key,
            optimizer_state=optimizer_state
        )
        print(f"Batch {batch_idx + 1}, Loss: {loss:.4f}")
    # Run validation on first test batch and plot once per epoch
    test_batch_key = jr.PRNGKey(i + 10000)
    test_batch_data = data_loading_for_batch(test_data, batch_size, 0)
    
    # Use masking approach for validation too
    test_inputs, test_mask_labels = create_forward_prediction_mask(test_batch_data, num_forward_steps=2)
    
    test_loss, predictions = validate_step(
        model=model,
        inputs=test_inputs,
        targets=test_batch_data,
        mask_labels=test_mask_labels,
        key=test_batch_key
    )
    print(f"Epoch {i+1} Validation Loss: {test_loss:.4f}")

    predicted_timesteps = test_batch_data[:, -2:, :]  # Last 2 timesteps (ground truth)
    predicted_outputs = predictions[:, -2:, :]        # Last 2 timesteps (predictions)
    
    plot_validation_samples(
        targets=predicted_timesteps,    
        predictions=predicted_outputs,    
        epoch=i,
        num_samples_to_plot=3,
        plots_dir="EST_Plots"
    )






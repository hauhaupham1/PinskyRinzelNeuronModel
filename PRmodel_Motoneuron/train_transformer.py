"""
Training script for signature prediction transformer
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import time

from transformer import Transformer
from generate_training_data import generate_signature_sequence

jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

# Training hyperparameters
LEARNING_RATE = 3e-4
BATCH_SIZE = 16
EPOCHS = 100
SEQUENCE_LENGTH = 9
N_EMBED = 64
N_HEAD = 8
N_LAYER = 6
DROPOUT = 0.1

def signature_mse_loss(pred_signatures, target_signatures):
    """Simple MSE loss between predicted and target signatures."""
    return jnp.mean((pred_signatures - target_signatures) ** 2)

@eqx.filter_jit
def compute_loss(model, sequences, targets):
    """Compute loss for a batch."""
    # Forward pass through transformer (handles batches directly)
    predictions = model(sequences)  # (batch_size, seq_len, sig_dim)
    
    # Get the last time step prediction (next signature prediction)
    last_predictions = predictions[:, -1, :]  # (batch_size, signature_dim)
    
    # Compute MSE loss
    loss = signature_mse_loss(last_predictions, targets)
    return loss

@eqx.filter_jit
def train_step(model, optimizer, opt_state, sequences, targets):
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, sequences, targets)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def create_batches(sequences, targets, batch_size, key):
    n_samples = len(sequences)
    #randomization
    indices = jr.permutation(key, n_samples)
    
    batches = []

    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_indices = indices[i:end_idx]
        batch_sequences = sequences[batch_indices]
        batch_targets = targets[batch_indices]
        batches.append((batch_sequences, batch_targets))
    
    return batches

def calculate_signature_dim(num_neurons, signature_depth):
    d = num_neurons + 1
    return sum(d ** i for i in range(1, signature_depth + 1))

def train_transformer():
    print("=== Training Signature Prediction ===")
    
    # Initialize model
    key = jr.PRNGKey(42)
    model_key, data_key, train_key = jr.split(key, 3)
    num_neurons = 2
    signature_depth = 3
    
    # Get signature dimension from a small test
    signature_dim = calculate_signature_dim(num_neurons=num_neurons, signature_depth=signature_depth)
    
    # Initialize transformer
    model = Transformer(
        signature_dim=signature_dim,
        n_embed=N_EMBED,
        n_head=N_HEAD, 
        n_layer=N_LAYER,
        max_length=SEQUENCE_LENGTH,
        dropout=DROPOUT,
        key=model_key
    )
    
    # Generate training data
    print("Generating training data...")
    sequences, targets = generate_signature_sequence(
        num_simulations=500,
        num_neurons=num_neurons,
        signature_depth=signature_depth,
        window_size=10.0,
        total_duration=100.0,
        sequence_length=SEQUENCE_LENGTH,
        key=data_key
    )
    
    # Split train/validation
    n_train = int(0.8 * len(sequences))
    train_sequences, val_sequences = sequences[:n_train], sequences[n_train:]
    train_targets, val_targets = targets[:n_train], targets[n_train:]
    
    
    # Initialize optimizer
    optimizer = optax.adamw(LEARNING_RATE, weight_decay=1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        epoch_key = jr.fold_in(train_key, epoch)
        batch_key = jr.split(epoch_key)[0]
        
        # Create batches for this epoch
        batches = create_batches(train_sequences, train_targets, BATCH_SIZE, batch_key)
        
        # Training
        epoch_losses = []
        for batch_seq, batch_targets in batches:
            model, opt_state, loss = train_step(
                model, optimizer, opt_state, batch_seq, batch_targets
            )
            epoch_losses.append(loss)
        
        avg_train_loss = jnp.mean(jnp.array(epoch_losses))
        
        # Validation
        val_loss = compute_loss(model, val_sequences, val_targets)
        
        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
        
        epoch_time = time.time() - epoch_start
        
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, time={epoch_time:.2f}s")
    
    
    return best_model

if __name__ == "__main__":
    model = train_transformer()
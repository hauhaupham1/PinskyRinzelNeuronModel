import optax
import equinox as eqx
from stndt_jax.loss_function.losses import poisson_nll_loss, bits_per_spike_jax
from stndt_jax.model.stnd_transformer import STNDT
from stndt_jax.train.utils.mask import UNMASKED_LABEL
from stndt_jax.train.utils.train_data_loader import (
    load_and_prepare_data,
    get_batch,
    apply_masking_for_training,
    apply_masking_for_training_contrast,
)
import jax.numpy as jnp
import jax.random as jrandom
import os

from stndt_jax.utils.visualization import visualize_predictions

UNMASKED_LABEL

# HYPERPARAMETER
NUM_NEURONS = 182
NUM_HEADS = 2
NUM_LAYERS = 4
SIMULATION_LENGTH = 700
BIN_SIZE = 5
NUM_TIME_BINS = int(SIMULATION_LENGTH / BIN_SIZE)
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_UPDATES = 50501
SEED = 290803
MASK_SPAN_RAMP_START = 8000
MASK_SPAN_RAMP_END = 12000
CONTRAST_MASK_SPAN_RAMP_START = 8000
CONTRAST_MASK_SPAN_RAMP_END = 12000
config = {
    "MASK_RATIO": 0.25,
    "MASK_MODE": "full",
    "MASK_TOKEN_RATIO": 0.8,
    "MASK_RANDOM_RATIO": 0.1,
    "USE_ZERO_MASK": True,
    "MASK_MAX_SPAN": 1,
    "MASK_SPAN_PROB": 0.0,
    "CONTEXT_FORWARD": -1,
    "CONTEXT_BACKWARD": -1,
    "LEARNABLE_POSITION": False,
    "POSITION": {"OFFSET": False},
    "PRE_NORM": True,
    "FIXUP_INIT": True,
    "EMBED_DIM": 1,
    "LOGRATE": True,
    "NUM_LAYERS": 4,
    "NUM_HEADS": 2,
    "HIDDEN_SIZE": 128,  # Missing parameter - default from original STNDT
    "DROPOUT": 0.1,  # General dropout rate - used in attention and feed-forward layers
    "DROPOUT_RATES": 0.2,  # Specific dropout for rate predictions
    "DROPOUT_EMBEDDING": 0.2,  # Dropout for embeddings
    "TEMPERATURE": 0.07,  # Temperature for InfoNCE loss
    "LINEAR_EMBEDDER": True,
    "USE_CONTRAST_PROJECTOR": False,
    "CONTRAST_LAYER": "embedder",
    "LAMBDA": 1e-1,
    "LINEAR_PROJECTOR": True,
    "LOSS":{"TYPE" : "poisson"},
}

contrast_config = {
    "CONTRAST_MASK_MODE": "full",
    "CONTRAST_MASK_RATIO": 0.05,
    "CONTRAST_MASK_TOKEN_RATIO": 0.5,
    "CONTRAST_MASK_RANDOM_RATIO": 0.5,
    "USE_ZERO_MASK": True,
    "CONTRAST_MASK_MAX_SPAN": 1,
    "CONTRAST_MASK_SPAN_PROB": 0.0,
}
# DATA LOADING
nwb_path = "../data_loading/data/000128/sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"
data = load_and_prepare_data(nwb_path)


def create_optimizer(
    learning_rate=1e-4, warmup_steps=5000, weight_decay=5e-5, max_grad_norm=200.0
):
    """Create optimizer with warmup, weight decay, and gradient
    clipping."""

    # Learning rate schedule with warmup
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,  # Start from 0
        peak_value=learning_rate,  # Target learning rate
        warmup_steps=warmup_steps,
        decay_steps=50501,  # Total training steps
        end_value=0.0,  # Final learning rate
    )

    # Chain optimizer components
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),  # Gradient clipping
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
        ),
    )

    return optimizer


def get_span_probability(step, ramp_start=8000, ramp_end=12000):
    """Get probability of using span masking vs single position."""
    if step < ramp_start:
        return 0.0  # No span masking
    elif step >= ramp_end:
        return 1.0  # Always use span masking
    else:
        # Gradual increase in span probability
        return (step - ramp_start) / (ramp_end - ramp_start)


@eqx.filter_jit
def train_step(model, optimizer_state, batch, opt, key, config, contrast_config):
    """Single training step"""
    masked_inputs, labels = apply_masking_for_training(batch, data, config)
    contrast1, _ = apply_masking_for_training_contrast(batch, data, contrast_config)
    contrast2, _ = apply_masking_for_training_contrast(batch, data, contrast_config)
    
    # Create a loss function that only depends on trainable parameters
    def loss_fn(params):
        # Combine params with static model structure
        model_with_params = eqx.combine(params, eqx.filter(model, lambda x: not eqx.is_array(x)))
        return model_with_params.forward(masked_inputs, labels, contrast1, contrast2, False, key)
    
    # Extract trainable parameters
    params = eqx.filter(model, eqx.is_array)
    
    # Compute gradients with respect to params only
    (loss, aux_data), grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)(params)
    
    # Unpack auxiliary data
    decoder_loss, contrast_loss, decoder_rate = aux_data
    
    # Update parameters
    updates, optimizer_state = opt.update(grad, optimizer_state, params)
    model = eqx.apply_updates(model, updates)
    return model, optimizer_state, loss


@eqx.filter_jit
def eval_step(model: STNDT, batch, config):
    masked_inputs, labels = apply_masking_for_training(batch, data, config)

    loss, decoder_loss, contrast_loss, decoder_rates, _, _ = model.forward(
        src=masked_inputs,
        mask_labels=labels,
        contrast_src1=None,
        contrast_src2=None,
        val_phase=True,
        key=jrandom.PRNGKey(1),
    )

    # Model loss is already Poisson NLL for valid positions
    # We need the null model NLL for the same positions
    valid_mask = jnp.where(labels != UNMASKED_LABEL, 1.0, 0.0)

    # Create null model (mean firing rate per neuron)
    null_rates = jnp.tile(
        jnp.nanmean(batch, axis=(0, 1), keepdims=True),
        (batch.shape[0], batch.shape[1], 1),
    )
    # Calculate null model loss only for the same masked positions
    null_loss = poisson_nll_loss(null_rates, labels, log_input=False)
    null_loss_masked = jnp.where(valid_mask, null_loss, 0.0)

    # Sum the losses and total spikes for valid positions only
    nll_model = jnp.sum(loss)  # Already filtered by model
    nll_null = jnp.sum(null_loss_masked)
    total_spikes = jnp.sum(jnp.where(valid_mask, batch, 0.0))
    # Bits per spike
    bps = (nll_null - nll_model) / total_spikes / jnp.log(2)
    return bps, decoder_rates


def visualize_batch(
    predictions, batch, step, save_dir="mc_maze", predictions_are_log_rates=False
):
    """Visualize predictions for a batch"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Save visualizations
    save_prefix = f"{save_dir}/step_{step}"
    if predictions_are_log_rates:
        predictions = jnp.exp(predictions)
    # Create heatmap visualization
    visualize_predictions(
        batch, predictions, sample_idx=0, save_path=f"{save_prefix}_heatmap.png"
    )

    print(f"  Visualizations saved to {save_dir}/")


if __name__ == "__main__":
    # Initialize Model
    key = jrandom.PRNGKey(29080301)
    keys = jrandom.split(key, 4)
    model = STNDT(
        config=config,
        trial_length=NUM_TIME_BINS,
        num_neurons=NUM_NEURONS,
        max_spikes=3,
        key=keys[3],
    )

    # Get a training batch

    optimizer = create_optimizer(
        learning_rate=1e-4,
        warmup_steps=5000,  # From MC Maze config
        weight_decay=5e-5,  # From MC Maze config
        max_grad_norm=200.0,  # From STNDT default config
    )
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for i in range(NUM_UPDATES):
        span_prob = get_span_probability(i)
        dynamic_config = config.copy()
        dynamic_config["MASK_SPAN_PROB"] = span_prob

        dynamic_contrast_config = contrast_config.copy()
        dynamic_contrast_config["MASK_SPAN_PROB"] = span_prob
        batch_data = get_batch(
            data, batch_size=BATCH_SIZE, split="train", key=jrandom.PRNGKey(i + 290803)
        )
        # TRAIN
        model, optimizer_state, loss = train_step(
            model,
            optimizer_state,
            batch_data,
            optimizer,
            keys[1],
            config=dynamic_config,
            contrast_config=dynamic_contrast_config,
        )
        if i % 10 == 0:
            print(f"  Step {i} loss: {loss}")

        # VALIDATE every 20 epochs
        if i % 20 == 0:
            val_key = jrandom.PRNGKey(i)
            val_data = get_batch(data, batch_size=BATCH_SIZE, split="val", key=val_key)
            bps, decoder_rates = eval_step(model, val_data, dynamic_config)

            
            visualize_batch(decoder_rates, val_data, i, predictions_are_log_rates=config["LOGRATE"])
            print(f"  Val bps: {bps}")

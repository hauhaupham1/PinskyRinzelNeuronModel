import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PRmodel_Motoneuron.Network import MotoneuronNetwork
import jax
import jax.numpy as jnp
import jax.random as jr
from stnd_transformer import STNDT
import optax
from mask import JAXMasker
import equinox as eqx



#data generation from Motoneuron model
num_neurons = 10
num_samples = 500

def input_current(t):
    current = jnp.zeros(num_neurons)
    current = current.at[0:5].set(jnp.where((t >= 10) & (t < 20), 5.0, 0.0))
    return current

def simulation_data(num_neurons, duration=100.0, dt=0.1):
    # Create a MotoneuronNetwork instance
    network = MotoneuronNetwork(num_neurons=num_neurons,
                                threshold=-37,
                                diffusion=True,)
    sol = network(input_current=input_current,
                    t0=0.0, 
                    t1=duration,
                    max_spikes=100,
                    num_samples=num_samples,
                    key = jr.PRNGKey(0),
                    max_steps=int(duration / dt),
                    memory_efficient=True,
                    spike_only=True,
                    dt0=dt,
                  )
    return sol

#create data for transformer
def create_data(sol, duration = 100.0, bin_size=1.0):
    bins = jnp.arange(0, duration + bin_size, bin_size)
    num_time_bins = len(bins) - 1
    spike_counts = jnp.zeros((num_samples, num_time_bins, num_neurons), dtype=jnp.int32)
    for sample_idx in range(num_samples):
        spike_times = sol.spike_times[sample_idx]
        spike_marks = sol.spike_marks[sample_idx]
        for neuron_idx in range(num_neurons):
            neuron_spike_mask = spike_marks[:, neuron_idx]
            neuron_spike_times = spike_times[neuron_spike_mask]
            valid_mask = jnp.isfinite(neuron_spike_times) 
            valid_spike_times = neuron_spike_times[valid_mask]

            if len(valid_spike_times) > 0:
                # Count spikes in each bin
                counts, _ = jnp.histogram(valid_spike_times, bins=bins)
                # Convert to int32 to match spike_counts dtype
                counts = counts.astype(jnp.int32)
                spike_counts = spike_counts.at[sample_idx, :, neuron_idx].set(counts)

    return spike_counts


#generate config
def create_config():
    return{
        'NUM_LAYERS': 2,
        'NUM_HEADS': 4,
        'HIDDEN_SIZE'   : 64,
        'DROPOUT'  : 0.1,

        'TRIAL_LENGTH': 100,
        'NUM_NEURONS': 10,

        'MASK_MODE': 'timestep',
        'MASK_RATIO': 0.15,
        'MASK_MAX_SPAN': 3,
        
        # Contrastive masking config
        'CONTRAST_MASK_MODE': 'timestep',
        'CONTRAST_MASK_RATIO': 0.2,
        'CONTRAST_MASK_MAX_SPAN': 3,
        'USE_CONTRAST_PROJECTOR': True,
        'LINEAR_PROJECTOR': False,
        'CONTRAST_LAYER': 'decoder',
        'TEMPERATURE': 0.5,
        'LAMBDA': 1.0,

        #TRAINING PARAMETERS
        'BATCH_SIZE': 32,
        'NUM_UPDATES': 1000,
        'LR' :{
            'INIT': 1e-4,
            'SCHEDULE': True,
            'SCHEDULER': 'cosine',
            'WARMUP': 100,
        },
        'MAX_GRAD_NORM': 200.0,
        'LOSS': {
            'TYPE': 'poisson',
        },
        'LOGRATE': False,
        
        # Context masking settings
        'FULL_CONTEXT': False,          # If True, no context masking (attend to all positions)
        'CONTEXT_FORWARD': 0,          # How many positions forward to attend (-1 = unlimited)
        'CONTEXT_BACKWARD': -1,         # How many positions backward to attend
        'CONTEXT_WRAP_INITIAL': False   # Whether to wrap initial context
    }

@eqx.filter_jit
def train_step_filtered(params, static, optimizer, opt_state, batch, masker, key):
    """Single training step with parameter filtering"""
    
    def loss_fn(params):
        # Combine params and static parts to reconstruct model
        model = eqx.combine(params, static)
        
        key1, key2, key3 = jr.split(key, 3)
        masked_batch, mask_labels, _, _, _, _ = masker.mask_batch(batch=batch, key=key1)
        
        # Create contrastive views
        from mask import create_contrastive_masks
        contrast_src1, _, contrast_src2, _ = create_contrastive_masks(batch, model.config, key=key2)

        outputs = model(src=masked_batch, mask_labels=mask_labels, 
                       contrast_src1=contrast_src1, contrast_src2=contrast_src2, 
                       val_phase=False, key=key3)

        loss, decoder_loss, contrast_loss, _ = outputs

        return loss.mean(), (decoder_loss.mean(), contrast_loss.mean())
    
    # Compute gradients only for trainable parameters
    (loss, (decoder_loss, contrast_loss)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(params)
    
    # Update only trainable parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)   
    
    return params, opt_state, loss, decoder_loss, contrast_loss

@eqx.filter_jit
def validation_step(model: STNDT, batch: jnp.ndarray, masker: JAXMasker, key):
    """Performs one validation step."""
    key1, key2 = jr.split(key)
    masked_batch, mask_labels, _, _, _, _ = masker.mask_batch(batch=batch, key=key1)
    outputs = model(src=masked_batch, mask_labels=mask_labels, val_phase=True, key=key2)
    per_sample_loss = outputs[0] 
    return per_sample_loss.mean()

def main():

    sol = simulation_data(num_neurons=num_neurons, duration=100.0)

    spike_data = create_data(sol, duration=100.0, bin_size=1.0)
    spike_data = jnp.clip(spike_data, 0, int(jnp.max(spike_data)))

    train_size = int(0.8 * num_samples)
    train_data = spike_data[:train_size]
    val_data = spike_data[train_size:]


    config = create_config()
    key = jr.PRNGKey(123456)
    key, model_key = jr.split(key)

    model = STNDT(config=config, 
                  trial_length=config['TRIAL_LENGTH'],
                  num_neurons=config['NUM_NEURONS'],
                  max_spikes=int(jnp.max(spike_data)),
                  key=model_key)
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config['LR']['INIT'],
        warmup_steps=config['LR']['WARMUP'],
        decay_steps=config['NUM_UPDATES']
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(config['MAX_GRAD_NORM']),  # Gradient clipping
        optax.adam(learning_rate=schedule, eps=1e-8)  # Adam with warmup schedule
    )
    # Partition model into trainable and static parts
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    masker = JAXMasker(config)
    batch_size = config['BATCH_SIZE']



    for step in range(config['NUM_UPDATES']):
        key, batch_key = jr.split(key)
        batch_idx = jr.choice(batch_key, train_size, (batch_size,))
        batch = train_data[batch_idx]
        key, train_key= jr.split(key)
        params, opt_state, loss, decoder_loss, contrast_loss = train_step_filtered(params, static, optimizer, opt_state, batch, masker, train_key)

        if step % 10 == 0:
            print(f"Step {step}, Train Loss: {loss:.4f}, Decoder: {decoder_loss:.4f}, Contrast: {contrast_loss:.4f}")
        
        # Validation every 50 steps
        if step % 50 == 0 and step > 0:
            key, val_key = jr.split(key)
            val_batch_idx = jr.choice(val_key, len(val_data), (min(batch_size, len(val_data)),), replace=True)
            val_batch = val_data[val_batch_idx]
            
            # Run validation (no gradient computation)
            model = eqx.combine(params, static)  # Reconstruct model for validation
            
            val_loss = validation_step(model, val_batch, masker, val_key)
            
            print(f"         >>> VALIDATION: Loss={val_loss:.4f} <<<")
            print()

    # Reconstruct final model
    final_model = eqx.combine(params, static)
    return final_model


if __name__ == "__main__":
    trained_model = main()
    print("Training complete.")
    # Save the model or perform further evaluation as needed.
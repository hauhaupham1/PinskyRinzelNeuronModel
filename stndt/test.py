from stnd_transformer import STNDT
import jax.random as jr
import jax.numpy as jnp


#testing out the STNDT class



def config():
    return{
        'EMBED_DIM': 1,
        'LINEAR_EMBEDDER': False,
        'USE_CONTRAST_PROJECTOR': False,
        'LINEAR_PROJECTOR' : False,
        'DROPOUT_RATES': 0.1,
        'SCALE_NORM': True,
        'NUM_LAYERS': 3,
        'DECODER': {
            'LAYERS': 1,
        },
        'LOGRATE': False,
        'NUM_HEADS': 2,
        'DROPOUT': 0.1,
        'HIDDEN_SIZE': 16,
        'PRE_NORM': False,  # Add this to match PyTorch
        'MAX_SPIKES': 5,  # Set max spikes to match data range
    }


key = jr.PRNGKey(0)
trial_length = 10  # Match the data time dimension
stndt = STNDT(config=config(), trial_length=trial_length, num_neurons=10, key=key)  # max_spikes=5 to match data range 0-5



#data format B x T x N - discrete spike counts
# Generate realistic discrete spike counts (0-5 spikes per bin)
key = jr.PRNGKey(42)
data = jr.randint(key, (2, trial_length, 10), 0, 2)  # Random integers 0-5 representing spike counts

print("Data shape:", data.shape)
print("Data range:", data.min(), "to", data.max())
print("Sample data:", data)  # Show first 5 time steps, 5 neurons of first batch

result = stndt(data, training=False)

print(result.shape)  # Should print the shape of the output tensor
print(result[:, -5:, :])  # Print the output tensor to verify the result



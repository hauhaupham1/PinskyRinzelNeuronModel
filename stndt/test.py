from stnd_transformer import STNDT
import jax.random as jr
import jax.numpy as jnp


#testing out the STNDT class



def config():
    return{
        'EMBED_DIM': 0,
        'LINEAR_EMBEDDER': True,
        'USE_CONTRAST_PROJECTOR': True,
        'LINEAR_PROJECTOR' : True,
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
        'LOGRATE': True, 
    }


key = jr.PRNGKey(0)
stndt = STNDT(config=config(), trial_length=100, num_neurons=10, max_spikes=5, key=key)  # max_spikes=5 to match data range 0-5



#data format B x T x N - discrete spike counts
# Generate realistic discrete spike counts (0-5 spikes per bin)
key = jr.PRNGKey(42)
data = jr.randint(key, (2, 100, 10), 0, 6)  # Random integers 0-5 representing spike counts

print("Data shape:", data.shape)
print("Data range:", data.min(), "to", data.max())
print("Sample data:", data[0, :5, :5])  # Show first 5 time steps, 5 neurons of first batch

result = stndt(data)

print(result.shape)  # Should print the shape of the output tensor
print(result)  # Print the output tensor to verify the result
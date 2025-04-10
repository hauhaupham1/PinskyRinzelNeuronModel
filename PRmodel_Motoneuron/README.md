# Motoneuron Model

This module extends the Pinsky-Rinzel model implementation to incorporate parameters specific to motoneurons as defined in `motoneuron.yaml`. The implementation uses JAX and Diffrax for efficient and differentiable simulation.

## Features

- Class-based implementation that extends the original Pinsky-Rinzel model
- Parameter mapping from motoneuron.yaml configuration file
- Support for different parameter sets ('20230210_84', '20230210_8', 'test0', 'p0')
- Separate leak conductances for somatic and dendritic compartments
- Enhanced calcium dynamics based on motoneuron parameters
- Analysis tools for studying firing properties (F-I curves, spike adaptation, etc.)

## Usage

### Basic Simulation

```python
from PRmodel_Motoneuron import MotoneuronModel

# Create a model with parameters from motoneuron.yaml
model = MotoneuronModel(yaml_file="path/to/motoneuron.yaml", param_set="20230210_84")

# Run a simulation
sol = model.solve(
    t_dur=2000,       # 2000 ms simulation
    I_stim=0.05,      # 0.05 nA stimulus
    stim_start=500,   # Start at 500 ms
    stim_end=1500     # End at 1500 ms
)

# Analyze the response
results = model.analyze_response(sol)
print(f"Number of spikes: {len(results['spike_times'])}")
print(f"Mean firing rate: {results['firing_rate']:.2f} Hz")
```

### Running Examples

The module includes several example scripts to demonstrate its usage:

1. `run_motoneuron.py` - Basic simulation and F-I curve analysis
2. `compare_models.py` - Compare the original Pinsky-Rinzel model with the motoneuron model

To run these examples, navigate to the PRmodel_Motoneuron directory and execute:

```bash
python run_motoneuron.py
python compare_models.py
```

## Parameter Sets

The motoneuron.yaml file contains several parameter sets:

- `20230210_84` - Default parameter set
- `20230210_8` - Alternative parameter set
- `test0` - Test parameter set
- `p0` - Initial parameter set

Each parameter set defines different conductances and membrane properties for the motoneuron model.

## Extending the Model

The MotoneuronModel class can be extended to incorporate additional features specific to motoneurons:

```python
class ExtendedMotoneuronModel(MotoneuronModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add additional parameters or customizations
        
    def vector_field(self, t, y, args):
        # Customize the vector field with additional dynamics
        dy = super().vector_field(t, y, args)
        # Modify or add to the dynamics
        return dy
```

## Dependencies

- JAX and Diffrax for differential equation solving
- PyYAML for parsing the motoneuron.yaml file
- Matplotlib for plotting results
- NumPy for array operations

## References

This implementation is based on:
1. The Pinsky-Rinzel model (Pinsky and Rinzel 1994)
2. An electrodiffusive, ion conserving Pinsky-Rinzel model with homeostatic mechanisms (Sætra et al. 2020)
3. Functional properties of motoneurons derived from mouse embryonic stem cells (as referenced in motoneuron.yaml)

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PRmodel_Diffrax.PinskyRinzel import PinskyRinzel
from PRmodel_Motoneuron import MotoneuronModel

def compare_with_defaults(yaml_file, param_set='20230210_84'):
    """Compare a motoneuron model with custom parameters to one with default parameters."""
    # Create motoneuron model with parameters from YAML
    motoneuron_model = MotoneuronModel(yaml_file=yaml_file, param_set=param_set)
    
    # Create a default motoneuron model (no yaml file, use PR defaults)
    default_model = MotoneuronModel()
    
    # Print parameters for comparison
    print("\nComparison with Default Parameters:")
    print("-" * 50)
    print(f"{'Parameter':<20} {'Default':<15} {'Motoneuron':<15}")
    print("-" * 50)
    
    # Get motoneuron parameters
    motoneuron_params = motoneuron_model.params
    default_params = default_model.params
    
    # Compare key parameters
    params_to_compare = [
        ('g_c', default_model.g_c, motoneuron_params['g_c']),
        ('C_m', default_model.C_m, motoneuron_params['C_m']),
        ('p', default_model.p, motoneuron_params['p']),
        ('g_L (soma)', default_model.g_L, motoneuron_params['g_L_soma']),
        ('g_L (dend)', default_model.g_L, motoneuron_params['g_L_dend']),
        ('g_Na', default_model.g_Na, motoneuron_params['g_Na']),
        ('g_DR', default_model.g_DR, motoneuron_params['g_DR']),
        ('g_Ca', default_model.g_Ca, motoneuron_params['g_Ca']),
        ('g_AHP', default_model.g_AHP, motoneuron_params['g_AHP']),
        ('g_C', default_model.g_C, motoneuron_params['g_C']),
        ('E_L', default_model.E_L, motoneuron_params['E_L']),
        ('E_Na', default_model.E_Na, motoneuron_params['E_Na']),
        ('E_K', default_model.E_K, motoneuron_params['E_K']),
        ('E_Ca', default_model.E_Ca, motoneuron_params['E_Ca']),
    ]
    
    for name, def_val, moto_val in params_to_compare:
        print(f"{name:<20} {def_val:<15.6f} {moto_val:<15.6f}")
    
    # Simulation parameters
    t_dur = 1000  # ms
    I_stim = 0.5  # nA
    stim_start = 200  # ms
    stim_end = 400  # ms
    
    # Run simulations
    default_sol = default_model.solve(
        t_dur=t_dur,
        I_stim=I_stim,
        stim_start=stim_start,
        stim_end=stim_end
    )
    
    motoneuron_sol = motoneuron_model.solve(
        t_dur=t_dur,
        I_stim=I_stim,
        stim_start=stim_start,
        stim_end=stim_end
    )
    
    # Compare spike counts and timing
    default_spikes = default_model.get_spike_times(default_sol)
    motoneuron_spikes = motoneuron_model.get_spike_times(motoneuron_sol)
    
    print("\nSpiking Behavior:")
    print(f"Default model: {len(default_spikes)} spikes")
    print(f"Motoneuron model with {param_set}: {len(motoneuron_spikes)} spikes")
    
    # Plot results - both soma and dendrite on the same graph for each model
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Default model with soma and dendrite on same graph
    plt.subplot(2, 1, 1)
    plt.plot(default_sol.ts, default_sol.ys[:, 0], 'g-', label='Default (Soma)')
    plt.plot(default_sol.ts, default_sol.ys[:, 1], 'g--', label='Default (Dendrite)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Default Parameters (Pinsky-Rinzel)')
    plt.axvspan(stim_start, stim_end, alpha=0.2, color='gray', label='Stimulus')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Motoneuron model with soma and dendrite on same graph
    plt.subplot(2, 1, 2)
    plt.plot(motoneuron_sol.ts, motoneuron_sol.ys[:, 0], 'r-', label='Motoneuron (Soma)')
    plt.plot(motoneuron_sol.ts, motoneuron_sol.ys[:, 1], 'r--', label='Motoneuron (Dendrite)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title(f'Motoneuron Model with {param_set} Parameters')
    plt.axvspan(stim_start, stim_end, alpha=0.2, color='gray', label='Stimulus')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle('Comparison of Default vs Motoneuron Parameters', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'default_vs_motoneuron_{param_set}.png')
    
    # Create a figure comparing compartments
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Compare soma traces
    plt.subplot(2, 1, 1)
    plt.plot(default_sol.ts, default_sol.ys[:, 0], 'g-', label='Default Parameters')
    plt.plot(motoneuron_sol.ts, motoneuron_sol.ys[:, 0], 'r-', label=f'Motoneuron {param_set}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Comparison of Somatic Compartments')
    plt.axvspan(stim_start, stim_end, alpha=0.2, color='gray')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Compare dendrite traces
    plt.subplot(2, 1, 2)
    plt.plot(default_sol.ts, default_sol.ys[:, 1], 'g-', label='Default Parameters')
    plt.plot(motoneuron_sol.ts, motoneuron_sol.ys[:, 1], 'r-', label=f'Motoneuron {param_set}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Comparison of Dendritic Compartments')
    plt.axvspan(stim_start, stim_end, alpha=0.2, color='gray')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle('Default vs Motoneuron Parameters: Compartment-wise Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'default_vs_motoneuron_compartments_{param_set}.png')
    
    plt.show()
    
    return default_model, motoneuron_model, default_sol, motoneuron_sol
    

if __name__ == "__main__":
    # Path to the YAML file
    yaml_file = "PRmodel_Motoneuron/motoneuron.yaml"
    
    # Compare default parameters with each parameter set
    param_sets = ['20230210_84', '20230210_8', 'test0', 'p0']
    
    for param_set in param_sets:
        print(f"\n\nComparing default parameters with parameter set: {param_set}")
        default_model, motoneuron_model, default_sol, motoneuron_sol = compare_with_defaults(yaml_file, param_set)
    
    print("\nComplete!")

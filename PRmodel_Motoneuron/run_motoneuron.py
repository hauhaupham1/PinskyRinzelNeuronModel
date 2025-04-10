import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

# Add path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PRmodel_Motoneuron import MotoneuronModel


def run_f_i_curve(yaml_file, param_set='20230210_84'):
    """
    Run a series of simulations with different current intensities
    to generate an F-I curve.
    
    Args:
        yaml_file: Path to the YAML file
        param_set: Which parameter set to use
        
    Returns:
        Currents and corresponding firing rates
    """
    # Get currents from the YAML file
    import yaml
    with open(yaml_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # Use currents from the f_I target in the YAML file
    if 'Targets' in yaml_data and 'f_I' in yaml_data['Targets']:
        currents = yaml_data['Targets']['f_I']['I']
        i_factor = yaml_data['Targets']['f_I'].get('I_factor', 1.0)
        expected_rates = yaml_data['Targets']['f_I'].get('mean', [])
    else:
        # Default currents if not found in YAML
        currents = [20, 30, 40, 50, 60, 70, 80]
        i_factor = 1.0e-3
        expected_rates = []
    
    # Adjust currents by the factor
    currents = [i * i_factor for i in currents]
    
    # Create the model
    model = MotoneuronModel(yaml_file=yaml_file, param_set=param_set)
    
    # Simulation parameters
    t_dur = 2000  # ms
    stim_start = 500  # ms
    stim_end = 1500  # ms
    
    # Run simulations for each current
    firing_rates = []
    
    for i_stim in currents:
        # Simulate
        sol = model.solve(
            t_dur=t_dur,
            I_stim=i_stim,
            stim_start=stim_start,
            stim_end=stim_end
        )
        
        # Get spike times
        spike_times = model.get_spike_times(sol, threshold=-37.0)
        
        # Count spikes during the stimulus period
        stim_spikes = spike_times[(spike_times >= stim_start) & (spike_times <= stim_end)]
        
        # Calculate firing rate in Hz
        if len(stim_spikes) > 1:
            # Duration in seconds
            stim_duration_s = (stim_end - stim_start) / 1000.0
            rate = len(stim_spikes) / stim_duration_s
        else:
            rate = 0.0
            
        firing_rates.append(rate)
    
    return currents, firing_rates, expected_rates


def run_spike_adaptation(yaml_file, param_set='20230210_84'):
    """
    Run a simulation to analyze spike adaptation.
    
    Args:
        yaml_file: Path to the YAML file
        param_set: Which parameter set to use
    """
    # Create the model
    model = MotoneuronModel(yaml_file=yaml_file, param_set=param_set)
    
    # Simulation with a constant current that produces multiple spikes
    sol = model.solve(
        t_dur=2000,
        I_stim=0.06,  # Adjust as needed to get multiple spikes
        stim_start=500,
        stim_end=1500
    )
    
    # Get spike times
    spike_times = model.get_spike_times(sol, threshold=-37.0)
    
    # Calculate instantaneous frequencies
    if len(spike_times) > 1:
        isis = np.diff(spike_times)
        inst_freqs = 1000.0 / isis  # Hz
        
        # Calculate adaptation index (first ISI / last ISI)
        if len(isis) > 1:
            adaptation_index = isis[0] / isis[-1]
            print(f"Spike adaptation index: {adaptation_index:.2f}")
    
    return sol, spike_times


if __name__ == "__main__":
    # Path to the YAML file
    yaml_file = "PRmodel_Motoneuron/motoneuron.yaml"
    
    # Run the F-I curve analysis
    currents, firing_rates, expected_rates = run_f_i_curve(yaml_file)
    
    # Plot the F-I curve
    plt.figure(figsize=(10, 6))
    plt.plot(currents, firing_rates, 'o-', label='Model')
    
    if expected_rates:
        plt.plot(currents, expected_rates, 'x--', label='Target')
        
    plt.xlabel('Current (nA)')
    plt.ylabel('Firing Rate (Hz)')
    plt.title('F-I Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig('motoneuron_f_i_curve.png')
    
    # Run and plot a sample simulation
    print("Running simulation for spike adaptation analysis...")
    sol, spike_times = run_spike_adaptation(yaml_file)
    
    # Plot the voltage trace with spike times marked
    plt.figure(figsize=(12, 6))
    plt.plot(sol.ts, sol.ys[:, 0], label='Vs (Soma)')
    plt.plot(sol.ts, sol.ys[:, 1], label='Vd (Dendrite)')
    
    if len(spike_times) > 0:
        plt.plot(spike_times, [-30] * len(spike_times), 'r|', markersize=10, label='Spikes')
        
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Motoneuron Spike Adaptation')
    plt.grid(True)
    plt.legend()
    plt.savefig('motoneuron_adaptation.png')
    
    # Show all plots
    plt.show()
    
    print("Complete!")

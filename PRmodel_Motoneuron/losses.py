import jax
import jax.numpy as jnp
from PRmodel_Motoneuron.MotoneuronModel import MotoneuronModel
import yaml
import optax

def load_target_values(yaml_file="PRmodel_Motoneuron/motoneuron.yaml"):
    """Load target values from YAML file"""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data['Targets']

def range_loss(value, lower, upper):
    """Compute loss for a value that should be within a range"""
    # No loss if within range, quadratic penalty otherwise
    return jnp.where(value < lower, 
                     ((value - lower) / lower) ** 2,
                     jnp.where(value > upper, 
                              ((value - upper) / upper) ** 2, 
                              0.0))

def squared_error(value, target):
    """Simple squared error normalized by target value"""
    return ((value - target) / (jnp.abs(target) + 1e-6)) ** 2

def calculate_rin(sol, model, I_amp=-0.1, t_start=250, t_end=1250):
    """Calculate input resistance (Rin) from voltage response to hyperpolarizing current"""
    # Extract voltage before and during stimulus
    times = sol.ts
    v_soma = sol.ys[:, 0]
    
    # Find indices corresponding to times
    idx_before = jnp.argmin(jnp.abs(times - (t_start - 50)))
    idx_during = jnp.argmin(jnp.abs(times - (t_start + (t_end - t_start)/2)))
    
    # Calculate voltage change
    v_baseline = v_soma[idx_before]
    v_during = v_soma[idx_during]
    delta_v = v_during - v_baseline
    
    # Calculate Rin in MOhm (I_amp in nA)
    rin = delta_v / I_amp
    return rin

def calculate_tau(sol, model, I_amp=-0.1, t_start=250, t_end=1250):
    """Calculate membrane time constant from voltage response to hyperpolarizing current"""
    # This is a simplified calculation - real tau calculation needs exponential fitting
    times = sol.ts
    v_soma = sol.ys[:, 0]
    
    # Find relevant indices
    idx_stim_start = jnp.argmin(jnp.abs(times - t_start))
    idx_after_start = jnp.argmin(jnp.abs(times - (t_start + 50)))
    
    # Get voltage values
    v_baseline = v_soma[idx_stim_start-10:idx_stim_start].mean()
    v_stim = v_soma[idx_stim_start:idx_after_start]
    t_stim = times[idx_stim_start:idx_after_start] - times[idx_stim_start]
    
    # Normalize voltage
    v_norm = (v_stim - v_stim[-1]) / (v_baseline - v_stim[-1])
    v_norm = jnp.log(jnp.maximum(v_norm, 1e-6))  # Avoid log(0)
    
    # Simple linear fit to extract tau
    # tau is the negative reciprocal of the slope
    slope = jnp.sum(v_norm * t_stim) / jnp.sum(t_stim ** 2)
    tau = -1.0 / slope
    return tau

def calculate_firing_rate(sol, model, t_start=500, t_end=1500, threshold=-37.0):
    """Calculate firing rate from spike times"""
    # Get spike times using the model's method
    spike_times = model.get_spike_times(sol, threshold=threshold)
    
    # Filter spikes within the analysis window
    mask = (spike_times >= t_start) & (spike_times <= t_end)
    spikes_in_window = spike_times[mask]
    
    # Calculate firing rate in Hz (convert from ms to s)
    if len(spikes_in_window) > 1:
        return len(spikes_in_window) / ((t_end - t_start) / 1000.0)
    else:
        return 0.0

def calculate_spike_amplitudes(sol, model, threshold=-37.0):
    """Calculate amplitudes of all spikes"""
    times = sol.ts
    v_soma = sol.ys[:, 0]
    
    # Get spike times
    spike_times = model.get_spike_times(sol, threshold=threshold)
    
    # If no spikes, return empty array
    if len(spike_times) == 0:
        return jnp.array([])
    
    # Find spike peaks
    amplitudes = []
    for spike_time in spike_times:
        # Find index closest to spike time
        idx = jnp.argmin(jnp.abs(times - spike_time))
        # Look at a window after the spike time to find peak
        window = slice(idx, min(idx + 20, len(times)))
        peak = jnp.max(v_soma[window])
        amplitudes.append(peak)
    
    return jnp.array(amplitudes)

def calculate_spike_adaptation(sol, model, t_start=500, t_end=1500, threshold=-37.0):
    """Calculate spike frequency adaptation ratio (early ISI / late ISI)"""
    # Get spike times using the model's method
    spike_times = model.get_spike_times(sol, threshold=threshold)
    
    # Filter spikes within the analysis window
    mask = (spike_times >= t_start) & (spike_times <= t_end)
    spikes_in_window = spike_times[mask]
    
    # Need at least 3 spikes to calculate adaptation
    if len(spikes_in_window) < 3:
        # Return default value with high uncertainty
        return 1.4, 100.0
    
    # Calculate ISIs
    isis = jnp.diff(spikes_in_window)
    
    # Calculate early and late ISI averages
    n_early = min(3, len(isis)//3)
    n_late = min(3, len(isis)//3)
    
    early_isis = isis[:n_early]
    late_isis = isis[-n_late:]
    
    early_avg = jnp.mean(early_isis)
    late_avg = jnp.mean(late_isis)
    
    # Calculate adaptation ratio (early/late)
    # If late_avg is very small, avoid division issues
    adaptation_ratio = jnp.where(late_avg > 1.0, late_avg / early_avg, 1.0)
    
    # Also return uncertainty based on number of spikes
    uncertainty = 1.0 / len(spikes_in_window)
    
    return adaptation_ratio, uncertainty

def comprehensive_loss(params, yaml_file="PRmodel_Motoneuron/motoneuron.yaml", weights=None):
    """
    Comprehensive loss function incorporating all target objectives
    
    Args:
        params: Dictionary of model parameters
        yaml_file: Path to YAML file with target values
        weights: Dictionary of weights for each target
        
    Returns:
        total_loss: Weighted sum of all target losses
    """
    # Set default weights if not provided
    if weights is None:
        weights = {
            'Rin': 1.0,
            'tau0': 1.0,
            'f_I': 2.0,        # Important for neuronal behavior
            'spike_amp': 1.0,
            'spike_adaptation': 1.0,
            'V_rest': 0.5,
            'V_hold': 0.5
        }
    
    # Load target values
    targets = load_target_values(yaml_file)
    
    # Initialize model with current parameters
    model = MotoneuronModel(yaml_file=yaml_file)
    
    # Apply parameters to model
    for key, value in params.items():
        setattr(model, key, value)
    
    # Initialize loss components
    losses = {}
    
    # 1. Input resistance (Rin)
    I_rin = targets['Rin']['I'][0] * targets['Rin']['I_factor']
    t_rin = targets['Rin']['t']
    sol_rin = model.solve(t_dur=t_rin[1]+100, I_stim=I_rin, stim_start=t_rin[0], stim_end=t_rin[1])
    rin_value = calculate_rin(sol_rin, model, I_amp=I_rin, t_start=t_rin[0], t_end=t_rin[1])
    losses['Rin'] = range_loss(rin_value, targets['Rin']['lower'][0], targets['Rin']['upper'][0])
    
    # 2. Membrane time constant (tau0)
    I_tau = targets['tau0']['I'][0]
    t_tau = targets['tau0']['t']
    sol_tau = model.solve(t_dur=t_tau[1]+100, I_stim=I_tau, stim_start=t_tau[0], stim_end=t_tau[1])
    tau_value = calculate_tau(sol_tau, model, I_amp=I_tau, t_start=t_tau[0], t_end=t_tau[1])
    losses['tau0'] = range_loss(tau_value, targets['tau0']['lower'][0], targets['tau0']['upper'][0])
    
    # 3. F-I relationship (multiple current levels)
    fi_losses = []
    t_fi = targets['f_I']['t']
    for i, I_value in enumerate(targets['f_I']['I']):
        I_amp = I_value * targets['f_I']['I_factor']
        sol_fi = model.solve(t_dur=t_fi[1]+100, I_stim=I_amp, stim_start=t_fi[0], stim_end=t_fi[1])
        firing_rate = calculate_firing_rate(sol_fi, model, t_start=t_fi[0], t_end=t_fi[1])
        target_rate = targets['f_I']['mean'][i]
        fi_loss = squared_error(firing_rate, target_rate)
        fi_losses.append(fi_loss)
    
    # Average f-I losses across currents
    losses['f_I'] = jnp.mean(jnp.array(fi_losses))
    
    # 4. Spike amplitude
    spike_amp_losses = []
    for i, I_value in enumerate(targets['f_I']['I']):
        I_amp = I_value * targets['f_I']['I_factor']
        sol_amp = model.solve(t_dur=t_fi[1]+100, I_stim=I_amp, stim_start=t_fi[0], stim_end=t_fi[1])
        amplitudes = calculate_spike_amplitudes(sol_amp, model)
        
        # Skip if no spikes
        if len(amplitudes) == 0:
            continue
            
        # Calculate loss for each amplitude
        amp_loss = range_loss(jnp.mean(amplitudes), 
                             targets['spike_amp']['lower'][i], 
                             targets['spike_amp']['upper'][i])
        spike_amp_losses.append(amp_loss)
    
    # If we had any spikes, calculate average amplitude loss
    if spike_amp_losses:
        losses['spike_amp'] = jnp.mean(jnp.array(spike_amp_losses))
    else:
        # Penalize no spikes if we expect them
        losses['spike_amp'] = jnp.array(10.0)
    
    # 5. Spike adaptation
    adaptation_losses = []
    for i, I_value in enumerate(targets['f_I']['I']):
        I_amp = I_value * targets['f_I']['I_factor']
        sol_adapt = model.solve(t_dur=t_fi[1]+100, I_stim=I_amp, stim_start=t_fi[0], stim_end=t_fi[1])
        adapt_ratio, uncertainty = calculate_spike_adaptation(sol_adapt, model, 
                                                             t_start=t_fi[0], t_end=t_fi[1])
        
        # Weight by certainty (more spikes = more certain)
        adapt_loss = range_loss(adapt_ratio, 
                               targets['spike_adaptation']['lower'][i], 
                               targets['spike_adaptation']['upper'][i])
        # Scale by uncertainty (less weight if fewer spikes)
        adapt_loss = adapt_loss * (1.0 - uncertainty)
        adaptation_losses.append(adapt_loss)
    
    if adaptation_losses:
        losses['spike_adaptation'] = jnp.mean(jnp.array(adaptation_losses))
    else:
        losses['spike_adaptation'] = jnp.array(10.0)
    
    # 6. Resting and holding potentials
    # Simple simulation without stimulus to check resting potential
    sol_rest = model.solve(t_dur=200, I_stim=0.0, stim_start=0, stim_end=0)
    v_rest = sol_rest.ys[-1, 0]  # Last voltage point
    losses['V_rest'] = squared_error(v_rest, targets['V_rest']['val'])
    
    # Calculate total loss as weighted sum
    total_loss = 0.0
    for key, loss in losses.items():
        total_loss += weights.get(key, 1.0) * loss
    
    return total_loss, losses

def optimize_model(yaml_file="PRmodel_Motoneuron/motoneuron.yaml", 
                 learning_rate=0.01, 
                 n_iterations=1000):
    """
    Optimize model parameters to minimize the comprehensive loss
    
    Args:
        yaml_file: Path to YAML file with target values
        learning_rate: Learning rate for optimizer
        n_iterations: Number of optimization iterations
        
    Returns:
        optimized_params: Dictionary of optimized parameters
    """
    # Load initial parameters from YAML
    with open(yaml_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # Get parameter space from YAML
    param_space = yaml_data['Space']
    
    # Create initial parameters (use midpoint of ranges)
    initial_params = {}
    for key, range_vals in param_space.items():
        initial_params[key] = (range_vals[0] + range_vals[1]) / 2.0
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(initial_params)
    
    # Define loss function for JAX
    @jax.jit
    def loss_fn(params):
        loss, _ = comprehensive_loss(params, yaml_file)
        return loss
    
    # Define update step
    @jax.jit
    def update(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Optimization loop
    params = initial_params
    for i in range(n_iterations):
        params, opt_state, loss = update(params, opt_state)
        
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")
    
    return params
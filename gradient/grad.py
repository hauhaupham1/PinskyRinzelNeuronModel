import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt
import matplotlib.pyplot as plt

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)

def pinsky_rinzel_model(t, y, params):
    """Vector field for the Pinsky-Rinzel model"""
    # Unpack parameters
    g_Ca = params['g_Ca']
    g_AHP = params['g_AHP']
    g_c = params['g_c']
    I_stim = params['I_stim']
    
    # Fixed parameters
    C_m = 3.0
    p = 0.5
    g_L = 0.1
    g_Na = 30.0
    g_DR = 15.0
    g_C = 15.0
    E_L = -68.0
    E_Na = 60.0
    E_K = -75.0
    E_Ca = 80.0
    
    # Unpack state variables
    Vs, Vd, n, h, s, c, q, Ca = y
    
    # Calculate ionic currents
    I_leak_s = g_L * (Vs - E_L)
    I_leak_d = g_L * (Vd - E_L)
    
    # m_inf calculation
    alpha_m = -0.32 * (Vs + 46.9) / (jnp.exp(-(Vs + 46.9) / 4.0) - 1.0)
    beta_m = 0.28 * (Vs + 19.9) / (jnp.exp((Vs + 19.9) / 5.0) - 1.0)
    m_inf = alpha_m / (alpha_m + beta_m)
    
    # Ionic currents
    I_Na = g_Na * m_inf**2 * h * (Vs - E_Na)
    I_DR = g_DR * n * (Vs - E_K)
    I_ds = g_c * (Vd - Vs)
    
    I_Ca = g_Ca * s**2 * (Vd - E_Ca)
    
    # Smoothed version of min function
    chi = jnp.minimum(Ca/250.0, 1.0)
    
    I_AHP = g_AHP * q * (Vd - E_K)
    I_C = g_C * c * chi * (Vd - E_K)
    I_sd = -I_ds
    
    # Calculate if stimulus should be applied (between 100ms and 300ms)
    stim_active = jnp.logical_and(t >= 100.0, t <= 300.0)
    I_applied = jnp.where(stim_active, I_stim, 0.0)
    
    # Differential equations
    dVsdt = (1.0/C_m) * (-I_leak_s - I_Na - I_DR + I_ds/p + I_applied/p)
    dVddt = (1.0/C_m) * (-I_leak_d - I_Ca - I_AHP - I_C + I_sd/(1.0-p))
    
    # Gating variables
    alpha_h = 0.128 * jnp.exp((-43.0 - Vs) / 18.0)
    beta_h = 4.0 / (1.0 + jnp.exp((-20.0 - Vs) / 5.0))
    
    alpha_n = -0.016 * (Vs + 24.9) / (jnp.exp(-(Vs + 24.9) / 5.0) - 1.0)
    beta_n = 0.25 * jnp.exp(-(Vs + 40.0) / 40.0)
    
    alpha_s = 1.6 / (1.0 + jnp.exp(-0.072 * (Vd - 5.0)))
    beta_s = 0.02 * (Vd + 8.9) / (jnp.exp((Vd + 8.9) / 5.0) - 1.0)
    
    # Handle the conditional for alpha_c with smooth transition
    V7 = Vd + 53.5
    V8 = Vd + 50.0
    alpha_c = jnp.where(
        Vd <= -10.0,
        0.0527 * jnp.exp(V8/11.0 - V7/27.0),
        2.0 * jnp.exp(-V7 / 27.0)
    )
    
    beta_c = jnp.where(
        Vd <= -10.0,
        2.0 * jnp.exp(-V7 / 27.0) - alpha_c,
        0.0
    )
    
    # Smooth approximation for alpha_q
    alpha_q = jnp.minimum(0.00002*Ca, 0.01)
    beta_q = 0.001
    
    dhdt = alpha_h * (1.0 - h) - beta_h * h
    dndt = alpha_n * (1.0 - n) - beta_n * n
    dsdt = alpha_s * (1.0 - s) - beta_s * s
    dcdt = alpha_c * (1.0 - c) - beta_c * c
    dqdt = alpha_q * (1.0 - q) - beta_q * q
    dCadt = -0.13 * I_Ca - 0.075 * Ca
    
    return jnp.array([dVsdt, dVddt, dndt, dhdt, dsdt, dcdt, dqdt, dCadt])

def simulate_model(params):
    """Run simulation and return results"""
    initial_state = jnp.array([
        -68.0,  # Vs
        -68.0,  # Vd
        0.001,  # n
        0.999,  # h
        0.009,  # s
        0.007,  # c
        0.01,   # q
        0.2     # Ca
    ])
    
    # Solve the ODE
    sol = diffeqsolve(
        ODETerm(pinsky_rinzel_model),
        solver=Dopri5(),
        t0=0,
        t1=500,  # 500ms simulation
        dt0=0.05,
        y0=initial_state,
        args=params,
        saveat=SaveAt(ts=jnp.linspace(0, 500, 1001)),  # 1ms resolution
        max_steps=1000000,
    )
    
    return sol

def loss_fn(params):
    """Loss function with better scaling"""
    sol = simulate_model(params)
    
    # Extract voltage traces
    Vs = sol.ys[:, 0]  # Somatic voltage
    
    # Scale down the targets to produce smaller gradients
    max_v = jnp.max(Vs)
    target_max_v = 30.0
    max_v_loss = jnp.square(jnp.maximum(0, target_max_v - max_v)) / 100.0
    
    v_var = jnp.var(Vs)
    target_var = 400.0
    var_loss = jnp.square(target_var - v_var) / 10000.0
    
    min_v = jnp.min(Vs)
    target_min_v = -70.0
    min_v_loss = jnp.square(target_min_v - min_v) / 100.0
    
    # Combined loss with better scaling
    total_loss = max_v_loss + 0.01 * var_loss + 0.1 * min_v_loss
    
    return total_loss

def clip_gradients(gradients, max_norm=1.0):
    """Clip gradients to prevent explosion"""
    # Calculate the gradient norm
    squared_sum = sum(jnp.sum(g**2) for g in gradients.values())
    global_norm = jnp.sqrt(squared_sum)
    
    # Clip if needed
    clip_factor = jnp.minimum(max_norm / (global_norm + 1e-6), 1.0)
    
    # Apply clipping
    clipped_gradients = {k: v * clip_factor for k, v in gradients.items()}
    return clipped_gradients

def enforce_constraints(params):
    """Ensure parameters stay in physiologically plausible ranges"""
    # Define min and max values for each parameter
    constraints = {
        'g_Ca': (0.1, 20.0),    # Calcium conductance
        'g_AHP': (0.1, 5.0),    # After-hyperpolarization conductance
        'g_c': (0.1, 10.0),     # Coupling conductance
        'I_stim': (0.1, 5.0)    # Stimulus current
    }
    
    # Apply constraints
    constrained_params = {}
    for k, v in params.items():
        min_val, max_val = constraints[k]
        constrained_params[k] = jnp.clip(v, min_val, max_val)
    
    return constrained_params



def plot_simulation(params):
    """Plot the simulation results for visualization"""
    sol = simulate_model(params)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(sol.ts, sol.ys[:, 0], label='Soma (Vs)')
    plt.plot(sol.ts, sol.ys[:, 1], label='Dendrite (Vd)')
    plt.axhline(y=0, color='k', linestyle=':')
    plt.axvspan(100, 300, alpha=0.2, color='yellow', label='Stimulus')
    plt.ylabel('Voltage (mV)')
    plt.legend()
    plt.title('Membrane Potentials')
    
    plt.subplot(2, 1, 2)
    plt.plot(sol.ts, sol.ys[:, 7], label='Ca²⁺ Concentration')
    plt.axvspan(100, 300, alpha=0.2, color='yellow')
    plt.ylabel('Concentration')
    plt.xlabel('Time (ms)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('pinsky_rinzel_simulation.png')
    plt.close()
    
    return sol

# Initial parameters
initial_params = {
    'g_Ca': 10.0,    # Calcium conductance
    'g_AHP': 0.8,    # After-hyperpolarization conductance 
    'g_c': 2.1,      # Coupling conductance
    'I_stim': 2.0    # Stimulus current
}

# Perform optimization with proper constraints and tiny learning rate
learning_rate = 0.0001  # Much smaller learning rate
print("\nPerforming 5 optimization steps with constraints...")

grad_fn = jax.grad(loss_fn)
params = initial_params.copy()

for i in range(5):
    # Calculate gradients
    gradients = grad_fn(params)
    print(f"Raw gradients: {gradients}")
    
    # Clip gradients
    clipped_gradients = clip_gradients(gradients, max_norm=1.0)
    print(f"Clipped gradients: {clipped_gradients}")
    
    # Update parameters
    for k in params:
        params[k] = params[k] - learning_rate * clipped_gradients[k]
    
    # Enforce constraints
    params = enforce_constraints(params)
    
    # Calculate new loss
    loss = loss_fn(params)
    print(f"Step {i+1}, Loss: {loss}, Params: {params}")

# Plot final simulation
print("\nFinal simulation after optimization:")
sol = plot_simulation(params)
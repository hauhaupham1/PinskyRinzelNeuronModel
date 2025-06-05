"""
Pinsky-Rinzel Motoneuron Network Implementation

Credits:
    - Solution class and paths implementation by Christian Holberg
    - Network implementation inspired by snnax package from Christian Holberg
"""

import functools as ft
from typing import Any, Callable, List, Optional, Sequence
import numpy as np
import gc
from diffrax import ODETerm, ConstantStepSize, Event, Euler, SubSaveAt, SaveAt, ControlTerm, MultiTerm, diffeqsolve
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optimistix 
from jax._src.ad_util import stop_gradient_p
from jax.interpreters import ad
from jaxtyping import Array, Bool, Float, Int, Real, PyTree, ArrayLike
import jax.tree_util as jtu
from .solution import Solution
from .paths import BrownianPath, SingleSpikeTrain
from .network_state import NetworkState
from .helpers import _build_w, buffers

# Work around JAX issue #22011, similar to what's in snn.py
def stop_gradient_transpose(ct, x):
    return (ct,)
ad.primitive_transposes[stop_gradient_p] = stop_gradient_transpose

class MotoneuronNetwork(eqx.Module):
    """A class representing a network of Pinsky-Rinzel motoneurons."""

    num_neurons: Int
    w: Float[Array, "neurons neurons"]
    network: Bool[ArrayLike, "neurons neurons"] = eqx.field(static=True)
    read_out_neurons: Sequence[Int]
    threshold: float
    v_reset: float
    vector_field: Callable[..., Float[ArrayLike, "neurons 8"]]
    _vf_implementation: Callable[..., Float[ArrayLike, "neurons 8"]]
    # PR model parameters
    C_m: Float
    p: Float
    g_c: Float
    g_L_soma: Float
    g_L_dend: Float
    g_Na: Float
    g_DR: Float
    g_Ca: Float
    g_AHP: Float
    g_C: Float
    E_L: Float
    E_Na: Float
    E_K: Float
    E_Ca: Float
    f_Caconc: Float
    alpha_Caconc: Float
    kCa_Caconc: Float
    sigma: Optional[Float[ArrayLike, "2 2"]]
    diffusion_vf: Optional[Callable[..., Float[ArrayLike, "neurons 8 2 neurons"]]]
    cond_fn: List[Callable[..., Float]]
    def __init__(
        self,
        num_neurons: Int,
        v_reset: Float = -60.0,
        threshold: Float = -20.0,
        w: Optional[Float[Array, "neurons neurons"]] = None,
        network: Optional[Bool[ArrayLike, "neurons neurons"]] = None,
        read_out_neurons: Optional[Sequence[Int]] = None,
        wmin: Float = 0.0,
        wmax: Float = 0.5,
        diffusion: bool = False,
        sigma: Optional[Float[ArrayLike, "2 2"]] = None,
        key: Optional[Any] = None,
        **pr_params # Allows other PR params to be passed
    ):
        """
        Initialize the motoneuron network.
        
        Args:
            num_neurons: The number of neurons in the network.
            v_reset: The reset voltage value for neurons. Defaults to -60.0.
            threshold: The threshold voltage for spike detection. Defaults to -20.0.
            w: The initial weight matrix. If none, randomly initialized.
            network: The connectivity matrix. If none, fully disconnected.
            read_out_neurons: Neurons designated as output. If none, empty list.
            wmin: Minimum weight for random initialization. Defaults to 0.0.
            wmax: Maximum weight for random initialization. Defaults to 0.5.
            diffusion: Whether to include diffusion term in the SDE. Defaults to False.
            sigma: A 2x2 diffusion matrix. If none, randomly initialized.
            key: Random key for initialization. If none, set to PRNGKey(0).
            **pr_params: Additional Pinsky-Rinzel model parameters.
        """
        self.num_neurons = num_neurons
        self.v_reset = v_reset
        self.threshold = threshold

        if key is None:
            key = jr.PRNGKey(0)

        w_key, sigma_key = jr.split(key, 2)

        if network is None:
            # Default to a disconnected network if none provided
            network = np.full((num_neurons, num_neurons), False, dtype=bool) 

        self.w = _build_w(w, network, w_key, wmin, wmax)
        self.network = network
        if read_out_neurons is None:
            read_out_neurons = []
        self.read_out_neurons = read_out_neurons
        # Set default Pinsky-Rinzel parameters
        self.C_m = 2.0  # Membrane capacitance
        self.p = 0.1  # Proportion of membrane area for soma
        self.g_c = 0.4  # Coupling conductance
        self.g_L_soma = 0.0004  # Leak conductance in soma
        self.g_L_dend = 0.00001  # Leak conductance in dendrite
        self.g_Na = 0.27  # Sodium conductance
        self.g_DR = 0.24  # Delayed rectifier K+ conductance
        self.g_Ca = 0.008  # Calcium conductance
        self.g_AHP = 0.005  # After-hyperpolarization K+ conductance
        self.g_C = 0.009  # Ca-dependent K+ conductance
        self.E_L = -62.0  # Leak reversal potential
        self.E_Na = 60.0  # Sodium reversal potential
        self.E_K = -75.0  # Potassium reversal potential
        self.E_Ca = 80.0  # Calcium reversal potential
        self.f_Caconc = 0.004  # Calcium dynamics parameter
        self.alpha_Caconc = 1.0  # Calcium dynamics parameter
        self.kCa_Caconc = 8.0  # Calcium dynamics parameter

        # Override with provided parameters from pr_params
        for param_key, value in pr_params.items():
            if hasattr(self, param_key):
                setattr(self, param_key, value)
        # Helper functions for Pinsky-Rinzel model
        def alpha_m(V):
            V1 = V + 46.9
            return -0.32 * V1 / (jnp.exp(-V1 / 4.) - 1.)
        def beta_m(V):
            V2 = V + 19.9
            return 0.28 * V2 / (jnp.exp(V2 / 5.) - 1.)
        def m_inf(V):
            return alpha_m(V) / (alpha_m(V) + beta_m(V))
        def alpha_h(V):
            return 0.128 * jnp.exp((-43. - V) / 18.)
        def beta_h(V):
            V5 = V + 20.
            return 4. / (1 + jnp.exp(-V5 / 5.))
        def alpha_n(V):
            V3 = V + 24.9
            return -0.016 * V3 / (jnp.exp(-V3 / 5.) - 1)
        def beta_n(V):
            V4 = V + 40.
            return 0.25 * jnp.exp(-V4 / 40.)
        def alpha_s(V):
            return 1.6 / (1 + jnp.exp(-0.072 * (V - 5.)))
        def beta_s(V):
            V6 = V + 8.9
            return 0.02 * V6 / (jnp.exp(V6 / 5.) - 1.)
        def alpha_c(V):
            V7 = V + 53.5
            V8 = V + 50.
            return jnp.where(
                V <= -10,
                0.0527 * jnp.exp(V8/11. - V7/27.),
                2 * jnp.exp(-V7 / 27.)
            )
        def beta_c(V):
            V7 = V + 53.5
            alpha_c_val = alpha_c(V)
            return jnp.where(
                V <= -10,
                2. * jnp.exp(-V7 / 27.) - alpha_c_val,
                0.
            )
        def alpha_q(Ca):
            return jnp.minimum(0.00002*Ca, 0.01)
        def beta_q(Ca):
            return 0.001
        def chi(Ca):
            return jnp.minimum(Ca/250., 1.)
        
        # Define the vector field function (drift term)
        @jax.vmap
        def _vf(y, ic):
            Vs, Vd, n, h, s, c, q, Ca = y
            # Somatic currents
            I_leak_s = self.g_L_soma * (Vs - self.E_L)
            I_Na = self.g_Na * (m_inf(Vs)**2 * h) * (Vs - self.E_Na)
            I_DR = self.g_DR * n * (Vs - self.E_K)
            I_ds = self.g_c * (Vd - Vs)
            # Dendritic currents
            I_leak_d = self.g_L_dend * (Vd - self.E_L)
            I_Ca = self.g_Ca * (s**2) * (Vd - self.E_Ca)
            I_AHP = self.g_AHP * q * (Vd - self.E_K)
            I_C = self.g_C * c * chi(Ca) * (Vd - self.E_K)
            I_sd = -I_ds
            # Differential equations
            dVsdt = (1.0/self.C_m) * (-I_leak_s - I_Na - I_DR + I_ds/self.p + ic)
            dVddt = (1.0/self.C_m) * (-I_leak_d - I_Ca - I_AHP - I_C + I_sd/(1.0-self.p))
            dhdt = alpha_h(Vs)*(1-h) - beta_h(Vs)*h
            dndt = alpha_n(Vs)*(1-n) - beta_n(Vs)*n
            dsdt = alpha_s(Vd)*(1-s) - beta_s(Vd)*s
            dcdt = alpha_c(Vd)*(1-c) - beta_c(Vd)*c
            dqdt = alpha_q(Ca)*(1-q) - beta_q(Ca)*q
            dCadt = -self.f_Caconc*I_Ca - self.alpha_Caconc*Ca/self.kCa_Caconc
            out = jnp.array([dVsdt, dVddt, dndt, dhdt, dsdt, dcdt, dqdt, dCadt])
            out = eqx.error_if(out, jnp.any(jnp.isnan(out)), "out is NaN")
            out = eqx.error_if(out, jnp.any(jnp.isinf(out)), "out is Inf")
            
            return out
        self._vf_implementation = _vf
                

        #Wrapper for the vector field function
        def vector_field_wrapper(t, y, ode_args_dict): 
            # Unpack from the ode_args_dict
            input_current_func = ode_args_dict['input_current'] # This is the callable      
            synaptic_I_arr = ode_args_dict['synaptic_I']       

            # Get current magnitudes by calling the function
            current_magnitudes_at_t = input_current_func(t)

            external_current_to_apply = current_magnitudes_at_t / self.p 
            total_current = external_current_to_apply + synaptic_I_arr
            
            # Call the stored core dynamics implementation
            return self._vf_implementation(y, total_current) 
        self.vector_field = vector_field_wrapper

        # Add optional diffusion term
        if diffusion:
            if sigma is None:
                sigma = jr.normal(sigma_key, (2, 2))
                sigma = jnp.dot(sigma, sigma.T)
                self.sigma = sigma
            sigma_large = jnp.zeros((num_neurons, 8, 2, num_neurons))
            # Only add noise to voltage components
            for k in range(num_neurons):
                sigma_large = sigma_large.at[k, :2, :, k].set(sigma)

            def diffusion_vf(t, y, args):
                return sigma_large

            self.diffusion_vf = diffusion_vf
            self.sigma = sigma
        else:
            self.sigma = None
            self.diffusion_vf = None

        # Define condition function for spike detection per neuron
        def cond_fn(t, y, args, n, **kwargs):
            # root when neuron n crosses threshold
            return y[n, 0] - self.threshold

        # List of condition functions for each neuron
        self.cond_fn = [ft.partial(cond_fn, n=n) for n in range(self.num_neurons) if n not in self.read_out_neurons]

    def _initialize_state(self, num_samples=1):
        """Initialize state variables for all neurons."""
        V_init = -60.0
        # Create initial state arrays for all neurons
        Vs0 = jnp.ones((num_samples, self.num_neurons)) * V_init
        Vd0 = jnp.ones((num_samples, self.num_neurons)) * V_init
        n0 = jnp.ones((num_samples, self.num_neurons)) * 0.001
        h0 = jnp.ones((num_samples, self.num_neurons)) * 0.999
        s0 = jnp.ones((num_samples, self.num_neurons)) * 0.009
        c0 = jnp.ones((num_samples, self.num_neurons)) * 0.007
        q0 = jnp.ones((num_samples, self.num_neurons)) * 0.01
        Ca0 = jnp.ones((num_samples, self.num_neurons)) * 0.2
        
        return jnp.stack([Vs0, Vd0, n0, h0, s0, c0, q0, Ca0], axis=-1)

    @eqx.filter_jit
    def __call__(
        self,
        input_current: Callable[..., Float[Array, " neurons"]],
        t0: Real,
        t1: Real,
        max_spikes: Int,
        num_samples: Int,
        *,
        key,
        input_spikes: Optional[Float[Array, "samples input_neurons"]] = None,
        input_weights: Optional[Float[Array, "neurons input_neurons"]] = None,
        y0: Optional[Float[Array, "samples neurons 8"]] = None,
        synaptic_I0: Optional[Float[Array, "samples neurons"]] = None,  # Initial synaptic currents
        num_save: Int = 2,
        dt0: Real = 0.01,
        max_steps: Int = 1000,
        store_vars: tuple = ('Vs',),  # Which state variables to store (default: only somatic voltage)
        memory_efficient: bool = True,  # Whether to use memory-efficient storage
        spike_only: bool = False,  # If True, only store spike times
        discard_voltage_between_spikes: bool = False,  # If True, only keep voltage near spikes
        is_final_simulation: bool = True,  # If True, include end spike at t1 (not for chunks)
        # Solver continuity parameters
        solver_state0: Optional[PyTree] = None,
        controller_state0: Optional[PyTree] = None,
        made_jump0: Optional[Bool[Array, "samples"]] = None,
    ):
        """
        Run the network simulation.
        
        Args:
            input_current: Function taking time t and returning currents for each neuron.
            t0: Start time of simulation.
            t1: End time of simulation.
            max_spikes: Maximum number of spikes to track per neuron.
            num_samples: Number of parallel simulations to run.
            key: Random key for simulation.
            input_spikes: Optional array of input spike times.
            input_weights: Optional weights for input spikes.
            y0: Optional initial state. If None, randomly initialized.
            synaptic_I0: Optional initial synaptic currents. If None, initialized to zero.
            num_save: Number of time points to save per spike event.
            dt0: Initial time step size.
            max_steps: Maximum number of steps in the ODE solver.
            
        Returns:
            Solution object containing simulation results.
        """
        #check is store_vars is a tuple
        if not isinstance(store_vars, tuple):
            raise ValueError("store_vars must be a tuple of state variable names.")
        # Convert scalar t0 to array for each sample
        t0_scalar, t1_scalar = float(t0), float(t1)  # Keep as JAX arrays for compatibility
        _t0_array = jnp.broadcast_to(t0_scalar, (num_samples,)) # For NetworkState.t0
        # Initialize random keys
        key, bm_key = jr.split(key, 2)
        # Split random keys for Brownian motion per sample
        bm_key = jr.split(bm_key, num_samples)
        
        # Setup input spikes if provided
        if input_weights is not None:
            assert input_spikes is not None
            input_dim = input_spikes.shape[1]
            input_w_large = jnp.zeros((self.num_neurons, 8, input_dim))
            # Only add input to the dendritic current
            input_w_large = input_w_large.at[:, 1, :].set(input_weights)
            def input_vf(t, y, args):
                return input_w_large
        # Initialize state if not provided
        y0_resolved = y0 if y0 is not None else self._initialize_state(num_samples)
        # Pre-calculate indices outside the loop for efficiency
        state_var_indices = {
            'Vs': 0, 'Vd': 1, 'n': 2, 'h': 3, 's': 4, 'c': 5, 'q': 6, 'Ca': 7
        }
        # Determine which state variables to store and their indices
        if spike_only:
            state_indices_array = jnp.array([0])  # Only track Vs for spike detection
            num_state_vars = 1
            extract_single_var = True
            store_vars = ('Vs',)  # Override to only track voltage for spikes
            # Minimal storage - only 2 time points per spike for detection
            ys = jnp.full((num_samples, max_spikes, self.num_neurons, 2, 1), jnp.inf)
            num_save = 2  # Override num_save for spike-only mode
        elif memory_efficient:
            # Validate requested state variables
            for var in store_vars:
                if var not in state_var_indices:
                    raise ValueError(f"Unknown state variable: {var}")
            
            # Pre-calculate the indices as a JAX array (crucial for efficiency)
            state_indices_array = jnp.array([state_var_indices[var] for var in store_vars])
            num_state_vars = len(store_vars)
            extract_single_var = (num_state_vars == 1)
            
            # Initialize storage with only the requested state variables
            ys = jnp.full((num_samples, max_spikes, self.num_neurons, num_save, num_state_vars), jnp.inf)
        else:
            # Store all 8 state variables (original behavior)
            store_vars = ('Vs', 'Vd', 'n', 'h', 's', 'c', 'q', 'Ca')
            state_indices_array = jnp.arange(8)  # [0, 1, 2, 3, 4, 5, 6, 7]
            extract_single_var = False
            ys = jnp.full((num_samples, max_spikes, self.num_neurons, num_save, 8), jnp.inf)
        
        # Initialize other storage arrays
        if spike_only:
            ts = jnp.full((num_samples, max_spikes, 2), jnp.inf)  # Minimal time storage
        else:
            ts = jnp.full((num_samples, max_spikes, num_save), jnp.inf)
        tevents = jnp.full((num_samples, max_spikes), jnp.inf)
        num_spikes = 0
        event_mask = jnp.full((num_samples, self.num_neurons), False)
        event_types = jnp.full((num_samples, max_spikes, self.num_neurons), False)

        #Initialize synaptic_I
        if synaptic_I0 is not None:
            init_synaptic_I = synaptic_I0
        else:
            init_synaptic_I = jnp.zeros((num_samples, self.num_neurons))

        _solver_state0_for_init = solver_state0
        _controller_state0_for_init = controller_state0
        _made_jump0_for_init = made_jump0

        # For vmap compatibility, keep as None if not provided initially
        # The vmap will broadcast None to all samples
        if _made_jump0_for_init is None:
            _made_jump0_for_init = None  # Keep as None for broadcasting
        
        # Create initial state
        init_state = NetworkState(
            ts=ts,
            ys=ys, 
            tevents=tevents,
            t0=_t0_array, 
            y0=y0_resolved, 
            num_spikes=num_spikes, 
            event_mask=event_mask, 
            event_types=event_types, 
            key=key,
            synaptic_I=init_synaptic_I,
            stored_vars=store_vars,
            spike_only=spike_only,
            discard_voltage_between_spikes=discard_voltage_between_spikes,
            solver_state=_solver_state0_for_init,
            controller_state=_controller_state0_for_init,
            made_jump=_made_jump0_for_init,
        )
        
        # Setup diffrax solver
        stepsize_controller = ConstantStepSize()
        vf = ODETerm(self.vector_field)
        root_finder = optimistix.Newton(1e-6, 1e-6, optimistix.rms_norm)
        event = Event(self.cond_fn, root_finder)
        solver = Euler()
        w_update = jnp.where(self.network, self.w, 0.0)
        
        # Define transition function to reset state after spikes
        @jax.vmap
        def trans_fn(y, event, key):
            Vs, Vd, n, h, s, c, q, Ca = y
            # reset all state variables upon spike event
            Vs_out = jnp.where(event, self.v_reset, Vs)
            Vd_out = jnp.where(event, self.v_reset, Vd)
            n_out  = jnp.where(event, 0.001, n)
            h_out  = jnp.where(event, 0.999, h)
            s_out  = jnp.where(event, 0.009, s)
            c_out  = jnp.where(event, 0.007, c)
            q_out  = jnp.where(event, 0.01, q)
            Ca_out = jnp.where(event, 0.2, Ca)
            return jnp.stack([Vs_out, Vd_out, n_out, h_out, s_out, c_out, q_out, Ca_out], axis=-1)

        # Main simulation loop
        def body_fun(state: NetworkState) -> NetworkState:
            new_key, trans_key = jr.split(state.key, 2)
            trans_key = jr.split(trans_key, num_samples)
            
            @ft.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None))
            def update(_t0_seg, _y0_seg, _trans_key_seg, _bm_key_seg, _input_spike_seg, _pre_synaptic_I_seg, 
                       _state_indices, _single_var_mode, _spike_only, _discard_voltage, 
                       _solver_state_seg, _controller_state_seg, _made_jump_seg):
                # Define time points to save
                # Use t1_scalar which is the overall end time for this __call__
                if _spike_only:
                    ts_save = jnp.array([_t0_seg, t1_scalar])
                else:
                    ts_save = jnp.where(
                        _t0_seg < t1_scalar - (t1_scalar - _t0_seg) / (10 * num_save), # Avoid div by zero if _t0_seg is close to t1_scalar
                        jnp.linspace(_t0_seg, t1_scalar, num_save),
                        jnp.full((num_save,), _t0_seg),
                    )
                ts_save = eqxi.error_if(ts_save, jnp.any(ts_save[1:] < ts_save[:-1]), "ts_save must be increasing")
                
                # Set up save points
                # trans_key is already per sample, split further if needed per neuron (not typical for saveat)
                saveat_ts = SubSaveAt(ts=ts_save)
                saveat_t1 = SubSaveAt(t1=True)
                saveat = SaveAt(subs=[saveat_ts, saveat_t1], solver_state=False, controller_state=False, made_jump=False)
                
                # Set up terms for solver
                current_ode_args = {
                      "synaptic_I": _pre_synaptic_I_seg, 
                      "input_current": input_current, 
                  }
                terms = vf
                multi_terms = []
                
                # Add diffusion if enabled
                if self.diffusion_vf is not None:
                    # BrownianPath t0, t1 should span the entire simulation window of this __call__
                    bm = BrownianPath(
                        t0_scalar -1, t1_scalar + 1, tol=dt0 / 2, shape=(2, self.num_neurons), key=_bm_key_seg 
                    )
                    cvf = ControlTerm(self.diffusion_vf, bm)
                    multi_terms.append(cvf)
                    
                # Add input spikes if provided
                if _input_spike_seg is not None:
                    assert input_weights is not None # This assertion is outside JIT, will only run once
                    # SingleSpikeTrain t0, t1 should span the segment or overall __call__ window
                    input_st = SingleSpikeTrain(t0_scalar, t1_scalar, _input_spike_seg) 
                    input_cvf = ControlTerm(input_vf, input_st) # input_vf is defined in __call__ scope
                    multi_terms.append(input_cvf)
                    
                # Combine all terms
                if multi_terms:
                    terms = MultiTerm(terms, *multi_terms)
                    
                # Solve the ODE system
                sol = diffeqsolve(
                    terms,
                    solver,
                    _t0_seg, # Start time for this segment
                    t1_scalar, # Overall end time for this __call__
                    dt0,
                    _y0_seg, # Initial state for this segment
                    args=current_ode_args,
                    throw=False,
                    stepsize_controller=stepsize_controller,
                    saveat=saveat,
                    event=event,
                    max_steps=max_steps,
                    solver_state=_solver_state_seg,
                    controller_state=_controller_state_seg,
                    made_jump=_made_jump_seg,
                )
                # Process results
                assert sol.event_mask is not None
                event_mask_out = jnp.array(sol.event_mask) # This is per-event, not per-neuron for the final state
                event_happened_out = jnp.any(event_mask_out) # If any of the discrete events triggered
                
                assert sol.ts is not None
                ts_from_sol = sol.ts[0] # Saved time points for this segment
                _t1_from_sol = sol.ts[1] # End time of this segment (either event time or t1_scalar)
                
                # Get the event time
                raw_tevent = _t1_from_sol[0] # This is a scalar per sample
                # If tevent > t1_scalar we normalize to keep within range (should not happen if t1_scalar is the solve limit)
                tevent_out = jnp.where(raw_tevent > t1_scalar, raw_tevent * (t1_scalar / raw_tevent), raw_tevent)
                tevent_out = eqxi.error_if(tevent_out, jnp.isnan(tevent_out), "tevent_out is nan")
                
                # If no event, tevent_out will be t1_scalar. If event, it's the event time.
                # spike_time_to_store should be inf if no actual spike event happened for this segment.
                # This means tevent_out should be the actual event time if an event happened, otherwise t1_scalar.
                # We need to distinguish between reaching t1_scalar and an event happening.
                # Only record the end time if it's actually the final simulation AND we reached t1 without spike
                # This prevents recording intermediate chunk boundaries as "spikes"
                is_final_time = jnp.abs(tevent_out - t1_scalar) < 1e-10
                spike_time_to_store_out = jnp.where(
                    event_happened_out,  # If real spike happened
                    tevent_out,          # Record the spike time
                    jnp.where(
                        is_final_simulation & is_final_time,  # If final sim AND reached t1 without spike
                        tevent_out,      # Record t1 (snnax behavior)
                        jnp.inf          # Otherwise, don't record (intermediate boundary)
                    )
                )
                assert sol.ys is not None
                full_ys_from_sol = sol.ys[0]  # Saved states for this segment
                _y1_from_sol = sol.ys[1] # State at the end of this segment
                
                # Get event state (all 8 variables needed for next iteration)
                yevent_for_trans = _y1_from_sol[0].reshape((self.num_neurons, 8))
                # If the segment solved all the way to t1_scalar without an event, yevent_for_trans is y(t1_scalar)
                # If an event occurred, it's y(event_time)
                yevent_for_trans = jnp.where(_t0_seg < t1_scalar, yevent_for_trans, _y0_seg) # Ensure we don't use uninitialized if _t0_seg >= t1_scalar
                yevent_for_trans = eqxi.error_if(yevent_for_trans, jnp.any(jnp.isnan(yevent_for_trans)), "yevent_for_trans is nan")
                yevent_for_trans = eqxi.error_if(yevent_for_trans, jnp.any(jnp.isinf(yevent_for_trans)), "yevent_for_trans is inf")
                
                # Determine which neurons spiked based on threshold crossing at yevent_for_trans
                # This event_array_neuron_specific is what should be stored in event_types
                event_array_neuron_specific = jnp.array(yevent_for_trans[:, 0] >= self.threshold - 1e-3)
                
                w_update_t = jnp.where(
                    jnp.tile(event_array_neuron_specific, (self.num_neurons, 1)).T, w_update, 0.0
                ).T
                w_update_t = jnp.where(event_happened_out, w_update_t, 0.0) # Only apply if an event actually triggered the stop
                
                # Apply transition function to reset after spikes
                # Split the key to get one key per neuron for the vmapped trans_fn
                trans_keys_per_neuron = jr.split(_trans_key_seg, self.num_neurons)
                ytrans_out = trans_fn(yevent_for_trans, event_array_neuron_specific, trans_keys_per_neuron)

                new_dI_out = jnp.sum(w_update_t, axis=-1) / self.p
                
                # Reshape the full outputs
                full_ys_from_sol_reshaped = jnp.transpose(full_ys_from_sol, (1, 0, 2)) # (neurons, times, state_vars)
                
                # Efficient state variable extraction using pre-calculated indices
                if _spike_only:
                    ys_stored = jnp.take(full_ys_from_sol_reshaped, jnp.array([0]), axis=-1) 
                    ys_stored = jnp.expand_dims(ys_stored[..., 0], axis=-1)
                elif _discard_voltage:
                    if memory_efficient:
                        ys_stored = jnp.take(full_ys_from_sol_reshaped, _state_indices, axis=-1)
                        ys_stored = jnp.where(_single_var_mode, 
                                      jnp.expand_dims(ys_stored[..., 0], axis=-1),
                                      ys_stored)
                    else:
                        ys_stored = full_ys_from_sol_reshaped
                    ys_stored = jnp.where(event_happened_out, ys_stored, jnp.full_like(ys_stored, jnp.inf))
                elif memory_efficient:
                    ys_stored = jnp.take(full_ys_from_sol_reshaped, _state_indices, axis=-1)
                    ys_stored = jnp.where(_single_var_mode, 
                                  jnp.expand_dims(ys_stored[..., 0], axis=-1),
                                  ys_stored)
                else:
                    ys_stored = full_ys_from_sol_reshaped
                
                # tevent_out is the time for the next segment's t0.
                # If no event, it's t1_scalar. If event, it's the event time.
                next_t0_for_loop = jnp.where(event_happened_out, tevent_out, t1_scalar)
                # Extract solver states from the solution
                solver_state_out = sol.solver_state if hasattr(sol, 'solver_state') else None
                controller_state_out = sol.controller_state if hasattr(sol, 'controller_state') else None
                made_jump_out = sol.made_jump if hasattr(sol, 'made_jump') else None
                
                return (ts_from_sol, ys_stored, next_t0_for_loop, ytrans_out, 
                        event_array_neuron_specific, new_dI_out, spike_time_to_store_out, 
                        event_happened_out, controller_state_out, solver_state_out, made_jump_out)
                
            # Call the vmapped update function
            (_ts_seg_data, _ys_seg_data, _next_t0_val, _y_for_next_seg, 
             _event_mask_neurons, _new_synaptic_I_val, _spike_time_to_store_val, 
             _event_happened_val, _controller_state_out, _solver_state_out, _made_jump_out) = update(
                state.t0, state.y0, trans_key, bm_key, input_spikes, state.synaptic_I, 
                state_indices_array, extract_single_var, state.spike_only, state.discard_voltage_between_spikes,
                state.solver_state, state.controller_state, state.made_jump
            )
            
            # Increment spike counter
            num_spikes = state.num_spikes + 1 # This is the index for storing this segment's data
            
            ts_updated = state.ts.at[:, state.num_spikes].set(_ts_seg_data)
            ys_updated = state.ys.at[:, state.num_spikes].set(_ys_seg_data)

            # Only record event types when we actually have a spike time to record
            should_record = jnp.isfinite(_spike_time_to_store_val)
            event_types_updated = state.event_types.at[:, state.num_spikes].set(
                jnp.where(should_record[:, None], _event_mask_neurons, jnp.zeros_like(_event_mask_neurons))
            )

            tevents_updated = state.tevents.at[:, state.num_spikes].set(_spike_time_to_store_val)
            
            # Return new state for the while_loop
            new_state = NetworkState(
                ts=ts_updated,
                ys=ys_updated,
                tevents=tevents_updated,
                t0=_next_t0_val, # This is the start time for the next segment
                y0=_y_for_next_seg, # This is the initial y for the next segment
                num_spikes=num_spikes,
                event_mask=_event_mask_neurons, # Store the neuron-specific mask for potential use (though event_types is primary)
                event_types=event_types_updated,
                key=new_key,
                synaptic_I=_new_synaptic_I_val,
                stored_vars=store_vars, # from __call__ scope
                spike_only=state.spike_only,
                discard_voltage_between_spikes=state.discard_voltage_between_spikes,
                solver_state=_solver_state_out,  # Use updated solver state
                controller_state=_controller_state_out,  # Use updated controller state
                made_jump=_made_jump_out,  # Use updated made_jump state
            )
            return new_state
            
        # Define stopping condition for the while_loop
        def stop_fn(state: NetworkState) -> bool:
            # Match snnax behavior: continue while num_spikes <= max_spikes AND min(t0) < t1
            return (state.num_spikes <= max_spikes) & (jnp.min(state.t0) < t1_scalar)
            
        # Run simulation using eqxi.while_loop
        final_loop_state = eqxi.while_loop(
            stop_fn,
            body_fun,
            init_state,
            buffers=buffers,
            max_steps=max_spikes, # Max iterations of the loop (max number of segments)
            kind="checkpointed",
        )
        
        # Prepare output solution
        sol = Solution(
            t1=t1_scalar, # Overall end time
            ys=final_loop_state.ys,
            ts=final_loop_state.ts,
            spike_times=final_loop_state.tevents,
            spike_marks=final_loop_state.event_types,
            num_spikes=final_loop_state.num_spikes, # Actual number of segments processed + 1
            max_spikes=max_spikes,
            stored_vars=store_vars, # from __call__ scope
            spike_only=spike_only, # from __call__ scope
            final_state=final_loop_state.y0,
            final_synaptic_I=final_loop_state.synaptic_I,
            solver_state = final_loop_state.solver_state,
            controller_state = final_loop_state.controller_state,
            made_jump = final_loop_state.made_jump,
        )
        return sol
    
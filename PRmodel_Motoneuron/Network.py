import functools as ft
from typing import Any, Callable, List, Optional, Sequence

import diffrax
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optimistix 
from jax._src.ad_util import stop_gradient_p
from jax.interpreters import ad
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, Real

from paths import BrownianPath, SingleSpikeTrain  


# Work around JAX issue #22011, similar to what's in snn.py
def stop_gradient_transpose(ct, x):
    return (ct,)

ad.primitive_transposes[stop_gradient_p] = stop_gradient_transpose


class Solution(eqx.Module):
    """Solution of the event-driven model."""

    t1: Real
    ys: Float[Array, "samples spikes neurons times 8"]
    ts: Float[Array, "samples spikes times"]
    spike_times: Float[Array, "samples spikes"]
    spike_marks: Float[Array, "samples spikes neurons"]
    num_spikes: int
    max_spikes: int


class NetworkState(eqx.Module):
    ts: Real[Array, "samples spikes times"]
    ys: Float[Array, "samples spikes neurons times 8"]
    tevents: eqxi.MaybeBuffer[Real[Array, "samples spikes"]]
    t0: Real[Array, " samples"]
    y0: Float[Array, "samples neurons 8"]
    num_spikes: int
    event_mask: Bool[Array, "samples neurons"]
    event_types: eqxi.MaybeBuffer[Bool[Array, "samples spikes neurons"]]
    key: Any


def buffers(state: NetworkState):
    assert type(state) is NetworkState
    return state.tevents, state.ts, state.ys, state.event_types


def _build_w(w, network, key, minval, maxval):
    if w is not None:
        return w
    w_a = jr.uniform(key, network.shape, minval=minval, maxval=maxval)
    return w_a.at[network].set(0.0)


class MotoneuronNetwork(eqx.Module):
    """A class representing a network of Pinsky-Rinzel motoneurons."""

    num_neurons: Int
    w: Float[Array, "neurons neurons"]
    network: Bool[ArrayLike, "neurons neurons"] = eqx.field(static=True)
    read_out_neurons: Sequence[Int]
    threshold: float
    v_reset: float
    drift_vf: Callable[..., Float[ArrayLike, "neurons 8"]]
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
    # Optional diffusion parameters
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
        **pr_params
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
            network = np.full((num_neurons, num_neurons), False)

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

        # Override with provided parameters
        for key, value in pr_params.items():
            if hasattr(self, key):
                setattr(self, key, value)

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
        def drift_vf(t, y, input_current):
            ic = input_current(t)

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
                
            return _vf(y, ic)
            
        self.drift_vf = drift_vf

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

    def _initialize_state(self, num_samples=1, key=None):
        """Initialize state variables for all neurons."""
        # if key is None:
        #     key = jr.PRNGKey(0)
            
        # v0_key, i0_key = jr.split(key, 2)
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
        
        # Stack into format for diffrax: [samples, neurons, 8]
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
        num_save: Int = 2,
        dt0: Real = 0.01,
        max_steps: Int = 1000,
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
            num_save: Number of time points to save per spike event.
            dt0: Initial time step size.
            max_steps: Maximum number of steps in the ODE solver.
            
        Returns:
            Solution object containing simulation results.
        """
        # Check input current shape
        ic_shape = jax.eval_shape(input_current, 0.0)
        assert ic_shape.shape == (self.num_neurons,)
        
        # Convert scalar t0 to array for each sample
        t0, t1 = float(t0), float(t1)
        _t0 = jnp.broadcast_to(t0, (num_samples,))
        
        # Initialize random keys
        key, bm_key, init_key = jr.split(key, 3)
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
      
        y0 = self._initialize_state(num_samples, init_key)
            
        # Initialize storage arrays for results
        ys = jnp.full((num_samples, max_spikes, self.num_neurons, num_save, 8), jnp.inf)
        ts = jnp.full((num_samples, max_spikes, num_save), jnp.inf)
        tevents = jnp.full((num_samples, max_spikes), jnp.inf)
        num_spikes = 0
        event_mask = jnp.full((num_samples, self.num_neurons), False)
        event_types = jnp.full((num_samples, max_spikes, self.num_neurons), False)
        
        # Create initial state
        init_state = NetworkState(
            ts, ys, tevents, _t0, y0, num_spikes, event_mask, event_types, key
        )
        
        # Setup diffrax solver
        stepsize_controller = diffrax.ConstantStepSize()
        vf = diffrax.ODETerm(self.drift_vf)
        root_finder = optimistix.Newton(1e-2, 1e-2, optimistix.rms_norm)
        event = diffrax.Event(self.cond_fn, root_finder)
        solver = diffrax.Euler()
        w_update = self.w.at[self.network].set(0.0)
        
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
            
            @jax.vmap
            def update(_t0, y0, trans_key, bm_key, input_spike):
                # Define time points to save
                ts = jnp.where(
                    _t0 < t1 - (t1 - t0) / (10 * num_save),
                    jnp.linspace(_t0, t1, num_save),
                    jnp.full((num_save,), _t0),
                )
                ts = eqxi.error_if(ts, ts[1:] < ts[:-1], "ts must be increasing")
                
                # Set up save points
                trans_key = jr.split(trans_key, self.num_neurons)
                saveat_ts = diffrax.SubSaveAt(ts=ts)
                saveat_t1 = diffrax.SubSaveAt(t1=True)
                saveat = diffrax.SaveAt(subs=(saveat_ts, saveat_t1))
                
                # Set up terms for solver
                terms = vf
                multi_terms = []
                
                # Add diffusion if enabled
                if self.diffusion_vf is not None:
                    bm = BrownianPath(
                        t0 - 1, t1 + 1, tol=dt0 / 2, shape=(2, self.num_neurons), key=bm_key
                    )
                    cvf = diffrax.ControlTerm(self.diffusion_vf, bm)
                    multi_terms.append(cvf)
                    
                # Add input spikes if provided
                if input_spike is not None:
                    assert input_weights is not None
                    input_st = SingleSpikeTrain(t0, t1, input_spike)
                    input_cvf = diffrax.ControlTerm(input_vf, input_st)
                    multi_terms.append(input_cvf)
                    
                # Combine all terms
                if multi_terms:
                    terms = diffrax.MultiTerm(terms, *multi_terms)
                    
                # Solve the ODE system
                sol = diffrax.diffeqsolve(
                    terms,
                    solver,
                    _t0,
                    t1,
                    dt0,
                    y0,
                    input_current,
                    throw=False,
                    stepsize_controller=stepsize_controller,
                    saveat=saveat,
                    event=event,
                    max_steps=max_steps,
                )
                
                # Process results
                assert sol.event_mask is not None
                event_mask = jnp.array(sol.event_mask)
                event_happened = jnp.any(event_mask)
                
                assert sol.ts is not None
                ts = sol.ts[0]
                _t1 = sol.ts[1]
                tevent = _t1[0]
                # If tevent > t1 we normalize to keep within range
                tevent = jnp.where(tevent > t1, tevent * (t1 / tevent), tevent)
                tevent = eqxi.error_if(tevent, jnp.isnan(tevent), "tevent is nan")
                
                assert sol.ys is not None
                ys = sol.ys[0]
                _y1 = sol.ys[1]
                yevent = _y1[0].reshape((self.num_neurons, 8))
                yevent = jnp.where(_t0 < t1, yevent, y0)
                yevent = eqxi.error_if(yevent, jnp.any(jnp.isnan(yevent)), "yevent is nan")
                yevent = eqxi.error_if(yevent, jnp.any(jnp.isinf(yevent)), "yevent is inf")
                
                # Determine which neurons spiked
                event_array = jnp.array(yevent[:, 0] > self.threshold - 1e-3)
                w_update_t = jnp.where(
                    jnp.tile(event_array, (self.num_neurons, 1)).T, w_update, 0.0
                ).T
                w_update_t = jnp.where(event_happened, w_update_t, 0.0)
                
                # Apply transition function to reset after spikes
                ytrans = trans_fn(yevent, event_array, trans_key)
                
                # Reshape outputs
                ys = jnp.transpose(ys, (1, 0, 2))
                
                return ts, ys, tevent, ytrans, event_array
                
            # Update all samples
            _ts, _ys, tevent, _ytrans, event_mask = update(
                state.t0, state.y0, trans_key, bm_key, input_spikes
            )
            
            # Increment spike counter
            num_spikes = state.num_spikes + 1
            
            ts = state.ts
            ts = ts.at[:, state.num_spikes].set(_ts)

            ys = state.ys
            ys = ys.at[:, state.num_spikes].set(_ys)

            event_types = state.event_types
            event_types = event_types.at[:, state.num_spikes].set(event_mask)

            tevents = state.tevents
            tevents = tevents.at[:, state.num_spikes].set(tevent)
            
            # Return new state
            new_state = NetworkState(
                ts=ts,
                ys=ys,
                tevents=tevents,
                t0=tevent,
                y0=_ytrans,
                num_spikes=num_spikes,
                event_mask=event_mask,
                event_types=event_types,
                key=new_key,
            )
            
            return new_state
            
        # Define stopping condition
        def stop_fn(state: NetworkState) -> bool:
            return (jnp.max(state.num_spikes) <= max_spikes) & (jnp.min(state.t0) < t1)
            
        # Run simulation
        final_state = eqxi.while_loop(
            stop_fn,
            body_fun,
            init_state,
            buffers=buffers,
            max_steps=max_spikes,
            kind="checkpointed",
        )
        
        # Prepare output solution
        ys = final_state.ys
        ts = final_state.ts
        spike_times = final_state.tevents
        spike_marks = final_state.event_types
        num_spikes = final_state.num_spikes
        
        sol = Solution(
            t1=t1,
            ys=ys,
            ts=ts,
            spike_times=spike_times,
            spike_marks=spike_marks,
            num_spikes=num_spikes,
            max_spikes=max_spikes,
        )
        
        return sol
############################
# Example usage

if __name__ == "__main__":
    # Define the network structure
    num_neurons = 3
    network = jnp.array([[0, 1, 0], 
                         [1, 0, 1], 
                         [1, 1, 0]], dtype=bool)
    
    weright_matrix = jnp.array([[0.0, 0.5, 0.0],
                                [0.2, 0.0, 0.1],
                                [0.5, 0.5, 0.0],])
    

    t_dur = 100
    I_stim = jnp.array([0.5, 1, 1])
    stim_start = jnp.zeros(num_neurons)
    stim_end = jnp.ones(num_neurons) * 50.0
    def input_current(t):
            stim = jnp.where((t>stim_start)&(t<stim_end), (I_stim/0.1), jnp.zeros_like(I_stim))
            return stim
    
    # Initialize the network
    network_model = MotoneuronNetwork(
        num_neurons=3,
        v_reset=-60.0,
        threshold=-37.0,
        w=weright_matrix,
        network=network
    )

    sol = network_model(
        input_current, 
        t0=0.0, 
        t1=100.0, 
        max_spikes=10, 
        num_samples=1, 
        key=jr.PRNGKey(0),
        dt0=0.01
    )

    print(sol.spike_marks)
    for k in range(sol.num_spikes):
        fired_mask   = sol.spike_marks[0, k]       # shape (neurons,)
        fired_neurons = jnp.where(fired_mask)[0]   # now non‐empty
        t_k          = sol.spike_times[0, k]
        print(f"Spike #{k} at t={t_k:.3f} ms by neurons {list(fired_neurons)}")
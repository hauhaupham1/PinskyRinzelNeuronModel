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
from jax._src.ad_util import stop_gradient_p  # pyright: ignore
from jax.interpreters import ad
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, Real
from diffrax import ODETerm, SaveAt, Dopri5, diffeqsolve, RESULTS, RecursiveCheckpointAdjoint, Event
from jax.ops import segment_sum


class Solution(eqx.Module):
    """Solution of the event-driven model."""

    t1: Real
    ys: Float[Array, "samples spikes neurons times 3"]
    ts: Float[Array, "samples spikes neurons times"]
    spike_times: Float[Array, "samples spikes"]
    spike_marks: Float[Array, "samples spikes neurons"]
    num_spikes: int
    max_spikes: int


def _build_w(w, network, key, minval, maxval):
    if w is not None:
        return w
    w_a = jr.uniform(key, network.shape, minval=minval, maxval=maxval)
    return w_a.at[network].set(0.0)


class NetworkState(eqx.Module):
    ts: Real[Array, "samples spikes times"]
    ys: Float[Array, "samples spikes neurons times 3"]
    tevents: eqxi.MaybeBuffer[Real[Array, "samples spikes"]]
    t0: Real[Array, " samples"]
    y0: Float[Array, "samples neurons 3"]
    num_spikes: int
    event_mask: Bool[Array, "samples neurons"]
    event_types: eqxi.MaybeBuffer[Bool[Array, "samples spikes neurons"]]

def buffers(state: NetworkState):
    assert type(state) is NetworkState
    return state.tevents, state.ts, state.ys, state.event_types

class MotoneuronNetwork(eqx.Module):
    num_neurons: Int
    w: Float[Array, "neurons neurons"]
    network: Bool[ArrayLike, "neurons neurons"] = eqx.field(static=True)
    threshold: float
    v_reset: float = -60.0
    drift_vf: Callable[..., Float[ArrayLike, "neurons 8"]]
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
    params: dict
    initial_state: Float[Array, "neurons 8"]
    cond_fn: List[Callable[..., Float]]

    def __init__(self, 
                 num_neurons: Int, 
                 w: Optional[Float[Array, "neurons neurons"]] = None,
                 network: Optional[Bool[ArrayLike, "neurons neurons"]] = None,
                 threshold: float = -20.0):
        
        self.num_neurons = num_neurons
        if network is None:
            network = np.full((num_neurons, num_neurons), False)

        w_key = jr.PRNGKey(0)
        wmin = 0.0
        wmax = 0.5

        self.w = _build_w(w, network, w_key, wmin, wmax)
        self.network = network

        self.C_m      = 2.0        # global_cm
        self.p        = 0.1           # pp
        self.g_c      = 0.4185837  # gc
        self.g_L_soma = 0.00036572208  # soma_g_pas
        self.g_L_dend = 1.0e-05  # dend_g_pas
        self.g_Na     = 0.2742448  # soma_gmax_Na
        self.g_DR     = 0.23978254  # soma_gmax_K
        self.g_Ca     = 0.0077317934  # soma_gmax_CaN
        self.g_AHP    = 0.005  # dend_gmax_KCa
        self.g_C      = 0.009327445  # soma_gmax_KCa
        # Reversal potentials
        self.E_L      = -62.0  # e_pas
        self.E_Na     = 60.0
        self.E_K      = -75.0
        self.E_Ca     = 80.0
        # Calcium dynamics
        self.f_Caconc     = 0.004  # soma_f_Caconc
        self.alpha_Caconc = 1.0  # soma_alpha_Caconc
        self.kCa_Caconc   = 8.0  # soma_kCa_Caconc

        self.params = {
            'C_m': self.C_m,
            'p': self.p,
            'g_c': self.g_c,
            'g_L_soma': self.g_L_soma,
            'g_L_dend': self.g_L_dend,
            'g_Na': self.g_Na,
            'g_DR': self.g_DR,
            'g_Ca': self.g_Ca,
            'g_AHP': self.g_AHP,
            'g_C': self.g_C,
            'E_L': self.E_L,
            'E_Na': self.E_Na,
            'E_K': self.E_K,
            'E_Ca': self.E_Ca,
            'f_Caconc': self.f_Caconc,
            'alpha_Caconc': self.alpha_Caconc,
            'kCa_Caconc': self.kCa_Caconc,
        }
        self.initial_state = self._initialize_state()
        self.threshold = threshold


        # def make_cond_fn(neuron_idx):
        #     def _cond_fn(t, y, args, **kwargs):
        #         # reshape to (state_vars, num_neurons) and detect when soma voltage crosses threshold
        #         y_reshaped = y.reshape(8, self.num_neurons)
        #         Vs = y_reshaped[0, neuron_idx]
        #         return Vs - self.threshold
        #     return _cond_fn
        
        # self.cond_fn = [make_cond_fn(n) for n in range(self.num_neurons)]
            
        

        def cond_fn(t, y, args, n, **kwargs):
            # y_reshaped = y.reshape(8, self.num_neurons)
            Vs = y[0, n]
            return Vs - self.threshold


        self.cond_fn = [
            ft.partial(cond_fn, n=n) for n in range(self.num_neurons)
        ]

        
        def alpha_m(Vs):
            """Na+ channel activation rate"""
            V1 = Vs + 46.9
            alpha = -0.32 * V1 / (jnp.exp(-V1 / 4.) - 1.)
            return alpha

        def beta_m(Vs):
            """Na+ channel deactivation rate"""
            V2 = Vs + 19.9
            beta = 0.28 * V2 / (jnp.exp(V2 / 5.) - 1.)
            return beta

        def alpha_h(Vs):
            """Na+ channel inactivation rate"""
            alpha = 0.128 * jnp.exp((-43. - Vs) / 18.)
            return alpha

        def beta_h(Vs):
            """Na+ channel deinactivation rate"""
            V5 = Vs + 20.
            beta = 4. / (1 + jnp.exp(-V5 / 5.))
            return beta

        def alpha_n(Vs):
            """K+ delayed rectifier activation rate"""
            V3 = Vs + 24.9
            alpha = -0.016 * V3 / (jnp.exp(-V3 / 5.) - 1)
            return alpha

        def beta_n(Vs):
            """K+ delayed rectifier deactivation rate"""
            V4 = Vs + 40.
            beta = 0.25 * jnp.exp(-V4 / 40.)
            return beta

        def alpha_s(Vd):
            """Ca2+ channel activation rate"""
            alpha = 1.6 / (1 + jnp.exp(-0.072 * (Vd-5.)))
            return alpha

        def beta_s(Vd):
            """Ca2+ channel deactivation rate"""
            V6 = Vd + 8.9
            beta = 0.02 * V6 / (jnp.exp(V6 / 5.) - 1.)
            return beta

        def alpha_c(Vd):
            """Ca-dependent K+ channel activation rate"""
            V7 = Vd + 53.5
            V8 = Vd + 50.
            return jnp.where(
                Vd <= -10,
                0.0527 * jnp.exp(V8/11. - V7/27.),
                2 * jnp.exp(-V7 / 27.)
            )

        def beta_c(Vd):
            """Ca-dependent K+ channel deactivation rate"""
            V7 = Vd + 53.5
            alpha_c_val = alpha_c(Vd)
            return jnp.where(
                Vd <= -10,
                2. * jnp.exp(-V7 / 27.) - alpha_c_val,
                0.
            )

        def alpha_q(Ca):
            """AHP K+ channel activation rate"""
            return jnp.minimum(0.00002*Ca, 0.01)

        def beta_q(Ca):
            """AHP K+ channel deactivation rate"""
            return 0.001

        def chi(Ca):
            """Ca-dependent activation function"""
            return jnp.minimum(Ca/250., 1.)

        def m_inf(Vs):
            """Na+ channel steady-state activation"""
            return alpha_m(Vs) / (alpha_m(Vs) + beta_m(Vs))
        
        def drift_vf(t, y, input_current):
            ic = input_current(t)

            @jax.vmap
            def _vf(y, ic):
                Vs, Vd, n, h, s, c, q, Ca = y
                I_leak_s = self.g_L_soma * (Vs - self.E_L)
                I_Na     = self.g_Na    * (m_inf(Vs)**2 * h) * (Vs - self.E_Na)
                I_DR     = self.g_DR    * n * (Vs - self.E_K)
                I_ds     = self.g_c     * (Vd - Vs)
                I_leak_d = self.g_L_dend* (Vd - self.E_L)
                I_Ca     = self.g_Ca    * (s**2) * (Vd - self.E_Ca)
                I_AHP    = self.g_AHP   * q * (Vd - self.E_K)
                I_C      = self.g_C     * c * chi(Ca) * (Vd - self.E_K)
                I_sd     = -I_ds

                dVsdt = (1.0/self.C_m)*( -I_leak_s - I_Na - I_DR + I_ds/self.p + ic)
                dVddt = (1.0/self.C_m)*( -I_leak_d - I_Ca - I_AHP - I_C + I_sd/(1.0-self.p))
                dhdt  = alpha_h(Vs)*(1-h) - beta_h(Vs)*h
                dndt  = alpha_n(Vs)*(1-n) - beta_n(Vs)*n
                dsdt  = alpha_s(Vd)*(1-s) - beta_s(Vd)*s
                dcdt  = alpha_c(Vd)*(1-c) - beta_c(Vd)*c
                dqdt  = alpha_q(Ca)*(1-q)  - beta_q(Ca)*q
                dCadt = -self.f_Caconc*I_Ca - self.alpha_Caconc*Ca/self.kCa_Caconc

                out = jnp.array([dVsdt, dVddt, dndt, dhdt, dsdt, dcdt, dqdt, dCadt])
                out = eqx.error_if(out, jnp.any(jnp.isnan(out)), "out is NaN")
                out = eqx.error_if(out, jnp.any(jnp.isinf(out)), "out is Inf")

                return out
            return _vf(y, ic)
        self.drift_vf = drift_vf

    def _initialize_state(self):
        """Initialize state variables for all neurons."""
        # Initial values from the PRmodel
        v_init = -60.0  
        
        # Create initial state arrays for all neurons
        Vs0 = jnp.ones(self.num_neurons) * v_init
        Vd0 = jnp.ones(self.num_neurons) * v_init
        n0 = jnp.ones(self.num_neurons) * 0.001
        h0 = jnp.ones(self.num_neurons) * 0.999
        s0 = jnp.ones(self.num_neurons) * 0.009
        c0 = jnp.ones(self.num_neurons) * 0.007
        q0 = jnp.ones(self.num_neurons) * 0.01
        Ca0 = jnp.ones(self.num_neurons) * 0.2
        
        # Stack into matrix format for diffrax: [8, num_neurons]
        return jnp.stack([Vs0, Vd0, n0, h0, s0, c0, q0, Ca0])
    

    # def vector_field(self, t, y, args):
    #     # unpack args and build an input_current function
    #     I_stim, stim_start, stim_end = args
    #     def input_current_fn(t):
    #         stim = jnp.where((t>stim_start)&(t<stim_end), I_stim/self.p, jnp.zeros_like(I_stim))
    #         return stim #+ jnp.dot(self.connections, stim) only apply the external current
    #     # delegate to vmapped drift_vf
    #     return self.drift_vf(t, y, input_current_fn)
    
    def __call__(self,
                input_current: Callable[..., Float[ArrayLike, "neurons"]],
                t_dur: Real,
                max_spikes: Int,
                num_samples: Int,
                *,
                y0: Optional[Float[Array, "neurons 8"]]=None,
                num_save: Int=2,
                dt0: Real=0.01,
                max_steps: Int = 1000,
        ):
        ic_shape = jax.eval_shape(input_current, 0.0)
        assert ic_shape.shape == (self.num_neurons,)

        t0 = 0.0
        y0 = y0 if y0 is not None else self.initial_state
        ys = jnp.full((num_samples, max_spikes, self.num_neurons, num_save, 3), jnp.inf)
        ts = jnp.full((num_samples, max_spikes, num_save), jnp.inf)
        tevents = jnp.full((num_samples, max_spikes), jnp.inf)
        num_spikes = 0
        event_mask = jnp.full((num_samples, self.num_neurons), False)
        event_types = jnp.full((num_samples, max_spikes, self.num_neurons), False)

        init_state = NetworkState(
            ts, ys, tevents, t0, y0, num_spikes, event_mask, event_types
        )
        stepsize_controller = diffrax.ConstantStepSize()
        vf = ODETerm(self.drift_vf)
        root_finder = optimistix.Newton(1e-2, 1e-2, optimistix.rms_norm)
        event = diffrax.Event(
            cond_fn=self.cond_fn, 
            root_finder=root_finder)
        solver = diffrax.Euler()
        w_update = self.w.at[self.network].set(0.0)

        @jax.vmap
        def trans_fn():
            Vs_out = self.v_reset
            Vd_out = self.v_reset
            n_out = 0.001
            h_out = 0.999
            s_out = 0.009
            c_out = 0.007
            q_out = 0.01
            Ca_out = 0.2

            return jnp.array([Vs_out, Vd_out, n_out, h_out, s_out, c_out, q_out, Ca_out])
        

        def body_fun(state: NetworkState) -> NetworkState:

            @jax.vmap
            def update(t0, y0):
                ts = jnp.where(
                    t0 < t_dur - (t_dur - t0) / (10 * num_save),
                    jnp.linspace(t0, t_dur, num_save),
                    jnp.full((num_save,), t0)
                )
                ts = eqxi.error_if(ts, ts[1:] < ts[: -1], "ts must be increasing")
                saveat_ts = diffrax.SubSaveAt(ts=ts)
                saveat_t1 = diffrax.SubSaveAt(t1=True)
                saveat = diffrax.SaveAt(subs=(saveat_ts, saveat_t1))
                terms = vf

                sol = diffeqsolve(
                    terms,
                    solver,
                    t0,
                    t_dur,
                    dt0,
                    y0,
                    input_current,
                    throw=False,
                    stepsize_controller=stepsize_controller,
                    saveat=saveat,
                    event=event,
                    max_steps=max_steps,
                )

                assert sol.event_mask is not None
                event_mask = jnp.array(sol.event_mask)
                event_happened = jnp.any(event_mask)

                assert sol.ts is not None
                ts = sol.ts[0]
                _t_dur = sol.ts[1]
                tevent = _t_dur[0]
                # If tevent > t1 we normalize to keep within range
                tevent = jnp.where(tevent > t_dur, tevent * (t_dur / tevent), tevent)
                tevent = eqxi.error_if(tevent, jnp.isnan(tevent), "tevent is nan")

                assert sol.ys is not None
                ys = sol.ys[0]
                _y1 = sol.ys[1]
                yevent = _y1[0].reshape((self.num_neurons, 8))
                yevent = jnp.where(t0 < t_dur, yevent, y0)
                yevent = eqxi.error_if(yevent, jnp.any(jnp.isnan(yevent)), "yevent is nan")
                yevent = eqxi.error_if(yevent, jnp.any(jnp.isinf(yevent)), "yevent is inf")

                event_array = jnp.array(yevent[:, 0] > self.threshold)
                w_update_t = jnp.where(
                    jnp.tile(event_array, (self.num_neurons, 1)).T, w_update, 0.0
                ).T
                w_update_t = jnp.where(event_happened, w_update_t, 0.0)
                ytrans = trans_fn(yevent, w_update_t)
                ys = jnp.transpose(ys, (1, 0, 2))

                return ts, ys, tevent, ytrans, event_array
            _ts, _ys, tevent, _ytrans, event_mask = update(
                state.t0, state.y0
            )
            num_spikes = state.num_spikes + 1
            ts = state.ts
            ts = ts.at[:, state.num_spikes].set(_ts)

            ys = state.ys
            ys = ys.at[:, state.num_spikes].set(_ys)

            event_types = state.event_types
            event_types = event_types.at[:, state.num_spikes].set(event_mask)

            tevents = state.tevents
            tevents = tevents.at[:, state.num_spikes].set(tevent)

            new_state = NetworkState(
                ts=ts,
                ys=ys,
                tevents=tevents,
                t0=tevent,
                y0=_ytrans,
                num_spikes=num_spikes,
                event_mask=event_mask,
                event_types=event_types,
            )

            return new_state
        
        def stop_fn(state: NetworkState) -> Bool:
            return (jnp.max(state.num_spikes) <= max_spikes) & (jnp.min(state.t0) < t_dur)
        
        final_state = eqxi.while_loop(
            stop_fn,
            body_fun,
            init_state,
            buffers=buffers,
            max_steps=max_spikes,
            kind="checkpointed",
        )
        ys = final_state.ys
        ts = final_state.ts
        spike_times = final_state.tevents
        spike_marks = final_state.event_types
        num_spikes = final_state.num_spikes
        sol = Solution(
            t1=t_dur,
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
    num_neurons = 10

    # Create the network
    network_model = MotoneuronNetwork(num_neurons=num_neurons)

    # Define a simple input current function
    I_stim = jnp.array([0.5] * num_neurons)  # Example input current
    stim_start = 0.5
    stim_end = 5

    def input_current(t):
        stim = jnp.where((t>stim_start)&(t<stim_end), I_stim/0.1, jnp.zeros_like(I_stim))
        return stim 

    # Run the simulation
    t_dur = 10
    max_spikes = 5
    num_samples = 1

    sol = network_model(input_current, t_dur, max_spikes, num_samples)
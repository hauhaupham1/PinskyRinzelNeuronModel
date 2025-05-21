"""
Pinsky-Rinzel Motoneuron Network Implementation

Credits:
    - Solution class and paths implementation by Christian Holberg
    - Network implementation inspired by snnax package from Christian Holberg
"""

import functools as ft
from typing import Any, Callable, List, Optional, Sequence

import diffrax
from diffrax import AbstractPath, BrownianIncrement, SpaceTimeLevyArea, VirtualBrownianTree
from diffrax._brownian.tree import _levy_diff, _make_levy_val
from diffrax._custom_types import RealScalarLike, levy_tree_transpose
from diffrax._misc import linear_rescale
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
from jaxtyping import Array, Bool, Float, Int, Real, PRNGKeyArray, PyTree

import matplotlib.pyplot as plt
from typing import Literal, Optional, Tuple, Union
import jax.tree_util as jtu
from typing_extensions import TypeAlias
_Spline: TypeAlias = Literal["sqrt", "quad", "zero"]


class Solution(eqx.Module):
    """Solution of the event-driven model."""

    t1: Real
    ys: Float[Array, "samples spikes neurons times state_vars"]  # state_vars can be 1 (Vs) or more
    ts: Float[Array, "samples spikes times"]
    spike_times: Float[Array, "samples spikes"]
    spike_marks: Float[Array, "samples spikes neurons"]
    num_spikes: int
    max_spikes: int
    stored_vars: tuple = eqx.field(static=True, default=('Vs',))
    # synaptic_I: Float[Array, "sample neurons"]

    def get_spikes(self, sample_idx=0):
        """
        Get the spike times for a specific sample.
        
        Args:
            sample_idx: Index of the sample to get spikes for
            
        Returns:
            Array of spike times for the sample
        """
        # Get the spike times for the sample
        spike_times = self.spike_times[sample_idx]
        
        # Filter out invalid (inf) spike times
        valid_spikes = jnp.isfinite(spike_times)
        return spike_times[valid_spikes]
        
    def get_voltages(self, sample_idx=0, neuron_idx=0):
        """
        Get the voltage traces for a specific sample and neuron.
        
        Args:
            sample_idx: Index of the sample to get voltages for
            neuron_idx: Index of the neuron to get voltages for
            
        Returns:
            Dictionary with 'times' and 'Vs' (somatic voltage) keys
        """
        # Extract time points and voltages
        all_times = []
        all_voltages = []
        
        # Number of segments (spike-to-spike) to process
        valid_spike_count = jnp.sum(jnp.isfinite(self.spike_times[sample_idx]))
        num_segments = min(valid_spike_count + 1, self.max_spikes)
        
        # Determine voltage index - for memory efficiency we might only store Vs
        voltage_idx = 0  # Default index for Vs (somatic voltage)
        
        # Iterate through each segment
        for spike_idx in range(num_segments):
            times = self.ts[sample_idx, spike_idx]
            
            # Handle case where we may only store somatic voltage
            if len(self.stored_vars) == 1 and self.stored_vars[0] == 'Vs':
                # If we're only storing Vs, we don't need to index the state dimension
                if self.ys.shape[-1] == 1:
                    voltages = self.ys[sample_idx, spike_idx, neuron_idx, :]
                else:
                    voltages = self.ys[sample_idx, spike_idx, neuron_idx, :, voltage_idx]
            else:
                # Otherwise find the correct index for Vs
                if 'Vs' in self.stored_vars:
                    voltage_idx = self.stored_vars.index('Vs')
                    
                voltages = self.ys[sample_idx, spike_idx, neuron_idx, :, voltage_idx]
            
            # Filter out invalid (inf) values
            valid_idx = jnp.isfinite(times) & jnp.isfinite(voltages)
            valid_times = times[valid_idx]
            valid_voltages = voltages[valid_idx]
            
            if len(valid_times) > 0:
                all_times.append(valid_times)
                all_voltages.append(valid_voltages)
        
        # Combine all segments
        if all_times:
            times_combined = jnp.concatenate(all_times)
            voltages_combined = jnp.concatenate(all_voltages)
            
            # Sort by time
            sort_idx = jnp.argsort(times_combined)
            times_sorted = times_combined[sort_idx]
            voltages_sorted = voltages_combined[sort_idx]
            
            return {
                'times': times_sorted,
                'Vs': voltages_sorted
            }
        
        return {
            'times': jnp.array([]),
            'Vs': jnp.array([])
        }
        
    def get_dendrite_voltages(self, sample_idx=0, neuron_idx=0):
        """
        Get the dendritic voltage traces for a specific sample and neuron.
        
        Args:
            sample_idx: Index of the sample to get voltages for
            neuron_idx: Index of the neuron to get voltages for
            
        Returns:
            Dictionary with 'times' and 'Vd' (dendritic voltage) keys
        """
        # Check if dendritic voltage is stored
        if 'Vd' not in self.stored_vars:
            return {
                'times': jnp.array([]),
                'Vd': jnp.array([])
            }
        
        # Find index of Vd in stored variables
        vd_index = self.stored_vars.index('Vd')
        
        # Extract time points and voltages
        all_times = []
        all_voltages = []
        
        # Number of segments (spike-to-spike) to process
        valid_spike_count = jnp.sum(jnp.isfinite(self.spike_times[sample_idx]))
        num_segments = min(valid_spike_count + 1, self.max_spikes)
        
        # Iterate through each segment
        for spike_idx in range(num_segments):
            times = self.ts[sample_idx, spike_idx]
            
            # Extract dendritic voltage at the right index
            if len(self.stored_vars) == 1 and self.stored_vars[0] == 'Vd':
                # Special case: if only storing Vd
                if self.ys.shape[-1] == 1:
                    voltages = self.ys[sample_idx, spike_idx, neuron_idx, :]
                else:
                    voltages = self.ys[sample_idx, spike_idx, neuron_idx, :, 0]  # Index 0 if only Vd
            else:
                voltages = self.ys[sample_idx, spike_idx, neuron_idx, :, vd_index]
            
            # Filter out invalid (inf) values
            valid_idx = jnp.isfinite(times) & jnp.isfinite(voltages)
            valid_times = times[valid_idx]
            valid_voltages = voltages[valid_idx]
            
            if len(valid_times) > 0:
                all_times.append(valid_times)
                all_voltages.append(valid_voltages)
        
        # Combine all segments
        if all_times:
            times_combined = jnp.concatenate(all_times)
            voltages_combined = jnp.concatenate(all_voltages)
            
            # Sort by time
            sort_idx = jnp.argsort(times_combined)
            times_sorted = times_combined[sort_idx]
            voltages_sorted = voltages_combined[sort_idx]
            
            return {
                'times': times_sorted,
                'Vd': voltages_sorted
            }
        
        return {
            'times': jnp.array([]),
            'Vd': jnp.array([])
        }


def plottable_paths(
    sol: Solution,
) -> Tuple[Real[Array, "samples times"], Float[Array, "samples neurons times 3"]]:
    """Takes an instance of `Solution` from `SpikingNeuralNet.__call__(...)` and outputs the times
    and values of the internal neuron states in a plottable format.

    **Arguments**:

    - `sol`: An instance of `Solution` as returned from `SpikingNeuralNet.__call__(...)`.

    **Returns**:

    - `ts`: The time axis of the path of shape `(samples, times)`.
    - `ys`: The values of the internal state of the neuron of shape `(samples, neurons, times, 3)`.
    """

    @jax.vmap
    def _plottable_neuron(ts, ys):
        t0 = ts[0, 0]
        t1 = sol.t1
        _, neurons, times, _ = ys.shape
        ys = ys.transpose((1, 0, 2, 3))
        ts_out = jnp.linspace(t0, t1, times)
        ts_flat = ts.flatten()
        ys_flat = ys.reshape((neurons, -1, 3))
        sort_idx = jnp.argsort(ts_flat)
        ts_flat = ts_flat[sort_idx]
        ys_flat = ys_flat[:, sort_idx, :]
        idx = jnp.searchsorted(ts_flat, ts_out)
        ys_out = ys_flat[:, idx, :]
        return ts_out, ys_out

    return _plottable_neuron(sol.ts, sol.ys)


def interleave(arr1: Array, arr2: Array) -> Array:
    out = jnp.empty((arr1.size + arr2.size,), dtype=arr1.dtype)
    out = out.at[0::2].set(arr2)
    out = out.at[1::2].set(arr1)
    return out


def marcus_lift(
    t0: RealScalarLike,
    t1: RealScalarLike,
    spike_times: Float[Array, " max_spikes"],
    spike_mask: Float[Array, "max_spikes num_neurons"],
) -> Float[Array, " 2_max_spikes"]:
    """Lifts a spike train to a discretisation of the Marcus lift
    (with time augmentation).

    **Arguments**:

    - `t0`: The start time of the path.
    - `t1`: The end time of the path.
    - `spike_times`: The times of the spikes.
    - `spike_mask`: A mask indicating the corresponding spiking neuron.

    **Returns**:

    - An array of shape `(2 * max_spikes, num_neurons + 1)` representing the Marcus lift.
    """
    num_neurons = spike_mask.shape[1]
    finite_spikes = jnp.where(jnp.isfinite(spike_times), spike_times, t1).reshape((-1, 1))
    spike_cumsum = jnp.cumsum(spike_mask, axis=0)
    # last_spike_time = jnp.max(jnp.where(spike_times < t1, spike_times, -jnp.inf))
    spike_cumsum_shift = jnp.roll(spike_cumsum, 1, axis=0)
    spike_cumsum_shift = spike_cumsum_shift.at[0, :].set(
        jnp.zeros(num_neurons, dtype=spike_cumsum_shift.dtype)
    )
    arr1 = jnp.hstack([finite_spikes, spike_cumsum])
    arr2 = jnp.hstack([finite_spikes, spike_cumsum_shift])
    out = jax.vmap(interleave, in_axes=1)(arr1, arr2).T
    # Makes sure the path starts at t0
    out = jnp.roll(out, 1, axis=0)
    out = out.at[0, :].set(jnp.insert(jnp.zeros(num_neurons), 0, t0))
    # time_capped = jnp.where(out[:, 0] < t1, out[:, 0], last_spike_time)
    # out = out.at[:, 0].set(time_capped)
    return out


@eqx.filter_jit
def cap_fill_ravel(ts, ys, spike_cap=10):
    # Cap the number of spikes
    ys_capped = ys[:spike_cap]
    ts_capped = ts[:spike_cap]
    spikes, neurons, times, _ = ys_capped.shape

    # Fill up infs
    idx = ts_capped > ts_capped[:, -1, None]
    idx_y = jnp.tile(idx[:, None, :, None], (1, neurons, 1, 3))
    ts_capped = jnp.where(idx, ts_capped[:, -1, None], ts_capped)
    ys_capped = jnp.where(idx_y, ys_capped[:, :, -1, None, :], ys_capped)

    xs = (ts_capped, ys_capped)
    carry_ys = jnp.zeros((neurons, times, 3))
    carr_ts = jnp.array(0.0)
    carry = (carr_ts, carry_ys)

    def _fill(carry, x):
        _ts, _ys = x
        carry_ts, carry_ys = carry
        _ys_fill_val = jnp.tile(_ys[:, None, -1], (1, times, 1))
        _ts_fill_val = _ts[-1]
        _ys_out = jnp.where(jnp.isinf(_ys), _ys_fill_val, _ys)
        _ts_out = jnp.where(jnp.isinf(_ts), _ts_fill_val, _ts)
        assert isinstance(_ys_out, Array)
        _ys_all_inf = jnp.all(jnp.isinf(_ys_out))
        _ts_all_inf = jnp.all(jnp.isinf(_ts_out))
        _ys_out = jnp.where(_ys_all_inf, carry_ys, _ys_out)
        _ts_out = jnp.where(_ts_all_inf, carry_ts, _ts_out)
        assert isinstance(_ys_out, Array)
        new_carry_ys = jnp.tile(_ys_out[:, None, -1], (1, times, 1))
        new_carry_ts = _ts_out[-1]
        new_carry = (new_carry_ts, new_carry_ys)
        out = (_ts_out, _ys_out)
        return new_carry, out

    _, xs_filled_capped = jax.lax.scan(_fill, carry, xs=xs)
    ts_filled_capped, ys_filled_capped = xs_filled_capped

    # Ravel out the "spikes" dimension
    # (spikes, neurons, times, 3) -> (neurons, spikes, times, 3)
    ys_filled_capped = jnp.transpose(ys_filled_capped, (1, 0, 2, 3))
    # (spikes, neurons, times, 3) -> (neurons, spikes*times, 3)
    ys_filled_capped_ravelled = ys_filled_capped.reshape((neurons, -1, 3))
    ts_filled_capped_ravelled = jnp.ravel(ts_filled_capped)
    return ts_filled_capped_ravelled, ys_filled_capped_ravelled


class SpikeTrain(AbstractPath):
    t0: Real
    t1: Real
    num_spikes: int
    spike_times: Array
    spike_cumsum: Array
    num_neurons: int

    def __init__(self, t0, t1, spike_times, spike_mask):
        max_spikes, num_neurons = spike_mask.shape
        self.num_neurons = num_neurons
        self.t0 = t0
        self.t1 = t1
        self.num_spikes = spike_times.shape[0]
        self.spike_times = jnp.insert(spike_times, 0, t0)
        self.spike_cumsum = jnp.cumsum(
            jnp.insert(spike_mask, 0, jnp.full_like(spike_mask[0], False), axis=0), axis=0
        )

    def evaluate(self, t0: Real, t1: Optional[Real] = None, left: Optional[bool] = True) -> Array:
        del left
        if t1 is not None:
            return self.evaluate(t1 - t0)
        idx = jnp.searchsorted(self.spike_times, t0)
        idx = jnp.where(idx > 0, idx - 1, idx)
        out = jax.lax.dynamic_slice(self.spike_cumsum, (idx, 0), (self.num_neurons, 1))[:, 0]
        return out


class SingleSpikeTrain(AbstractPath):
    t0: Real
    t1: Real
    spike_times: Array

    def evaluate(self, t0: Real, t1: Optional[Real] = None, left: Optional[bool] = True) -> Array:
        del left
        if t1 is not None:
            return self.evaluate(t1 - t0)
        return jnp.where(self.spike_times >= t0, 1.0, 0.0)


# A version of VirtualBrownianTree that will not throw an error when differentiated
class BrownianPath(VirtualBrownianTree):
    @eqxi.doc_remove_args("_spline")
    def __init__(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        tol: RealScalarLike,
        shape: Union[tuple[int, ...], PyTree[jax.ShapeDtypeStruct]],
        key: PRNGKeyArray,
        levy_area: type[Union[BrownianIncrement, SpaceTimeLevyArea]] = BrownianIncrement,
        _spline: _Spline = "sqrt",
    ):
        super().__init__(t0, t1, tol, shape, key, levy_area, _spline)

    @eqx.filter_jit
    def evaluate(
        self,
        t0: RealScalarLike,
        t1: Optional[RealScalarLike] = None,
        left: bool = True,
        use_levy: bool = False,
    ) -> Union[PyTree[Array], BrownianIncrement, SpaceTimeLevyArea]:
        t0 = jax.lax.stop_gradient(t0)
        # map the interval [self.t0, self.t1] onto [0,1]
        t0 = linear_rescale(self.t0, t0, self.t1)
        levy_0 = self._evaluate(t0)
        if t1 is None:
            levy_out = levy_0
            levy_out = jtu.tree_map(_make_levy_val, self.shape, levy_out)
        else:
            t1 = jax.lax.stop_gradient(t1)
            # map the interval [self.t0, self.t1] onto [0,1]
            t1 = linear_rescale(self.t0, t1, self.t1)
            levy_1 = self._evaluate(t1)
            levy_out = jtu.tree_map(_levy_diff, self.shape, levy_0, levy_1)

        levy_out = levy_tree_transpose(self.shape, levy_out)
        # now map [0,1] back onto [self.t0, self.t1]
        levy_out = self._denormalise_bm_inc(levy_out)
        assert isinstance(levy_out, (BrownianIncrement, SpaceTimeLevyArea))
        return levy_out if use_levy else levy_out.W



# Work around JAX issue #22011, similar to what's in snn.py
def stop_gradient_transpose(ct, x):
    return (ct,)

ad.primitive_transposes[stop_gradient_p] = stop_gradient_transpose


class NetworkState(eqx.Module):
    ts: Real[Array, "samples spikes times"]
    ys: Float[Array, "samples spikes neurons times state_vars"]  # Flexible state variables
    tevents: eqxi.MaybeBuffer[Real[Array, "samples spikes"]]
    t0: Real[Array, " samples"]
    y0: Float[Array, "samples neurons 8"]  # Still need full state for dynamics
    #Add synaptic_currents so we can directly modify when a neuron spikes
    synaptic_I: Float[Array, "samples neurons"]
    num_spikes: int
    event_mask: Bool[Array, "samples neurons"]
    event_types: eqxi.MaybeBuffer[Bool[Array, "samples spikes neurons"]]
    key: Any
    stored_vars: tuple = eqx.field(static=True, default=('Vs',))


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
    # input_current: Float[ArrayLike, "neurons |"]
    # stim_start: Float[ArrayLike, "neurons |"]
    # stim_end: Float[ArrayLike, "neurons |"]
    # Optional diffusion parameters
    sigma: Optional[Float[ArrayLike, "2 2"]]
    diffusion_vf: Optional[Callable[..., Float[ArrayLike, "neurons 8 2 neurons"]]]
    cond_fn: List[Callable[..., Float]]

    def __init__(
        self,
        num_neurons: Int,
        # input_current: Float = 1.0,
        # stim_start: Float = 0.0,
        # stim_end: Float = 50.0,
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
            network = np.full((num_neurons, num_neurons), 0)

        self.w = _build_w(w, network, w_key, wmin, wmax)
        self.network = network

        if read_out_neurons is None:
            read_out_neurons = []

        self.read_out_neurons = read_out_neurons

        #ensure that the input_current, stim_start, and stim_end are arrays of length num_neurons
        # self.input_current = jnp.asarray(input_current)
        # self.stim_start = jnp.asarray(stim_start)
        # self.stim_end = jnp.asarray(stim_end)


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
        store_vars: tuple = ('Vs',),  # Which state variables to store (default: only somatic voltage)
        memory_efficient: bool = True,  # Whether to use memory-efficient storage
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
        # ic_shape = jax.eval_shape(input_current, 0.0)
        # assert ic_shape.shape == (self.num_neurons,)
        

        #check is store_vars is a tuple
        if not isinstance(store_vars, tuple):
            raise ValueError("store_vars must be a tuple of state variable names.")
        
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
            
        # Determine which state variables to store and their indices
        if memory_efficient:
            # Map state variable names to their indices
            state_var_indices = {
                'Vs': 0,  # Somatic voltage
                'Vd': 1,  # Dendritic voltage
                'n': 2,   # n gate
                'h': 3,   # h gate
                's': 4,   # s gate
                'c': 5,   # c gate
                'q': 6,   # q gate
                'Ca': 7,  # Calcium concentration
            }
            
            # Validate requested state variables
            for var in store_vars:
                if var not in state_var_indices:
                    raise ValueError(f"Unknown state variable: {var}")
            
            # Number of state variables to store
            num_state_vars = len(store_vars)
            
            # Initialize storage with only the requested state variables
            ys = jnp.full((num_samples, max_spikes, self.num_neurons, num_save, num_state_vars), jnp.inf)
        else:
            # Store all 8 state variables (original behavior)
            store_vars = ('Vs', 'Vd', 'n', 'h', 's', 'c', 'q', 'Ca')
            ys = jnp.full((num_samples, max_spikes, self.num_neurons, num_save, 8), jnp.inf)
        
        # Initialize other storage arrays
        ts = jnp.full((num_samples, max_spikes, num_save), jnp.inf)
        tevents = jnp.full((num_samples, max_spikes), jnp.inf)
        num_spikes = 0
        event_mask = jnp.full((num_samples, self.num_neurons), False)
        event_types = jnp.full((num_samples, max_spikes, self.num_neurons), False)

        #Initialize synaptic_I
        init_synaptic_I = jnp.zeros((num_samples, self.num_neurons))
        
        # Create initial state
        init_state = NetworkState(
            ts=ts,
            ys=ys, 
            tevents=tevents,
            t0=_t0, 
            y0=y0, 
            num_spikes=num_spikes, 
            event_mask=event_mask, 
            event_types=event_types, 
            key=key,
            synaptic_I=init_synaptic_I,
            stored_vars=store_vars,
        )
        
        # Setup diffrax solver
        stepsize_controller = diffrax.ConstantStepSize()
        vf = diffrax.ODETerm(self.vector_field)
        root_finder = optimistix.Newton(1e-6, 1e-6, optimistix.rms_norm)
        event = diffrax.Event(self.cond_fn, root_finder)
        solver = diffrax.Euler()
        # w_update = self.w.at[self.network].set(0.0)
        #using jnp.where
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

            # Update synaptic currents
            pre_synaptic_I = state.synaptic_I
            
            @jax.vmap
            def update(_t0, y0, trans_key, bm_key, input_spike, _pre_synaptic_I):
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
                saveat = diffrax.SaveAt(subs=[saveat_ts, saveat_t1])
                
                # Set up terms for solver
                current_ode_args = {
                      "synaptic_I": _pre_synaptic_I, # As before
                      "input_current": input_current, # Passed into __call__
                  }
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
                    args=current_ode_args,
                    throw=False,
                    stepsize_controller=stepsize_controller,
                    saveat=saveat,
                    event=event,
                    max_steps=max_steps,
                )

                # jax.debug.print("t0={t} tevent={tev} result={res} mask={m}",
                #                 t=_t0, tev=sol.ts[1][0], res=sol.result, m=sol.event_mask)
                
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
                full_ys = sol.ys[0]  # Get the full state array from solver
                _y1 = sol.ys[1]
                
                # Get event state (all 8 variables needed for next iteration)
                yevent = _y1[0].reshape((self.num_neurons, 8))
                yevent = jnp.where(_t0 < t1, yevent, y0)
                yevent = eqxi.error_if(yevent, jnp.any(jnp.isnan(yevent)), "yevent is nan")
                yevent = eqxi.error_if(yevent, jnp.any(jnp.isinf(yevent)), "yevent is inf")
                
                # Determine which neurons spiked (always use Vs for spike detection)
                event_array = jnp.array(yevent[:, 0] > self.threshold - 1e-3)
                # event_array = jnp.array(sol.event_mask)
                w_update_t = jnp.where(
                    jnp.tile(event_array, (self.num_neurons, 1)).T, w_update, 0.0
                ).T
                w_update_t = jnp.where(event_happened, w_update_t, 0.0)
                
                # Apply transition function to reset after spikes
                ytrans = trans_fn(yevent, event_array, trans_key)

                # Update synaptic currents
                # new_dI = jnp.dot(event_array.astype(self.w.dtype), self.w) / self.p
                new_dI = jnp.sum(w_update_t, axis=-1) / self.p
                
                # Reshape the full outputs
                full_ys = jnp.transpose(full_ys, (1, 0, 2))
                
                # Memory-efficient storage: extract only the requested state variables
                if memory_efficient:
                    # Map state variable names to their indices
                    state_var_indices = {
                        'Vs': 0,  # Somatic voltage
                        'Vd': 1,  # Dendritic voltage
                        'n': 2,   # n gate
                        'h': 3,   # h gate
                        's': 4,   # s gate
                        'c': 5,   # c gate
                        'q': 6,   # q gate
                        'Ca': 7,  # Calcium concentration
                    }
                    
                    # Extract only the requested variables by index
                    indices = [state_var_indices[var] for var in store_vars]
                    
                    # If only storing one variable
                    if len(indices) == 1:
                        ys = full_ys[..., indices[0]]
                        # Add a dimension to maintain expected shape
                        ys = jnp.expand_dims(ys, axis=-1)
                    else:
                        # Extract multiple variables
                        ys = jnp.stack([full_ys[..., idx] for idx in indices], axis=-1)
                else:
                    # Return all state variables (original behavior)
                    ys = full_ys
                
                return ts, ys, tevent, ytrans, event_array, new_dI
                
            # Update all samples
            _ts, _ys, tevent, _ytrans, event_mask, new_synaptic_I = update(
                state.t0, state.y0, trans_key, bm_key, input_spikes, pre_synaptic_I
            )
            # jax.debug.print("pre_synaptic_I: {x}", x=pre_synaptic_I)
            # jax.debug.print("new_synaptic_I: {x}", x=new_synaptic_I)

        
            
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
                synaptic_I=new_synaptic_I,
                stored_vars=store_vars,
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

        #synaptic_I
        # after_spike_I = final_state.synaptic_I
        
        sol = Solution(
            t1=t1,
            ys=ys,
            ts=ts,
            spike_times=spike_times,
            spike_marks=spike_marks,
            num_spikes=num_spikes,
            max_spikes=max_spikes,
            stored_vars=store_vars,  # Include which variables were stored
            # synaptic_I=after_spike_I,
        )
        
        return sol
    
def plot_simulation_results(
    sol: Solution,
    neurons_to_plot: Optional[Sequence[Int]] = None,
    plot_spikes: bool = True,
    plot_dendrite: bool = True  # Parameter to control Vd plotting
):
    """
    Plot voltage traces from a simulation solution.
    
    Args:
        sol: Solution object containing simulation results
        neurons_to_plot: Specific neurons to plot. If None, plots all neurons
        plot_spikes: Whether to mark spike times on the plot
        plot_dendrite: Whether to plot dendritic voltage traces (if available)
    """
    num_samples = sol.ys.shape[0]
    num_neurons = sol.ys.shape[2]  # Extract neuron count from solution
    
    # Check if dendrite plotting is possible
    if plot_dendrite and hasattr(sol, 'stored_vars'):
        if 'Vd' not in sol.stored_vars:
            plot_dendrite = False
            print("Warning: Dendritic voltage (Vd) not stored, dendrite plotting disabled")
    
    if neurons_to_plot is None:
        neurons_to_plot = range(num_neurons)
    elif not isinstance(neurons_to_plot, Sequence) or not all(isinstance(n, int) for n in neurons_to_plot):
        raise ValueError("neurons_to_plot must be a sequence of integers or None.")

    for sample_idx in range(num_samples):
        plt.figure(figsize=(12, 6))
        print(f"Processing plot for Sample {sample_idx}...")
        
        plotted_lines = []
        plotted_labels = []

        # Iterate through the neurons requested for plotting
        for neuron_index in neurons_to_plot:
            if neuron_index < 0 or neuron_index >= num_neurons:
                print(f"Warning: Neuron index {neuron_index} out of range (0-{num_neurons-1}). Skipping.")
                continue

            # Get somatic voltage using the Solution's method
            v_data = sol.get_voltages(sample_idx, neuron_index)
            
            if len(v_data['times']) > 0:
                # Plot soma voltage
                line_soma, = plt.plot(v_data['times'], v_data['Vs'], 
                                     label=f"Neuron {neuron_index} (Soma)",
                                     color=f"C{neuron_index}")
                plotted_lines.append(line_soma)
                plotted_labels.append(f"Neuron {neuron_index} (Soma)")
                
                # Plot dendrite voltage if requested
                if plot_dendrite:
                    # Get dendritic voltage using the Solution's method
                    vd_data = sol.get_dendrite_voltages(sample_idx, neuron_index)
                    if len(vd_data['times']) > 0:
                        line_dend, = plt.plot(vd_data['times'], vd_data['Vd'],
                                            label=f"Neuron {neuron_index} (Dendrite)",
                                            linestyle='--', color=f"C{neuron_index}")
                        plotted_lines.append(line_dend)
                        plotted_labels.append(f"Neuron {neuron_index} (Dendrite)")
            else:
                print(f"No valid data found for Neuron {neuron_index} in Sample {sample_idx}.")

        # Plot spike markers
        if plot_spikes:
            # Use get_spikes method to get valid spike times
            valid_spike_times = sol.get_spikes(sample_idx)
            added_spike_legend = False
            if len(plotted_lines) > 0:
                min_t, max_t = plt.xlim() # Get current plot time range
                for spike_t in valid_spike_times:
                    if spike_t >= min_t and spike_t <= max_t:
                        label = 'Spike' if not added_spike_legend else ""
                        plt.axvline(spike_t, color='r', linestyle='--', alpha=0.6, label=label)
                        added_spike_legend = True

        # Finalize plot for the current sample
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.title(f"Sample {sample_idx}: Neuron Voltage Traces")
        
        if plotted_lines:
            custom_legend_handles = plotted_lines
            custom_legend_labels = plotted_labels
            if plot_spikes and added_spike_legend:
                from matplotlib.lines import Line2D
                spike_legend_entry = Line2D([0], [0], color='r', linestyle='--', alpha=0.6, label='Spike')
                custom_legend_handles.append(spike_legend_entry)
                custom_legend_labels.append('Spike')

            plt.legend(handles=custom_legend_handles, labels=custom_legend_labels)

        plt.grid(True)
        plt.show() # Show the plot for the current sample


# Add a wrapper method to the MotoneuronNetwork class for compatibility
def _motoneuron_plot_wrapper(self, sol, **kwargs):
    """
    Wrapper for the standalone plotting function to maintain backward compatibility.
    """
    return plot_simulation_results(sol, **kwargs)


# Add the wrapper method to MotoneuronNetwork class
MotoneuronNetwork.plot_simulation_results = _motoneuron_plot_wrapper
############################
# Example usage

if __name__ == "__main__":
    # Define the network structure
    num_neurons = 5
    network = jnp.array([[0, 1, 1, 1, 1],
                         [1, 0, 1, 1, 1],
                         [1, 1, 0, 1, 1],
                         [1, 1, 1, 0, 1],
                         [1, 1, 1, 1, 0]])
    weight = jnp.array([[0, 0.5, 0.5, 0.5, 0.5],
                         [0, 0, 0.5, 0.5, 0.5],
                         [0, 0, 0, 0.5, 0.5],
                         [0, 0, 0, 0, 0.5],
                         [0, 0, 0.5, 0.5, 0]])
            
    #I stim is array of shape (num_neurons,) with values of 1
    I_stim = jnp.zeros(num_neurons)
    #only stimulate neuron 0
    I_stim = I_stim.at[0].set(3.5)
    I_stim = I_stim.at[1].set(1.5)
    I_stim = I_stim.at[2].set(2)
    I_stim = I_stim.at[3].set(2.5)
    I_stim = I_stim.at[4].set(1)
    stim_start = jnp.zeros(num_neurons)
    stim_end = jnp.ones(num_neurons) * 10
    
    # Initialize the network
    network_model = MotoneuronNetwork(
        num_neurons=num_neurons,
        network=network,
        w=weight,
        v_reset=-60.0,
        threshold=-37.0,
        input_current= I_stim,
        stim_start=stim_start,
        stim_end=stim_end,
    )

    sol = network_model(
        t0=0.0, 
        t1=20, 
        max_spikes=20, 
        num_samples=1, 
        key=jr.PRNGKey(0),
        dt0=0.01,
        num_save=50,
        max_steps=1500,
    )

    # Plot the simulation results
    print(sol.get_spikes())
    # print(sol.num_spikes)
    network_model.plot_simulation_results(sol, neurons_to_plot=[0, 1, 2, 3, 4], plot_spikes=True, plot_dendrite=False)
    # print(network_model.input_current)

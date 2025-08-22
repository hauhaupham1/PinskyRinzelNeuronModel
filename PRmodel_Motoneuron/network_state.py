"""

Credits:
    - NetworkState's implementation inspired by Christian Holberg
"""

from typing import Any, Optional
import equinox as eqx
import equinox.internal as eqxi
from jax._src.ad_util import stop_gradient_p
from jax.interpreters import ad
from jaxtyping import Array, Bool, Float, Real, PyTree
from typing import Optional


# Work around JAX issue #22011, similar to what's in snn.py
def stop_gradient_transpose(ct, x):
    return (ct,)


ad.primitive_transposes[stop_gradient_p] = stop_gradient_transpose


class NetworkState(eqx.Module):
    ts: Real[Array, "samples spikes times"]
    ys: Float[
        Array, "samples spikes neurons times state_vars"
    ]  # Flexible state variables
    tevents: eqxi.MaybeBuffer[Real[Array, "samples spikes"]]
    t0: Real[Array, " samples"]
    y0: Float[Array, "samples neurons 8"]  # Still need full state for dynamics
    # Add synaptic_currents so we can directly modify when a neuron spikes
    synaptic_I: Float[Array, "samples neurons"]
    num_spikes: int
    event_mask: Bool[Array, "samples neurons"]
    event_types: eqxi.MaybeBuffer[Bool[Array, "samples spikes neurons"]]
    key: Any
    stored_vars: tuple = eqx.field(static=True, default=("Vs",))
    spike_only: bool = eqx.field(static=True, default=False)
    discard_voltage_between_spikes: bool = eqx.field(static=True, default=False)
    # For chunk continuity
    solver_state: Optional[PyTree] = None
    controller_state: Optional[PyTree] = None
    made_jump: Optional[Bool[Array, "samples"]] = None

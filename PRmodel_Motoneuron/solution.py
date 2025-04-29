import equinox as eqx
from jaxtyping import Array, Float, Real


class Solution(eqx.Module):
    """Solution of the event-driven model."""

    t1: Real
    ys: Float[Array, "samples spikes neurons times 8"]
    ts: Float[Array, "samples spikes times"]
    spike_times: Float[Array, "samples spikes"]
    spike_marks: Float[Array, "samples spikes neurons"]
    num_spikes: int
    max_spikes: int
    # synaptic_I: Float[Array, "sample neurons"]

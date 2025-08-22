from typing import Callable, Optional
from .Network import MotoneuronNetwork
from .solution import Solution
from jax import random as jr
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, Float, Int, Real
import gc
import jax


# Chunked simulation implementation (Optimized for speed, keeping NumPy conversion for memory)
def run_chunked_simulation(
    network: MotoneuronNetwork,
    input_current: Callable[..., Float[Array, " neurons"]],
    t0: Real,
    t1: Real,
    chunk_duration: Real = 10.0,
    max_spikes_per_chunk: Int = 100,
    num_samples: Int = 1,
    num_saves_per_chunk: Optional[Int] = 2,
    key=None,
    dt0: Real = 0.01,
    max_steps_per_chunk: Int = 1000,
    spike_only: bool = True,  # Default to spike-only for backward compatibility
    store_vars: tuple = ("Vs",),  # Which state variables to store
    memory_efficient: bool = True,  # Whether to use memory-efficient storage
    max_total_spikes: Optional[Int] = None,  # Fix for Marcus lift shape consistency
    **kwargs,
) -> Solution:
    if key is None:
        key = jr.PRNGKey(0)
    total_duration = t1 - t0
    num_chunks = int(jnp.ceil(total_duration / chunk_duration))

    current_y0_np = None
    current_synaptic_I_np = None
    current_solver_state_jax = None
    current_controller_state_jax = None
    current_made_jump_jax = None

    all_spike_times_np_per_sample = [[] for _ in range(num_samples)]
    all_spike_marks_np_per_sample = [[] for _ in range(num_samples)]

    # Add storage for voltage data when not in spike_only mode
    all_ys_np_per_sample = [[] for _ in range(num_samples)] if not spike_only else None
    all_ts_np_per_sample = [[] for _ in range(num_samples)] if not spike_only else None

    for chunk_idx in range(num_chunks):
        y0_jax = jnp.array(current_y0_np) if current_y0_np is not None else None
        synaptic_I0_jax = (
            jnp.array(current_synaptic_I_np)
            if current_synaptic_I_np is not None
            else None
        )

        chunk_t0_loop = t0 + chunk_idx * chunk_duration
        chunk_t1_loop = min(chunk_t0_loop + chunk_duration, t1)

        if chunk_t0_loop >= chunk_t1_loop:
            continue

        # Determine if this is the final chunk
        is_final_chunk = chunk_t1_loop >= t1

        key, chunk_key = jr.split(key)
        chunk_sol: Solution = network(
            input_current=input_current,
            t0=chunk_t0_loop,
            t1=chunk_t1_loop,
            max_spikes=max_spikes_per_chunk,
            num_samples=num_samples,
            key=chunk_key,
            num_save=num_saves_per_chunk,
            y0=y0_jax,
            synaptic_I0=synaptic_I0_jax,
            dt0=dt0,
            max_steps=max_steps_per_chunk,
            spike_only=spike_only,
            store_vars=store_vars,
            memory_efficient=memory_efficient,
            is_final_simulation=is_final_chunk,  # Only allow end spike for final chunk
            solver_state0=current_solver_state_jax,
            controller_state0=current_controller_state_jax,
            made_jump0=current_made_jump_jax,
            **kwargs,
        )
        for sample_idx in range(num_samples):
            s_times_jax = chunk_sol.spike_times[sample_idx]
            s_marks_jax = chunk_sol.spike_marks[sample_idx]

            valid_mask_jax = jnp.isfinite(s_times_jax)

            valid_s_times_jax = s_times_jax[valid_mask_jax]
            valid_s_marks_jax = s_marks_jax[valid_mask_jax]

            # Only call block_until_ready if not in gradient computation
            if hasattr(valid_s_times_jax, "block_until_ready"):
                valid_s_times_jax.block_until_ready()
                valid_s_marks_jax.block_until_ready()

            if valid_s_times_jax.shape[0] > 0:
                all_spike_times_np_per_sample[sample_idx].append(
                    np.array(valid_s_times_jax)
                )
                all_spike_marks_np_per_sample[sample_idx].append(
                    np.array(valid_s_marks_jax)
                )

            # Collect voltage data if not in spike_only mode
            if not spike_only:
                ys_jax = chunk_sol.ys[sample_idx]
                ts_jax = chunk_sol.ts[sample_idx]
                if hasattr(ys_jax, "block_until_ready"):
                    ys_jax.block_until_ready()
                    ts_jax.block_until_ready()
                all_ys_np_per_sample[sample_idx].append(np.array(ys_jax))
                all_ts_np_per_sample[sample_idx].append(np.array(ts_jax))

        # --- Update inter-chunk states (convert to NumPy) ---
        if chunk_sol.final_state is not None:
            final_state_jax = chunk_sol.final_state
            if hasattr(final_state_jax, "block_until_ready"):
                final_state_jax = final_state_jax.block_until_ready()
            current_y0_np = np.array(final_state_jax)
            del final_state_jax
        else:
            current_y0_np = None

        if chunk_sol.final_synaptic_I is not None:
            final_syn_I_jax = chunk_sol.final_synaptic_I
            if hasattr(final_syn_I_jax, "block_until_ready"):
                final_syn_I_jax = final_syn_I_jax.block_until_ready()
            current_synaptic_I_np = np.array(final_syn_I_jax)
            del final_syn_I_jax
        else:
            current_synaptic_I_np = None

        current_solver_state_jax = jtu.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            chunk_sol.solver_state,
        )
        current_controller_state_jax = jtu.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            chunk_sol.controller_state,
        )
        current_made_jump_jax = jtu.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            chunk_sol.made_jump,
        )

        del chunk_sol
        gc.collect()
        jax.clear_caches()
    actual_max_spikes = 0
    concatenated_spike_times_jax_list = []
    concatenated_spike_marks_jax_list = []

    for sample_idx in range(num_samples):
        if all_spike_times_np_per_sample[sample_idx]:
            concat_times_np = np.concatenate(all_spike_times_np_per_sample[sample_idx])
            valid_marks_to_stack = [
                m
                for m in all_spike_marks_np_per_sample[sample_idx]
                if m.ndim == 2 and m.shape[0] > 0
            ]
            if valid_marks_to_stack:
                concat_marks_np = np.vstack(valid_marks_to_stack)
            else:
                concat_marks_np = np.empty((0, network.num_neurons), dtype=bool)

            concatenated_spike_times_jax_list.append(jnp.array(concat_times_np))
            concatenated_spike_marks_jax_list.append(jnp.array(concat_marks_np))
            actual_max_spikes = max(actual_max_spikes, concat_times_np.shape[0])
        else:
            concatenated_spike_times_jax_list.append(jnp.array([], dtype=jnp.float32))
            concatenated_spike_marks_jax_list.append(
                jnp.array([], dtype=bool).reshape(0, network.num_neurons)
            )

    # Use provided max_total_spikes or fall back to actual data size
    if max_total_spikes is not None:
        final_max_spikes = max_total_spikes
    else:
        final_max_spikes = actual_max_spikes

    if final_max_spikes == 0:
        final_max_spikes = 1

    padded_spike_times_jax = jnp.full(
        (num_samples, final_max_spikes), jnp.inf, dtype=jnp.float32
    )
    padded_spike_marks_jax = jnp.full(
        (num_samples, final_max_spikes, network.num_neurons), False, dtype=bool
    )

    for sample_idx in range(num_samples):
        num_s = concatenated_spike_times_jax_list[sample_idx].shape[0]
        if num_s > 0:
            # Truncate if necessary to fit within final_max_spikes
            num_to_set = min(num_s, final_max_spikes)
            padded_spike_times_jax = padded_spike_times_jax.at[
                sample_idx, :num_to_set
            ].set(concatenated_spike_times_jax_list[sample_idx][:num_to_set])
            padded_spike_marks_jax = padded_spike_marks_jax.at[
                sample_idx, :num_to_set
            ].set(concatenated_spike_marks_jax_list[sample_idx][:num_to_set])

    # Handle voltage data concatenation if not in spike_only mode
    if not spike_only:
        # Concatenate all voltage data across chunks
        concatenated_ys_list = []
        concatenated_ts_list = []

        for sample_idx in range(num_samples):
            if all_ys_np_per_sample[sample_idx]:
                # Stack all ys arrays along the spike dimension (axis 0)
                concat_ys = np.concatenate(all_ys_np_per_sample[sample_idx], axis=0)
                concat_ts = np.concatenate(all_ts_np_per_sample[sample_idx], axis=0)
                concatenated_ys_list.append(jnp.array(concat_ys))
                concatenated_ts_list.append(jnp.array(concat_ts))
            else:
                # Empty arrays if no data
                num_vars = len(store_vars) if memory_efficient else 8
                concatenated_ys_list.append(
                    jnp.array([]).reshape(
                        0, network.num_neurons, num_saves_per_chunk or 2, num_vars
                    )
                )
                concatenated_ts_list.append(
                    jnp.array([]).reshape(0, num_saves_per_chunk or 2)
                )

        # Pad to consistent shape
        max_segments = max([y.shape[0] for y in concatenated_ys_list] + [1])
        num_saves = num_saves_per_chunk or 2
        num_vars = (
            concatenated_ys_list[0].shape[-1]
            if concatenated_ys_list[0].shape[0] > 0
            else (len(store_vars) if memory_efficient else 8)
        )

        ys_final = jnp.full(
            (num_samples, max_segments, network.num_neurons, num_saves, num_vars),
            jnp.inf,
        )
        ts_final = jnp.full((num_samples, max_segments, num_saves), jnp.inf)

        for sample_idx in range(num_samples):
            num_segs = concatenated_ys_list[sample_idx].shape[0]
            if num_segs > 0:
                ys_final = ys_final.at[sample_idx, :num_segs].set(
                    concatenated_ys_list[sample_idx]
                )
                ts_final = ts_final.at[sample_idx, :num_segs].set(
                    concatenated_ts_list[sample_idx]
                )
    else:
        # Use placeholder arrays for spike_only mode
        ys_final = jnp.full(
            (num_samples, final_max_spikes, network.num_neurons, 2, 1), jnp.inf
        )
        ts_final = jnp.full((num_samples, final_max_spikes, 2), jnp.inf)

    # Final state passed to Solution object should be JAX arrays
    final_state_jax_for_sol = (
        jnp.array(current_y0_np) if current_y0_np is not None else None
    )
    final_syn_I_jax_for_sol = (
        jnp.array(current_synaptic_I_np) if current_synaptic_I_np is not None else None
    )

    final_solution = Solution(
        t1=t1,
        ys=ys_final,
        ts=ts_final,
        spike_times=padded_spike_times_jax,
        spike_marks=padded_spike_marks_jax,
        num_spikes=final_max_spikes,
        max_spikes=final_max_spikes,
        stored_vars=store_vars,
        spike_only=spike_only,
        final_state=final_state_jax_for_sol,
        final_synaptic_I=final_syn_I_jax_for_sol,
        solver_state=current_solver_state_jax,
        controller_state=current_controller_state_jax,
        made_jump=current_made_jump_jax,
    )

    return final_solution

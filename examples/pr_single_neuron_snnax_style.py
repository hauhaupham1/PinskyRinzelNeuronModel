"""
Pinsky-Rinzel Single Neuron Parameter Estimation (snnax style)

Direct adaptation of the snnax single_neuron notebook for our Pinsky-Rinzel model.
This mimics the exact structure and approach of the snnax example.
"""

import functools as ft
import pickle
import time
from datetime import datetime
import os
import sys
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
from jaxtyping import Array, Float, Real
import signax

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PRmodel_Motoneuron.Network import MotoneuronNetwork
from PRmodel_Motoneuron.paths import marcus_lift


# Configure JAX
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

SAVE_IDX = datetime.today().strftime("%Y-%m-%d-%H")
key = jr.PRNGKey(12345)


steps = 400
num_save = 2
max_spikes = 3
t0 = 0
t1 = 100  # 100ms simulation
dt0 = 1e-2
s = 2
sigma = jnp.array([[0, 0, 0, 0, 0, 0, 0, 0],
                   [s, s, s, s, s, s, s, s],
                   [s, s, s, s, s, s, s, s],
                   [s, s, s, s, s, s, s, s],
                   [s, s, s, s, s, s, s, s],
                   [s, s, s, s, s, s, s, s],
                   [s, s, s, s, s, s, s, s],
                   [s, s, s, s, s, s, s, s]])
sample_sizes = [16, 32, 64, 128]
c = 20.0  # input current amplitude (estimand)


@jax.vmap
def get_marcus_lifts(spike_times, spike_marks):
    return marcus_lift(t0, t1, spike_times, spike_marks)

@eqx.filter_jit
def get_data(data_size, c, key):
    """Generate data from PR model with input current c."""
    
    def input_current(t: Float) -> Array:
        # Constant current like in snnax
        return jnp.array([c])

    network = MotoneuronNetwork(
        num_neurons=1,
        threshold=-37.0,
        v_reset=-60.0,
        diffusion=True,
        key=key,
        sigma=sigma,
    )

    sol = network(
        input_current=input_current,
        t0=t0,
        t1=t1,
        max_spikes=max_spikes,
        num_samples=data_size,
        key=key,
        num_save=num_save,
        dt0=dt0,
        spike_only=True,
    )

    spike_trains = get_marcus_lifts(sol.spike_times, sol.spike_marks)
    return spike_trains

def dataloader(data, batch_size, loop, *, key):
    """Dataloader following snnax pattern."""
    spike_trains = data
    data_size, _, _ = spike_trains.shape
    indices = jnp.arange(data_size)
    
    while True:
        if batch_size == data_size:
            yield spike_trains
            if not loop:
                break
        perm = jr.permutation(key, indices)
        key = jr.split(key, 1)[0]
        start = 0
        end = batch_size
        while end < data_size:
            batch_perm = perm[start:end]
            yield spike_trains[batch_perm]
            start = end
            end = start + batch_size
        if not loop:
            break


class PinskyRinzelModel(eqx.Module):
    """PR model wrapper for parameter estimation (mimicking snnax SNN)."""
    c: Real

    def __call__(self, batch_size, key):
        return get_data(batch_size, self.c, key)

def expected_signature(y: Float[Array, ""], depth: int) -> Array:
    """Compute expected signature using signax (following snnax)."""
    signatures = jax.vmap(ft.partial(signax.signature, depth=depth))(y)
    return jnp.mean(signatures, axis=0)

@eqx.filter_jit
def expected_signature_loss(
    y_1: Float[Array, "... dim"],
    y_2: Float[Array, "... dim"],
    depth: int,
    match_spikes: bool = True,
) -> Real:
    """Signature kernel MMD between two batches (following snnax exactly)."""
    if match_spikes and y_1.shape[-1] > 2:
        spike_counts_1 = jnp.max(y_1[:, :, 1:], axis=1)
        spike_counts_2 = jnp.max(y_2[:, :, 1:], axis=1)
        spike_counts = jnp.minimum(spike_counts_1, spike_counts_2)
        y_1 = y_1.at[:, :, 1:].set(jax.vmap(jnp.minimum)(y_1[:, :, 1:], spike_counts))
        y_2 = y_2.at[:, :, 1:].set(jax.vmap(jnp.minimum)(y_2[:, :, 1:], spike_counts))

    sig_1 = expected_signature(y_1, depth)
    sig_2 = expected_signature(y_2, depth)
    return jnp.mean((sig_1 - sig_2) ** 2)

def get_n_first_spikes(
    y: Float[Array, "samples double_spikes _neurons"], n: int
) -> Float[Array, "samples neurons"]:
    """Extract first n spike times (following snnax exactly)."""
    @jax.vmap
    def _outer(_y):
        @jax.vmap
        def _inner(k):
            idx = jnp.sum(_y[:, 1:] < k, axis=0)
            return _y[idx, 0]
        return _inner(jnp.arange(n) + 1)
    out = _outer(y)
    return out

@eqx.filter_jit
def spike_MAE_loss(y_1: Float[Array, "... dim"], y_2: Float[Array, "... dim"], n: int) -> Real:
    """Mean absolute error between average n first spike times (following snnax exactly)."""
    first_spikes = get_n_first_spikes(y_1, n)
    avg_first_spikes_1 = jnp.mean(first_spikes, axis=0)
    first_spikes = get_n_first_spikes(y_2, n)
    avg_first_spikes_2 = jnp.mean(first_spikes, axis=0)
    return jnp.mean(jnp.abs(avg_first_spikes_1 - avg_first_spikes_2))

@eqx.filter_jit
def spike_MSE_loss(y_1: Float[Array, "... dim"], y_2: Float[Array, "... dim"], n: int) -> Real:
    """Mean squared error between average n first spike times (following snnax exactly)."""
    first_spikes = get_n_first_spikes(y_1, n)
    avg_first_spikes_1 = jnp.mean(first_spikes, axis=0)
    first_spikes = get_n_first_spikes(y_2, n)
    avg_first_spikes_2 = jnp.mean(first_spikes, axis=0)
    return jnp.mean((avg_first_spikes_1 - avg_first_spikes_2) ** 2)

@eqx.filter_jit
def fs_loss(model, data, batch_size, key, n=1):
    """First n spikes MSE loss."""
    spike_trains_gen = model(batch_size, key)
    return spike_MSE_loss(spike_trains_gen, data, n=n)
@eqx.filter_jit
def spike_train_es_loss(model, data, batch_size, key, depth=2, match_spikes=True):
    """Expected signature loss on spike trains."""
    spike_trains_gen = model(batch_size, key)
    return expected_signature_loss(spike_trains_gen, data, depth=depth, match_spikes=match_spikes)
@eqx.filter_jit
def fs_mae_loss(model, data, batch_size, key, n=1):
    """First n spike times MAE loss."""
    spike_trains_gen = model(batch_size, key)
    return spike_MAE_loss(spike_trains_gen, data, n=n)



@eqx.filter_jit
def make_step(model, grad_loss, optim, data, batch_size, opt_state, key):
    loss, grads = grad_loss(model, data, batch_size, key)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

def train(
    c,
    loss,
    *,
    test_loss=None,
    lr=1e-3,
    batch_size=128,
    steps=500,
    steps_per_print=10,
    data_size=1024,
    seed=567,
):
    """Training function following snnax pattern exactly."""
    key = jr.PRNGKey(seed)
    (
        data_key,
        test_key,
        c_key,
        dataloader_key,
        step_key,
    ) = jr.split(key, 5)

    c_init = jr.uniform(c_key, minval=10, maxval=40.0)  # Random initial guess
    generator = PinskyRinzelModel(c=c_init)

    grad_loss = eqx.filter_value_and_grad(loss)
    if test_loss is None:
        test_loss = ft.partial(fs_mae_loss, n=max_spikes)

    assert data_size % batch_size == 0
    num_batches = data_size // batch_size
    spike_trains = jnp.zeros((data_size, 2 * max_spikes, 2))
    print("Generating data...")
    for i in range(num_batches):
        data_key = jr.fold_in(data_key, i)
        spike_train = get_data(batch_size, c, data_key)
        li, ui = i * batch_size, (i + 1) * batch_size
        spike_trains = spike_trains.at[li:ui].set(spike_train)
        print(f"Batch {i + 1} / {num_batches} done.")
    data = spike_trains

    test_data = get_data(batch_size, c, test_key)
    print("Data generated. Starting training...")

    c_true = c
    loss_hist = []
    test_loss_hist = []
    c_hist = []
    # Increase learning rate for c=20 case
    if c <= 20:
        lr = lr   
    optim = optax.rmsprop(lr*2, decay=0.7, momentum=0.3)
    opt_state = optim.init(eqx.filter(generator, eqx.is_inexact_array))
    infinite_dataloader = dataloader(data, batch_size, loop=True, key=dataloader_key)

    for step, dat_i in zip(range(steps), infinite_dataloader):
        start = time.time()
        step = jnp.asarray(step)
        step_key = jr.fold_in(step_key, step)
        score, generator, opt_state = make_step(
            generator,
            grad_loss,
            optim,
            dat_i,
            batch_size,
            opt_state,
            step_key,
        )
        test_score = test_loss(generator, test_data, batch_size, step_key)
        c_current = generator.c
        loss_hist.append(score)
        test_loss_hist.append(test_score)
        c_hist.append(c_current)
        end = time.time()
        if (step % steps_per_print) == 0 or step == steps - 1:
            print(
                f"Step: {step}, Loss: {score}, Test loss: {test_score},"
                f"Computation time: {end - start}"
            )

    results = {
        "model": generator,
        "loss_hist": loss_hist,
        "test_loss_hist": test_loss_hist,
        "c_hist": c_hist,
        "c_true": c_true,
        "sample_size": data_size,
    }
    return results


def main():
    """Main execution following snnax notebook pattern."""
    
    loss_fns = {
        "sig_mmd": ft.partial(spike_train_es_loss, depth=3),
        "fs_mse": ft.partial(fs_loss, n=3),
    }

    res_dict = {}
    for k, fn in loss_fns.items():
        res_dict[k] = []
        for n in sample_sizes:
            print(f"\nTraining with {k} loss, sample size {n}")
            res_dict[k].append(train(c, fn, steps=steps, data_size=n, batch_size=n))

        # Save results
        fname = "./pr_single_neuron_" + k + "_" + SAVE_IDX
        with open(fname, "wb") as f:
            pickle.dump(res_dict[k], f)
        print(f"Results saved to {fname}")

    def plot_results(res_list, name):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        for res in res_list:
            n = res["sample_size"]
            loss_hist = res["loss_hist"]
            c_hist = res["c_hist"]
            ax[0].plot(loss_hist, lw=1, alpha=0.7, label=f"{n}")
            ax[1].plot(c_hist, lw=1, alpha=0.7)
        
        ax[0].set_title("Test Loss")
        ax[0].set_ylabel("Test loss")
        ax[0].legend(title="Sample size")
        
        ax[1].set_title("Parameter Estimation")
        ax[1].set_ylabel(r"$c$")
        ax[1].axhline(c, color="grey", linestyle="--")
        
        ax[0].set_xlabel("Step")
        ax[1].set_xlabel("Step")

        plt.savefig(f"./pr_single_neuron_{name}_{SAVE_IDX}.pdf", dpi=200, bbox_inches="tight")
        plt.show()

    for k in loss_fns.keys():
        print(f"\nPlotting results for {k}...")
        plot_results(res_dict[k], k)

    print(f"\nTrue c: {c}")
    for k in loss_fns.keys():
        print(f"\n{k.upper()} results:")
        for res in res_dict[k]:
            n = res["sample_size"]
            c_final = res["c_hist"][-1]
            error = abs(c_final - c)
            print(f"  Sample size {n}: c_est = {c_final:.3f}, error = {error:.3f}")

if __name__ == "__main__":
    main()
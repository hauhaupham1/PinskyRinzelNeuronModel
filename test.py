import yaml
import os

from PRmodel_Motoneuron.MotoneuronModel import MotoneuronModel
from diffrax import SaveAt
import jax.numpy as jnp
import matplotlib.pyplot as plt

model = MotoneuronModel(yaml_file='PRmodel_Motoneuron/motoneuron.yaml', param_set=20230210_84)

print(model.params)


ts = jnp.linspace(0, 2000, 20000)
save = SaveAt(ts=ts)
I_stim = -0.1  # -100.0 * 1.0e-3 

sol = model.solve(t_dur=2000, I_stim=I_stim, stim_start=250, stim_end=1250, 
                 dt=0.015, saveat=save, max_steps=1000000)

# Calculate Rin
baseline_mask = (sol.ts >= 200) & (sol.ts < 250)
baseline_v = jnp.mean(sol.ys[baseline_mask, 0])
steady_mask = (sol.ts >= 1200) & (sol.ts < 1250)
steady_v = jnp.mean(sol.ys[steady_mask, 0])
delta_v = steady_v - baseline_v
Rin = abs(delta_v) / abs(I_stim)

print(f"\nRin Calculation:")
print(f"Baseline voltage: {baseline_v:.2f} mV")
print(f"Steady-state voltage: {steady_v:.2f} mV")
print(f"Voltage change (ΔV): {delta_v:.2f} mV")
print(f"Input resistance (Rin): {Rin:.2f} MΩ")
print(f"Target range: 540.0 - 598.0 MΩ")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(sol.ts, sol.ys[:,0], 'b-', label='Somatic Voltage (Vs)')
plt.plot(sol.ts, sol.ys[:,1], 'r-', label='Dendritic Voltage (Vd)')
plt.axvspan(250, 1250, alpha=0.2, color='yellow', label='Current Injection')
plt.axhline(y=baseline_v, color='k', linestyle=':', label=f'Baseline: {baseline_v:.2f} mV')
plt.axhline(y=steady_v, color='g', linestyle=':', label=f'Steady-state: {steady_v:.2f} mV')

plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.title('Input Resistance (Rin) Test')
plt.legend()
plt.grid(True)
plt.savefig('rin_test_fixed.png')
plt.show()
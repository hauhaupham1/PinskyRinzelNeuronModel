import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt


class PinskyRinzel:

    def __init__(self, 
                 g_c=2.1,           # Coupling conductance [mS cm**-2]
                 C_m=3.0,           # Membrane capacitance [uF cm**-2]
                 p=0.5,             # Proportion of membrane area for soma
                 g_L=0.1,           # Leak conductance [mS cm**-2]
                 g_Na=30.0,         # Sodium conductance [mS cm**-2] 
                 g_DR=15.0,         # Delayed rectifier K+ conductance [mS cm**-2]
                 g_Ca=10.0,         # Calcium conductance [mS cm**-2]
                 g_AHP=0.8,         # After-hyperpolarization K+ conductance [mS cm**-2]
                 g_C=15.0,          # Ca-dependent K+ conductance [mS cm**-2]
                 E_L=-68.0,         # Leak reversal potential [mV]
                 E_Na=60.0,         # Sodium reversal potential [mV]
                 E_K=-75.0,         # Potassium reversal potential [mV]
                 E_Ca=80.0,         # Calcium reversal potential [mV]
                 initial_state=None #Optional initial state
        ):
        self.g_c = g_c
        self.C_m = C_m
        self.p = p

        # Conductances
        self.g_L = g_L
        self.g_Na = g_Na
        self.g_DR = g_DR
        self.g_Ca = g_Ca
        self.g_AHP = g_AHP
        self.g_C = g_C
        # Reversal potentials
        self.E_L = E_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_Ca = E_Ca

        if initial_state is None:
            self.initial_state = jnp.array([-68.0, -68., 0.001, 0.999, 0.009, 0.007, 0.01, 0.2])
        
        else:
            self.initial_state = jnp.array(initial_state)


        #Rate Functions
    def alpha_m(self, Vs):
        """Na+ channel activation rate"""
        V1 = Vs + 46.9
        alpha = -0.32 * V1 / (jnp.exp(-V1 / 4.) - 1.)
        return alpha
    
    def beta_m(self, Vs):
        """Na+ channel deactivation rate"""
        V2 = Vs + 19.9
        beta = 0.28 * V2 / (jnp.exp(V2 / 5.) - 1.)
        return beta

    def alpha_h(self, Vs):
        """Na+ channel inactivation rate"""
        alpha = 0.128 * jnp.exp((-43. - Vs) / 18.)
        return alpha

    def beta_h(self, Vs):
        """Na+ channel deinactivation rate"""
        V5 = Vs + 20.
        beta = 4. / (1 + jnp.exp(-V5 / 5.))
        return beta

    def alpha_n(self, Vs):
        """K+ delayed rectifier activation rate"""
        V3 = Vs + 24.9
        alpha = -0.016 * V3 / (jnp.exp(-V3 / 5.) - 1)
        return alpha

    def beta_n(self, Vs):
        """K+ delayed rectifier deactivation rate"""
        V4 = Vs + 40.
        beta = 0.25 * jnp.exp(-V4 / 40.)
        return beta

    def alpha_s(self, Vd):
        """Ca2+ channel activation rate"""
        alpha = 1.6 / (1 + jnp.exp(-0.072 * (Vd-5.)))
        return alpha

    def beta_s(self, Vd):
        """Ca2+ channel deactivation rate"""
        V6 = Vd + 8.9
        beta = 0.02 * V6 / (jnp.exp(V6 / 5.) - 1.)
        return beta

    def alpha_c(self, Vd):
        """Ca-dependent K+ channel activation rate"""
        V7 = Vd + 53.5
        V8 = Vd + 50.
        return jnp.where(
            Vd <= -10,
            0.0527 * jnp.exp(V8/11. - V7/27.),
            2 * jnp.exp(-V7 / 27.)
        )

    def beta_c(self, Vd):
        """Ca-dependent K+ channel deactivation rate"""
        V7 = Vd + 53.5
        alpha_c_val = self.alpha_c(Vd)
        return jnp.where(
            Vd <= -10,
            2. * jnp.exp(-V7 / 27.) - alpha_c_val,
            0.
        )

    def alpha_q(self, Ca):
        """AHP K+ channel activation rate"""
        return jnp.minimum(0.00002*Ca, 0.01)

    def beta_q(self, Ca):
        """AHP K+ channel deactivation rate"""
        return 0.001

    def chi(self, Ca):
        """Ca-dependent activation function"""
        return jnp.minimum(Ca/250., 1.)

    def m_inf(self, Vs):
        """Na+ channel steady-state activation"""
        return self.alpha_m(Vs) / (self.alpha_m(Vs) + self.beta_m(Vs))
    
    def vector_field(self, t, y, args):
        """Vector field for the Pinsky-Rinzel model."""
        Vs, Vd, n, h, s, c, q, Ca = y
        
        # Get stimulus parameters from args
        I_stim, stim_start, stim_end = args
        
        # Calculate ionic currents in somatic compartment
        I_leak_s = self.g_L * (Vs - self.E_L)
        I_Na = self.g_Na * self.m_inf(Vs)**2 * h * (Vs - self.E_Na)
        I_DR = self.g_DR * n * (Vs - self.E_K)
        I_ds = self.g_c * (Vd - Vs)  # Coupling current from dendrite to soma
        
        # Calculate ionic currents in dendritic compartment
        I_leak_d = self.g_L * (Vd - self.E_L)
        I_Ca = self.g_Ca * s**2 * (Vd - self.E_Ca)
        I_AHP = self.g_AHP * q * (Vd - self.E_K)
        I_C = self.g_C * c * self.chi(Ca) * (Vd - self.E_K)
        I_sd = -I_ds  # Coupling current from soma to dendrite
        
        # Apply stimulus using jax's where for conditional logic
        stimulus = jnp.where((t > stim_start) & (t < stim_end), I_stim/self.p, 0.0)
        
        # Differential equations
        dVsdt = (1./self.C_m) * (-I_leak_s - I_Na - I_DR + I_ds/self.p + stimulus)
        dVddt = (1./self.C_m) * (-I_leak_d - I_Ca - I_AHP - I_C + I_sd/(1-self.p))
        dhdt = self.alpha_h(Vs)*(1-h) - self.beta_h(Vs)*h 
        dndt = self.alpha_n(Vs)*(1-n) - self.beta_n(Vs)*n
        dsdt = self.alpha_s(Vd)*(1-s) - self.beta_s(Vd)*s
        dcdt = self.alpha_c(Vd)*(1-c) - self.beta_c(Vd)*c
        dqdt = self.alpha_q(Ca)*(1-q) - self.beta_q(Ca)*q
        dCadt = -0.13*I_Ca - 0.075*Ca

        return jnp.array([dVsdt, dVddt, dndt, dhdt, dsdt, dcdt, dqdt, dCadt])
    
    def solve(self,
              t_dur,
              I_stim=0.0,
              stim_start=0.0,
              stim_end=0.0,
              dt=0.05,
              saveat=None,
              ):
        term = ODETerm(self.vector_field)
        solver = Dopri5()
        if saveat is None:
            saveat = SaveAt(ts=jnp.linspace(0, t_dur, int(t_dur/dt)+1))

        sol = diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=t_dur,
            y0=self.initial_state,
            args=(I_stim, stim_start, stim_end),
            dt0=dt,
            saveat=saveat,
            max_steps=None,
        )
        return sol
    
    def get_spike_times(self,
                       sol,
                       threshold=-20.0,
                       ):
        Vs = sol.ys[:, 0]
        times = sol.ts
        spikes = jnp.logical_and(Vs[:-1] < threshold,
                                 Vs[1:] >= threshold)
        spike_times = times[:-1][spikes]
        return spike_times
    


#Example usage
if __name__ == "__main__":
    # Create a model with default parameters
    model = PinskyRinzel()
    
    # Simulate with a step stimulus
    sol = model.solve(
        t_dur=1000,       # 1000 ms simulation
        I_stim=0.5,      # 0.75 nA stimulus
        stim_start=200,   # Start at 100 ms
        stim_end=400      # End at 300 ms
    )
    
    # Access the results
    times = sol.ts
    Vs = sol.ys[:, 0]     # Somatic voltage
    Vd = sol.ys[:, 1]     # Dendritic voltage
    Ca = sol.ys[:, 7]     # Calcium concentration
    
    #plot the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(times, Vs, label='Vs')
    plt.ylabel('Vs (mV)')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(times, Vd, label='Vd')
    plt.ylabel('Vd (mV)')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(times, Ca, label='Ca')
    plt.ylabel('Ca (mM)')
    plt.xlabel('Time (ms)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, Event, RESULTS
import optimistix
import functools as ft

class MotoneuronNetwork:
    def __init__(self, num_neurons: int, connections = None, threshold: float = -20.0):
        self.num_neurons = num_neurons

        if connections is None:
            self.connections = jax.numpy.zeros((num_neurons, num_neurons))
        else:
            self.connections = connections

        self.C_m = jnp.ones(num_neurons) * 2.0        # global_cm
        self.p = jnp.ones(num_neurons) * 0.1           # pp
        self.g_c = jnp.ones(num_neurons) * 0.4185837  # gc
        self.g_L_soma = jnp.ones(num_neurons) * 0.00036572208  # soma_g_pas
        self.g_L_dend = jnp.ones(num_neurons) * 1.0e-05  # dend_g_pas
        self.g_Na = jnp.ones(num_neurons) * 0.2742448  # soma_gmax_Na
        self.g_DR = jnp.ones(num_neurons) * 0.23978254  # soma_gmax_K
        self.g_Ca = jnp.ones(num_neurons) * 0.0077317934  # soma_gmax_CaN
        self.g_AHP = jnp.ones(num_neurons) * 0.005  # dend_gmax_KCa
        self.g_C = jnp.ones(num_neurons) * 0.009327445  # soma_gmax_KCa
        
        # Reversal potentials
        self.E_L = jnp.ones(num_neurons) * -62.0  # e_pas
        self.E_Na = jnp.ones(num_neurons) * 60.0
        self.E_K = jnp.ones(num_neurons) * -75.0
        self.E_Ca = jnp.ones(num_neurons) * 80.0
        
        # Calcium dynamics
        self.f_Caconc = jnp.ones(num_neurons) * 0.004  # soma_f_Caconc
        self.alpha_Caconc = jnp.ones(num_neurons) * 1.0  # soma_alpha_Caconc
        self.kCa_Caconc = jnp.ones(num_neurons) * 8.0  # soma_kCa_Caconc

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
        def make_cond_fn(neuron_idx):
            def _cond_fn(t, y, args, **kwargs):
                # reshape to (state_vars, num_neurons) and detect when soma voltage crosses threshold
                y_reshaped = y.reshape(8, self.num_neurons)
                Vs = y_reshaped[0, neuron_idx]
                return Vs - self.threshold
            return _cond_fn
        
        self.cond_fn = [make_cond_fn(n) for n in range(self.num_neurons)]
            
        # Create the event for spike detection
        self.event = Event(
            cond_fn=self.cond_fn,
            root_finder=optimistix.Newton(
                rtol=1e-5,
                atol=1e-5
            )
        )
      
    def _initialize_state(self):
        """Initialize state variables for all neurons."""
        # Initial values from the PRmodel
        v_init = -60.0  # From motoneuron.yaml Numerics section
        
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
    
    def drift_vf(self, t, state, input_current):
        # vectorized drift: state is shape (8, N), ic is (N,)
        ic = input_current(t)
        # unpack per-variable arrays
        Vs, Vd, n, h, s, c, q, Ca = state
        # ionic currents
        I_leak_s = self.g_L_soma * (Vs - self.E_L)
        I_Na     = self.g_Na    * (self.m_inf(Vs)**2 * h) * (Vs - self.E_Na)
        I_DR     = self.g_DR    * n * (Vs - self.E_K)
        I_ds     = self.g_c     * (Vd - Vs)
        I_leak_d = self.g_L_dend* (Vd - self.E_L)
        I_Ca     = self.g_Ca    * (s**2) * (Vd - self.E_Ca)
        I_AHP    = self.g_AHP   * q * (Vd - self.E_K)
        I_C      = self.g_C     * c * jnp.minimum(Ca/250.,1.) * (Vd - self.E_K)
        I_sd     = -I_ds
        # derivatives per variable
        dVsdt = (1.0/self.C_m)*( -I_leak_s - I_Na - I_DR + I_ds/self.p + ic)
        dVddt = (1.0/self.C_m)*( -I_leak_d - I_Ca - I_AHP - I_C + I_sd/(1.0-self.p))
        dhdt  = self.alpha_h(Vs)*(1-h) - self.beta_h(Vs)*h
        dndt  = self.alpha_n(Vs)*(1-n) - self.beta_n(Vs)*n
        dsdt  = self.alpha_s(Vd)*(1-s) - self.beta_s(Vd)*s
        dcdt  = self.alpha_c(Vd)*(1-c) - self.beta_c(Vd)*c
        dqdt  = self.alpha_q(Ca)*(1-q)  - self.beta_q(Ca)*q
        dCadt = -self.f_Caconc*I_Ca - self.alpha_Caconc*Ca/self.kCa_Caconc
        # stack into (8, N)
        return jnp.stack([dVsdt, dVddt, dndt, dhdt, dsdt, dcdt, dqdt, dCadt], axis=0)

    def vector_field(self, t, y, args):
        # unpack args and build an input_current function
        I_stim, stim_start, stim_end = args
        def input_current_fn(t):
            stim = jnp.where((t>stim_start)&(t<stim_end), I_stim/self.p, jnp.zeros_like(I_stim))
            return stim #+ jnp.dot(self.connections, stim) only apply the external current
        # delegate to vmapped drift_vf
        return self.drift_vf(t, y, input_current_fn)
    
    def solve(self, 
              t_dur, 
              I_stim=None, 
              stim_start=None, 
              stim_end=None,
              dt=0.05,
              max_steps=100000):
        if I_stim is None:
            I_stim = jnp.zeros(self.num_neurons)
        elif not isinstance(I_stim, jnp.ndarray):
            I_stim = jnp.ones(self.num_neurons) * I_stim
            
        if stim_start is None:
            stim_start = jnp.zeros(self.num_neurons)
        elif not isinstance(stim_start, jnp.ndarray):
            stim_start = jnp.ones(self.num_neurons) * stim_start
            
        if stim_end is None:
            stim_end = jnp.zeros(self.num_neurons)
        elif not isinstance(stim_end, jnp.ndarray):
            stim_end = jnp.ones(self.num_neurons) * stim_end
        term = ODETerm(self.vector_field)
        solver = Dopri5()
        # Warm-start integration loop: continue until full duration
        solver_state = None
        made_jump = None
        controller_state = None
        t0_current = 0.0
        y0_current = self.initial_state
        sol = None
        while t0_current < t_dur:
            prev_t0 = t0_current
            #save points for this segment: times from t0_current to t_dur
            n_steps = int((t_dur - t0_current) / dt)
            ts_loop = jnp.linspace(t0_current, t_dur, n_steps + 1)
            # include controller_state for warm-start
            saveat = SaveAt(ts=ts_loop, solver_state=True, controller_state=True, made_jump=True)
            sol = diffeqsolve(
                term,
                solver,
                t0=t0_current,
                t1=t_dur,
                dt0=dt,
                y0=y0_current,
                args=(I_stim, stim_start, stim_end),
                saveat=saveat,
                event=self.event,
                max_steps=max_steps,
                solver_state=solver_state,
                controller_state=controller_state,
                made_jump=made_jump
            )
            # If solve did not end on an event (i.e., reached t_dur), break out
            if sol.result != RESULTS.event_occurred:
                break
            #Update the current time and state
            ts_arr = sol.ts
            finite_ts = ts_arr[jnp.isfinite(ts_arr)]
            t_last = float(finite_ts[-1])
            if t_last <= prev_t0:
                # no progression
                break
            t0_current = t_last
            ys_arr = sol.ys
            finite_ys = ys_arr[: finite_ts.shape[0]]
            y0_current = finite_ys[-1]
            solver_state = sol.solver_state
            controller_state = sol.controller_state
            made_jump = sol.made_jump

            #Propagate the weights if a spike occurred
            if sol.result == RESULTS.event_occurred:
                # Index of the neuron that spiked
                true_indices = jnp.where(jnp.array([bool(em) for em in sol.event_mask]))[0]
                true_index = true_indices[0]
                #Get the connected neurons index
                #array of connected neurons is the row of the connections matrix
                connected_neurons = self.connections[true_index]
                #the weights are the value of the connections matrix
                #now increase the weights of the connected neurons
                I_stim = I_stim + connected_neurons
                # reset the state of the neuron that spiked using JAX index update
                y0_current = y0_current.at[:8, true_index].set(self.initial_state[:8, true_index])
                #reset solver_state
                solver_state = None
        return sol
    
#example usage
if __name__ == "__main__":
    num_neurons = 2
    connections=jnp.array([[0, 0.5], [0, 0]], dtype=jnp.float32)
    model = MotoneuronNetwork(num_neurons, connections=connections, threshold=-37.0)
    
    # Solve the system
    t_dur = 100.0
    I_stim = jnp.array([0, 1])
    stim_start = jnp.zeros(num_neurons)
    stim_end = jnp.ones(num_neurons) * 50.0
    
    sol = model.solve(t_dur, I_stim, stim_start, stim_end)
    
    # Print the solution
    print(sol.event_mask)
    print(sol.ys[-1])
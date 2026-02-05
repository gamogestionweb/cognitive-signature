"""
snn_engine.py — Spiking Neural Network Engine
==============================================

Prioridad: 2

Input:  NetworkTopology (from topology_generator.py)
Output: Instantiated SNN with LIF adaptive neurons

Uses Brian2 (Python, flexible, ideal for prototyping) as the simulation backend.
Can be migrated to NEST (C++, scalable) or Norse (PyTorch GPU) for production.

Neuron model: Leaky Integrate-and-Fire with Spike-Frequency Adaptation (AdLIF)

    τ_m · dV/dt = -(V - V_rest) + R_m · I_syn(t) + I_noise(t) + I_thalamic(t)
    
    When V ≥ V_threshold:
        spike = 1
        V → V_reset
        V_threshold += Δ_thresh  (spike-frequency adaptation)
    
    τ_thresh · dV_threshold/dt = -(V_threshold - V_thresh_base)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from enum import Enum


class SimBackend(Enum):
    """Available simulation backends."""
    NUMPY = "numpy"       # Pure numpy (slow but no dependencies)
    BRIAN2 = "brian2"      # Brian2 simulator
    NORSE = "norse"        # PyTorch-based SNN (GPU)
    NEST = "nest"          # NEST simulator (C++, scalable)


@dataclass
class LIFParams:
    """Parameters for Leaky Integrate-and-Fire neuron with adaptation."""
    
    # Membrane
    tau_m: float = 20.0          # ms - membrane time constant
    V_rest: float = -65.0        # mV - resting potential
    V_thresh_base: float = -50.0 # mV - base threshold
    V_reset: float = -70.0       # mV - reset potential after spike
    R_m: float = 10.0            # MΩ - membrane resistance
    
    # Adaptation (spike-frequency adaptation)
    delta_thresh: float = 2.0    # mV - threshold increase per spike
    tau_thresh: float = 100.0    # ms - threshold recovery time constant
    
    # Noise
    noise_sigma: float = 0.5     # mV - noise standard deviation
    
    # Synaptic
    tau_syn_exc: float = 5.0     # ms - excitatory synaptic time constant
    tau_syn_inh: float = 10.0    # ms - inhibitory synaptic time constant
    
    # Refractory period
    t_refrac: float = 2.0        # ms - absolute refractory period


@dataclass
class SimulationState:
    """Complete state of the SNN at a given timestep."""
    
    timestep: int = 0
    time_ms: float = 0.0
    
    # Neuron state variables
    V: np.ndarray = field(default_factory=lambda: np.array([]))          # membrane potential
    V_thresh: np.ndarray = field(default_factory=lambda: np.array([]))   # adaptive threshold
    I_syn: np.ndarray = field(default_factory=lambda: np.array([]))      # synaptic current
    last_spike_time: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Spike recording
    spike_indices: List[np.ndarray] = field(default_factory=list)
    spike_times: List[np.ndarray] = field(default_factory=list)
    
    # Neuromodulation levels (set by neuromodulation.py)
    dopamine: float = 1.0
    serotonin: float = 1.0
    noradrenaline: float = 1.0
    acetylcholine: float = 0.5
    
    # Thalamic input (set by thalamus.py)
    I_thalamic: np.ndarray = field(default_factory=lambda: np.array([]))


class SNNEngine:
    """
    Spiking Neural Network engine using numpy backend.
    
    This is the core simulation loop. It takes a NetworkTopology and creates
    a running SNN with:
      - LIF neurons with spike-frequency adaptation
      - Exponential synapses (exc and inh)
      - Support for external currents (thalamic, neuromodulation)
      - Spike recording for analysis and plasticity
    
    Usage:
        from cogmind.topology_generator import TopologyGenerator, NetworkTopology
        
        topology = TopologyGenerator("signature.json").generate(n_total=100_000)
        engine = SNNEngine(topology)
        
        # Run 1 second of simulation
        state = engine.run(duration_ms=1000.0, dt=0.1)
        
        # Get spike raster
        spikes = engine.get_spike_raster()
    """
    
    def __init__(
        self,
        topology,  # NetworkTopology
        params: Optional[LIFParams] = None,
        backend: SimBackend = SimBackend.NUMPY,
    ):
        self.topology = topology
        self.params = params or LIFParams()
        self.backend = backend
        self.n = topology.n_total
        
        # Initialize state
        self.state = self._init_state()
        
        # Build sparse connectivity for fast lookup
        self._build_synapse_lookup()
        
        # Delay buffer for spike propagation
        self.max_delay_steps = int(np.ceil(np.max(topology.connection_delays) / 0.1))
        self.spike_buffer = np.zeros((self.max_delay_steps, self.n), dtype=bool)
        
        print(f"[SNNEngine] Initialized with {self.n:,} neurons, "
              f"{len(topology.connections_pre):,} synapses")
        print(f"  Backend: {backend.value}")
        print(f"  Max delay: {np.max(topology.connection_delays):.1f}ms "
              f"({self.max_delay_steps} steps)")
    
    def _init_state(self) -> SimulationState:
        """Initialize all neuron state variables."""
        p = self.params
        state = SimulationState()
        
        # Membrane potentials: randomized near rest
        state.V = np.random.uniform(p.V_rest - 5, p.V_rest + 5, size=self.n).astype(np.float32)
        state.V_thresh = np.full(self.n, p.V_thresh_base, dtype=np.float32)
        state.I_syn = np.zeros(self.n, dtype=np.float32)
        state.last_spike_time = np.full(self.n, -1000.0, dtype=np.float32)
        state.I_thalamic = np.zeros(self.n, dtype=np.float32)
        
        return state
    
    def _build_synapse_lookup(self):
        """Build efficient sparse connectivity structure."""
        topo = self.topology
        
        # For each post-synaptic neuron: list of (pre_idx, weight, delay_steps)
        # Using numpy structured approach for speed
        self.syn_post_indices = {}
        
        # Convert delays to timestep indices  
        delay_steps = np.round(topo.connection_delays / 0.1).astype(np.int32)
        
        # Group by post-synaptic neuron for efficient update
        sort_idx = np.argsort(topo.connections_post)
        self._sorted_pre = topo.connections_pre[sort_idx]
        self._sorted_post = topo.connections_post[sort_idx]
        self._sorted_weights = topo.connection_weights[sort_idx]
        self._sorted_delays = delay_steps[sort_idx]
        
        # Find boundaries for each post-synaptic neuron
        unique_post, counts = np.unique(self._sorted_post, return_counts=True)
        self._post_boundaries = np.zeros(self.n + 1, dtype=np.int64)
        cumsum = np.cumsum(counts)
        self._post_boundaries[unique_post + 1] = counts
        self._post_boundaries = np.cumsum(self._post_boundaries)
    
    def step(self, dt: float = 0.1, external_current: Optional[np.ndarray] = None):
        """
        Advance simulation by one timestep.
        
        Args:
            dt: timestep in ms (default 0.1ms for numerical stability)
            external_current: optional external input current per neuron
        """
        p = self.params
        s = self.state
        
        # Current time
        t = s.time_ms
        
        # ── 1. Check refractory period ──
        refrac_mask = (t - s.last_spike_time) > p.t_refrac
        
        # ── 2. Compute total input current ──
        I_total = s.I_syn + s.I_thalamic
        if external_current is not None:
            I_total += external_current
        
        # Add noise
        I_noise = np.random.normal(0, p.noise_sigma, size=self.n).astype(np.float32)
        I_total += I_noise
        
        # ── 3. Apply neuromodulation ──
        # Noradrenaline: modulates gain (signal-to-noise)
        gain = 1.0 + (s.noradrenaline - 1.0) * 0.5
        I_total *= gain
        
        # Serotonin: modulates threshold
        thresh_mod = (s.serotonin - 1.0) * 5.0  # mV shift
        effective_thresh = s.V_thresh + thresh_mod
        
        # ── 4. Membrane dynamics (Euler integration) ──
        dV = (-((s.V - p.V_rest) - p.R_m * I_total)) / p.tau_m * dt
        s.V += dV * refrac_mask  # only update non-refractory neurons
        
        # ── 5. Spike detection ──
        spike_mask = (s.V >= effective_thresh) & refrac_mask
        spike_indices = np.where(spike_mask)[0]
        
        if len(spike_indices) > 0:
            # Reset spiking neurons
            s.V[spike_indices] = p.V_reset
            
            # Spike-frequency adaptation
            s.V_thresh[spike_indices] += p.delta_thresh
            
            # Record spike times
            s.last_spike_time[spike_indices] = t
            s.spike_indices.append(spike_indices.copy())
            s.spike_times.append(np.full(len(spike_indices), t))
            
            # Insert spikes into delay buffer
            buffer_idx = s.timestep % self.max_delay_steps
            self.spike_buffer[buffer_idx, :] = False
            self.spike_buffer[buffer_idx, spike_indices] = True
        
        # ── 6. Threshold recovery ──
        dthresh = -(s.V_thresh - p.V_thresh_base) / p.tau_thresh * dt
        s.V_thresh += dthresh
        
        # ── 7. Synaptic current update ──
        # Decay existing currents
        s.I_syn *= np.exp(-dt / p.tau_syn_exc)
        
        # Propagate spikes through delayed connections
        self._propagate_spikes(dt)
        
        # ── 8. Advance time ──
        s.timestep += 1
        s.time_ms += dt
    
    def _propagate_spikes(self, dt: float):
        """Propagate spikes through synaptic connections with delays."""
        s = self.state
        topo = self.topology
        
        # Check each delay bin for spikes that should arrive now
        for delay_offset in range(min(self.max_delay_steps, 5)):
            buf_idx = (s.timestep - delay_offset) % self.max_delay_steps
            if buf_idx < 0:
                continue
            
            spiking = np.where(self.spike_buffer[buf_idx])[0]
            if len(spiking) == 0:
                continue
            
            # Find all outgoing connections from spiking neurons with matching delay
            for pre_neuron in spiking:
                # Find connections where this neuron is presynaptic
                mask = self._sorted_pre == pre_neuron
                if not np.any(mask):
                    continue
                
                # Get subset with approximately matching delay
                conn_mask = mask & (np.abs(self._sorted_delays - delay_offset) <= 1)
                
                if np.any(conn_mask):
                    post_neurons = self._sorted_post[conn_mask]
                    weights = self._sorted_weights[conn_mask]
                    
                    # Acetylcholine: modulate external vs internal weight balance
                    # (simplified: affects all weights uniformly here)
                    ach_mod = 1.0 + (s.acetylcholine - 0.5) * 0.3
                    
                    # Add weighted input to postsynaptic neurons
                    np.add.at(s.I_syn, post_neurons, weights * ach_mod)
    
    def run(
        self,
        duration_ms: float,
        dt: float = 0.1,
        external_current_fn=None,
        progress_interval_ms: float = 100.0,
    ) -> SimulationState:
        """
        Run simulation for a given duration.
        
        Args:
            duration_ms: Total simulation time in milliseconds
            dt: Timestep in ms
            external_current_fn: Optional function(time_ms) → current_array
            progress_interval_ms: Print progress every N ms
            
        Returns:
            Final SimulationState
        """
        n_steps = int(duration_ms / dt)
        last_progress = 0.0
        
        print(f"\n[SNNEngine] Running {duration_ms:.0f}ms simulation "
              f"({n_steps:,} steps, dt={dt}ms)...")
        
        for step in range(n_steps):
            t = step * dt
            
            # Get external current if provided
            ext = None
            if external_current_fn is not None:
                ext = external_current_fn(t)
            
            self.step(dt=dt, external_current=ext)
            
            # Progress reporting
            if t - last_progress >= progress_interval_ms:
                n_spikes = sum(len(s) for s in self.state.spike_indices[-int(progress_interval_ms/dt):])
                rate = n_spikes / (self.n * progress_interval_ms / 1000.0)
                print(f"  t={t:.0f}ms | spikes: {n_spikes:,} | "
                      f"rate: {rate:.1f} Hz/neuron")
                last_progress = t
        
        total_spikes = sum(len(s) for s in self.state.spike_indices)
        avg_rate = total_spikes / (self.n * duration_ms / 1000.0)
        print(f"\n[SNNEngine] ✓ Complete. Total spikes: {total_spikes:,}, "
              f"avg rate: {avg_rate:.1f} Hz/neuron")
        
        return self.state
    
    def get_spike_raster(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get spike raster data for plotting.
        
        Returns:
            (spike_times, spike_neuron_ids) - both as 1D arrays
        """
        if not self.state.spike_indices:
            return np.array([]), np.array([])
        
        all_indices = np.concatenate(self.state.spike_indices)
        all_times = np.concatenate(self.state.spike_times)
        return all_times, all_indices
    
    def get_firing_rates(self, window_ms: float = 100.0) -> np.ndarray:
        """Compute firing rate per neuron over the last window_ms."""
        t_now = self.state.time_ms
        t_start = t_now - window_ms
        
        rates = np.zeros(self.n, dtype=np.float32)
        for times, indices in zip(self.state.spike_times, self.state.spike_indices):
            mask = times >= t_start
            if np.any(mask):
                valid_indices = indices[mask] if len(mask.shape) == 0 else indices
                np.add.at(rates, valid_indices, 1.0)
        
        rates /= (window_ms / 1000.0)  # Convert to Hz
        return rates
    
    def get_module_activity(self) -> Dict[str, float]:
        """Get average firing rate per cortical module."""
        rates = self.get_firing_rates()
        module_names = self.topology.parameters.get("module_names", [])
        
        activity = {}
        for mid, name in enumerate(module_names):
            mask = self.topology.neuron_modules == mid
            if np.any(mask):
                activity[name] = float(np.mean(rates[mask]))
        
        return activity
    
    def reset(self):
        """Reset simulation to initial state."""
        self.state = self._init_state()
        self.spike_buffer[:] = False
        print("[SNNEngine] State reset")


def create_brian2_network(topology, params: Optional[LIFParams] = None):
    """
    Create a Brian2 Network from the topology (for users with Brian2 installed).
    
    This provides higher-fidelity simulation than the numpy backend.
    
    Returns:
        Brian2 Network object ready to run
    """
    try:
        from brian2 import (
            NeuronGroup, Synapses, Network, SpikeMonitor, StateMonitor,
            ms, mV, second, Hz, MOhm
        )
    except ImportError:
        raise ImportError(
            "Brian2 is required for this backend. "
            "Install with: pip install brian2"
        )
    
    p = params or LIFParams()
    N = topology.n_total
    
    # ── Define neuron equations ──
    eqs = '''
    dv/dt = (-(v - v_rest) + R_m * I_syn + I_ext + I_noise) / tau_m : volt
    dv_thresh/dt = -(v_thresh - v_thresh_base) / tau_thresh : volt
    dI_syn/dt = -I_syn / tau_syn : amp
    I_ext : amp
    I_noise = noise_sigma * xi * tau_m**-0.5 : volt
    v_rest : volt
    v_thresh_base : volt
    tau_m : second
    tau_thresh : second
    tau_syn : second
    R_m : ohm
    noise_sigma : volt
    neuron_type : 1  # 1=exc, -1=inh
    '''
    
    threshold = 'v > v_thresh'
    reset = '''
    v = {V_reset}*mV
    v_thresh += {delta_thresh}*mV
    '''.format(V_reset=p.V_reset, delta_thresh=p.delta_thresh)
    
    # Create neuron group
    neurons = NeuronGroup(
        N, eqs,
        threshold=threshold,
        reset=reset,
        refractory=p.t_refrac * ms,
        method='euler'
    )
    
    # Set parameters
    neurons.v = np.random.uniform(p.V_rest - 5, p.V_rest + 5, N) * mV
    neurons.v_rest = p.V_rest * mV
    neurons.v_thresh = p.V_thresh_base * mV
    neurons.v_thresh_base = p.V_thresh_base * mV
    neurons.tau_m = p.tau_m * ms
    neurons.tau_thresh = p.tau_thresh * ms
    neurons.R_m = p.R_m * MOhm
    neurons.noise_sigma = p.noise_sigma * mV
    neurons.neuron_type = topology.neuron_types
    
    # ── Create synapses ──
    syn = Synapses(neurons, neurons, 
                   'w : 1', 
                   on_pre='I_syn += w * nA',
                   delay=topology.connection_delays * ms)
    
    syn.connect(i=topology.connections_pre.tolist(), 
                j=topology.connections_post.tolist())
    syn.w = topology.connection_weights
    
    # ── Monitors ──
    spike_mon = SpikeMonitor(neurons)
    
    # Build network
    net = Network(neurons, syn, spike_mon)
    
    print(f"[Brian2] Created network: {N:,} neurons, "
          f"{len(topology.connections_pre):,} synapses")
    
    return net, neurons, syn, spike_mon

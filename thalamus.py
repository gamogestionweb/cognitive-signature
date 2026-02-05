"""
thalamus.py — Thalamic Router, Oscillator, and Attention Gate
=============================================================

Prioridad: 4

Input:  Oscillation frequency config, cortical module states
Output: Synchronization signals injected into cortical neurons

The thalamus serves 3 critical functions:

  1. OSCILLATOR: Generates rhythmic signals at biologically relevant frequencies
     - Theta  (4-8 Hz):   memory encoding, navigation
     - Alpha  (8-13 Hz):  idle/inhibition, sensory gating
     - Gamma  (30-100 Hz): binding, integration, consciousness
     
  2. ROUTER: Routes information between cortical modules
     - Thalamo-cortical loops create the recurrence that LLMs lack
     
  3. GATE: Filters sensory input (attention mechanism)
     - Unlike transformer self-attention: dynamic, modulatory, state-dependent

The gamma oscillations (30-100 Hz) are the physical substrate for
information integration. They create ~10-33ms windows where neurons
from different modules can synchronize → this is what generates Φ
(integrated information) in the IIT framework.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class OscillationBand:
    """A frequency band with its parameters."""
    name: str
    frequency_hz: float
    amplitude: float = 1.0
    phase_offset: float = 0.0
    
    # Which modules does this band primarily target?
    target_modules: List[str] = field(default_factory=list)
    
    # Coupling strength to each module (0-1)
    coupling_strength: float = 0.5


@dataclass
class ThalamusConfig:
    """Configuration for the thalamic system."""
    
    # Oscillation bands
    bands: List[OscillationBand] = field(default_factory=lambda: [
        OscillationBand("theta", 6.0, amplitude=0.8, coupling_strength=0.3),
        OscillationBand("alpha", 10.0, amplitude=1.0, coupling_strength=0.5),
        OscillationBand("gamma_low", 40.0, amplitude=0.6, coupling_strength=0.7),
        OscillationBand("gamma_high", 80.0, amplitude=0.4, coupling_strength=0.8),
    ])
    
    # Gating parameters
    gate_threshold: float = 0.3      # below this, input is suppressed
    gate_gain: float = 2.0           # amplification for attended inputs
    
    # Thalamo-cortical coupling
    tc_coupling_strength: float = 0.5  # overall coupling to cortex
    
    # Number of thalamic relay neurons (typically ~5-10% of cortical)
    n_relay_fraction: float = 0.05
    
    # Reticular nucleus (inhibitory gating)
    n_reticular_fraction: float = 0.02


class Thalamus:
    """
    Thalamic system: oscillator + router + attention gate.
    
    Generates rhythmic synchronization signals that bind cortical modules.
    Without this, modules process in isolation. With it, they integrate 
    information — the key ingredient for IIT's Φ.
    
    Usage:
        thalamus = Thalamus(n_cortical=100_000)
        
        # Each simulation step:
        I_thalamic = thalamus.step(
            dt=0.1,
            cortical_activity=firing_rates,
            sensory_input=sensor_data,
            arousal=neuromod.noradrenaline,
            attention_target="frontal_L"
        )
        
        # Inject into SNN
        engine.state.I_thalamic = I_thalamic
    """
    
    def __init__(
        self,
        n_cortical: int,
        config: Optional[ThalamusConfig] = None,
        module_names: Optional[List[str]] = None,
        neuron_modules: Optional[np.ndarray] = None,
    ):
        self.config = config or ThalamusConfig()
        self.n_cortical = n_cortical
        self.module_names = module_names or []
        self.neuron_modules = neuron_modules
        
        # Internal state
        self.time_ms = 0.0
        self.phases = np.zeros(len(self.config.bands))
        
        # Thalamic relay neurons
        self.n_relay = int(n_cortical * self.config.n_relay_fraction)
        self.relay_activity = np.zeros(self.n_relay, dtype=np.float32)
        
        # Reticular nucleus (inhibitory gating layer)
        self.n_reticular = int(n_cortical * self.config.n_reticular_fraction)
        self.reticular_activity = np.zeros(self.n_reticular, dtype=np.float32)
        
        # Attention state (which module is being "attended to")
        self.attention_weights = np.ones(len(self.module_names), dtype=np.float32)
        self.attention_weights /= max(len(self.module_names), 1)
        
        # Gate state per module
        self.gate_state = np.ones(len(self.module_names), dtype=np.float32)
        
        # Cross-frequency coupling state
        self.theta_phase = 0.0
        self.gamma_amplitude_modulation = 1.0
        
        print(f"[Thalamus] Initialized:")
        print(f"  Relay neurons: {self.n_relay:,}")
        print(f"  Reticular neurons: {self.n_reticular:,}")
        print(f"  Oscillation bands: {[b.name for b in self.config.bands]}")
    
    def step(
        self,
        dt: float,
        cortical_activity: Optional[np.ndarray] = None,
        sensory_input: Optional[np.ndarray] = None,
        arousal: float = 1.0,
        attention_target: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute thalamic output for one timestep.
        
        Args:
            dt: Timestep in ms
            cortical_activity: Firing rates per cortical neuron
            sensory_input: External sensory input per cortical neuron
            arousal: Arousal level (from noradrenaline, 0-2)
            attention_target: Name of module to attend to (optional)
            
        Returns:
            I_thalamic: Current to inject into each cortical neuron
        """
        self.time_ms += dt
        t = self.time_ms / 1000.0  # convert to seconds for Hz calculations
        
        # ── 1. Generate oscillation signals ──
        osc_signals = self._generate_oscillations(t, dt, arousal)
        
        # ── 2. Update attention gate ──
        if attention_target is not None:
            self._update_attention(attention_target, dt)
        
        # ── 3. Apply sensory gating ──
        gated_input = np.zeros(self.n_cortical, dtype=np.float32)
        if sensory_input is not None:
            gated_input = self._gate_sensory_input(sensory_input)
        
        # ── 4. Compute thalamo-cortical output ──
        I_thalamic = self._compute_output(osc_signals, gated_input, arousal)
        
        # ── 5. Cross-frequency coupling (theta-gamma) ──
        I_thalamic = self._apply_cross_frequency_coupling(I_thalamic, t)
        
        return I_thalamic
    
    def _generate_oscillations(
        self, t: float, dt: float, arousal: float
    ) -> Dict[str, float]:
        """
        Generate oscillation signals for each frequency band.
        
        Arousal modulates the balance between bands:
          - Low arousal: dominated by alpha/theta (sleep-like)
          - High arousal: dominated by gamma (alert, processing)
        """
        signals = {}
        
        for i, band in enumerate(self.config.bands):
            # Phase evolution
            self.phases[i] += 2 * np.pi * band.frequency_hz * (dt / 1000.0)
            self.phases[i] %= 2 * np.pi
            
            # Base signal
            signal = np.sin(self.phases[i] + band.phase_offset) * band.amplitude
            
            # Arousal modulation
            if band.name.startswith("gamma"):
                # Gamma increases with arousal
                signal *= np.clip(arousal, 0.2, 2.0)
            elif band.name == "alpha":
                # Alpha is strongest during relaxed wakefulness
                # Decreases with high arousal (desynchronization)
                signal *= np.clip(2.0 - arousal, 0.1, 1.5)
            elif band.name == "theta":
                # Theta modulated by memory encoding state
                signal *= np.clip(arousal * 0.8, 0.3, 1.2)
            
            signals[band.name] = signal
            
            # Track theta phase for cross-frequency coupling
            if band.name == "theta":
                self.theta_phase = self.phases[i]
        
        return signals
    
    def _update_attention(self, target_module: str, dt: float):
        """
        Shift attention to a target module.
        
        Attention is implemented as differential gating:
          - Attended module: gate opens (more signal passes through)
          - Unattended modules: gate partially closes
        """
        tau_attention = 50.0  # ms - time to shift attention
        alpha = dt / tau_attention
        
        for i, name in enumerate(self.module_names):
            if target_module in name:  # supports partial matching
                target_weight = self.config.gate_gain
            else:
                target_weight = 1.0 / self.config.gate_gain
            
            self.attention_weights[i] += alpha * (target_weight - self.attention_weights[i])
        
        # Normalize
        total = np.sum(self.attention_weights)
        if total > 0:
            self.attention_weights /= total
            self.attention_weights *= len(self.module_names)
    
    def _gate_sensory_input(self, sensory_input: np.ndarray) -> np.ndarray:
        """
        Apply thalamic gating to sensory input.
        
        The reticular nucleus (inhibitory) can suppress inputs below threshold.
        This is the biological attention mechanism.
        """
        gated = sensory_input.copy()
        
        if self.neuron_modules is not None:
            for i, name in enumerate(self.module_names):
                mask = self.neuron_modules == i
                
                # Apply attention weight
                gated[mask] *= self.attention_weights[i]
                
                # Apply gate threshold
                below_threshold = np.abs(gated[mask]) < self.config.gate_threshold
                gated_subset = gated[mask]
                gated_subset[below_threshold] *= 0.1  # suppress but don't zero
                gated[mask] = gated_subset
        
        return gated
    
    def _compute_output(
        self,
        osc_signals: Dict[str, float],
        gated_input: np.ndarray,
        arousal: float,
    ) -> np.ndarray:
        """
        Compute the total thalamic current for each cortical neuron.
        
        Combines oscillation signals (synchronizing) with gated sensory input.
        """
        I_thalamic = np.zeros(self.n_cortical, dtype=np.float32)
        
        # ── Oscillation-driven current ──
        for band in self.config.bands:
            signal = osc_signals.get(band.name, 0.0)
            
            # Scale by coupling strength and overall tc coupling
            current = signal * band.coupling_strength * self.config.tc_coupling_strength
            
            # Apply per-module attention weights
            if self.neuron_modules is not None and len(self.module_names) > 0:
                for i, name in enumerate(self.module_names):
                    mask = self.neuron_modules == i
                    I_thalamic[mask] += current * self.attention_weights[i]
            else:
                I_thalamic += current
        
        # ── Gated sensory input ──
        I_thalamic += gated_input
        
        return I_thalamic
    
    def _apply_cross_frequency_coupling(
        self, I_thalamic: np.ndarray, t: float
    ) -> np.ndarray:
        """
        Apply theta-gamma coupling (phase-amplitude coupling).
        
        The amplitude of gamma oscillations is modulated by the phase of theta.
        This is observed in real brains and is thought to support sequential
        memory encoding (each gamma cycle within a theta cycle represents
        a different item).
        
        This is one of the mechanisms that creates temporal structure in
        neural processing — something completely absent in feedforward networks.
        """
        # Gamma amplitude is highest at the peak of theta
        theta_modulation = 0.5 + 0.5 * np.cos(self.theta_phase)
        
        # Apply to gamma components (higher frequency components of the signal)
        # Simplified: modulate the entire thalamic output
        self.gamma_amplitude_modulation = theta_modulation
        
        # Apply modulation (subtle effect)
        modulation_factor = 0.8 + 0.2 * theta_modulation
        I_thalamic *= modulation_factor
        
        return I_thalamic
    
    def get_oscillation_state(self) -> Dict[str, float]:
        """Get current oscillation amplitudes for each band."""
        t = self.time_ms / 1000.0
        state = {}
        for i, band in enumerate(self.config.bands):
            signal = np.sin(self.phases[i]) * band.amplitude
            state[band.name] = float(signal)
        state["theta_phase"] = float(self.theta_phase)
        state["gamma_modulation"] = float(self.gamma_amplitude_modulation)
        return state
    
    def get_attention_map(self) -> Dict[str, float]:
        """Get current attention distribution across modules."""
        return {
            name: float(self.attention_weights[i])
            for i, name in enumerate(self.module_names)
        }

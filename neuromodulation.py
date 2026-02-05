"""
neuromodulation.py — Global Neuromodulation System
==================================================

Prioridad: 5

Input:  Environment state + reward signals
Output: Levels of DA/5HT/NE/ACh that modulate global network parameters

The 4 "global dials" that an LLM doesn't have:

  DOPAMINE (DA):      Learning rate of STDP. Reward/prediction signal.
                      High DA → STDP learns faster from what just happened.
                      Implementation: A_plus *= dopamine_level

  SEROTONIN (5HT):    Global firing threshold. Network selectivity.
                      High 5HT → higher threshold → more selective, less noise.
                      Implementation: V_thresh += serotonin_level * 5mV

  NORADRENALINE (NE):  Neuronal gain. Signal-to-noise ratio.
                      High NE → strong signals stronger, weak signals weaker.
                      Implementation: gain = 1 + noradrenaline_level

  ACETYLCHOLINE (ACh): Balance external vs internal inputs.
                      High ACh → more weight to sensory input, less to memory.
                      Implementation: w_external *= (1 + ach); w_internal *= (1 - ach)

WHY THIS MATTERS:
  A LLM has nothing equivalent. Its activation functions (ReLU, GELU) are
  local and fixed. Neuromodulation is GLOBAL and DYNAMIC: a single "dial"
  changes the behavior of the entire network simultaneously. This enables
  states like "alert", "relaxed", "focused", "creative". Without neuromodulation,
  the network can process but cannot have STATES.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class NeuromodulatorState:
    """Current levels of all neuromodulators."""
    dopamine: float = 1.0       # 0.0 - 3.0 (baseline = 1.0)
    serotonin: float = 1.0      # 0.3 - 2.0 (baseline = 1.0)
    noradrenaline: float = 1.0  # 0.2 - 3.0 (baseline = 1.0)
    acetylcholine: float = 0.5  # 0.0 - 1.0 (baseline = 0.5)
    
    def as_dict(self) -> Dict[str, float]:
        return {
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "noradrenaline": self.noradrenaline,
            "acetylcholine": self.acetylcholine,
        }
    
    def __repr__(self):
        return (f"Neuromod(DA={self.dopamine:.2f}, 5HT={self.serotonin:.2f}, "
                f"NE={self.noradrenaline:.2f}, ACh={self.acetylcholine:.2f})")


@dataclass
class NeuromodConfig:
    """Configuration for neuromodulation dynamics."""
    
    # Time constants (ms) - how fast each modulator changes
    tau_dopamine: float = 200.0
    tau_serotonin: float = 2000.0     # slow, mood-like
    tau_noradrenaline: float = 100.0  # fast, arousal responses
    tau_acetylcholine: float = 500.0
    
    # Bounds
    da_min: float = 0.0
    da_max: float = 3.0
    ht_min: float = 0.3
    ht_max: float = 2.0
    ne_min: float = 0.2
    ne_max: float = 3.0
    ach_min: float = 0.0
    ach_max: float = 1.0
    
    # Baseline (resting) levels
    da_baseline: float = 1.0
    ht_baseline: float = 1.0
    ne_baseline: float = 1.0
    ach_baseline: float = 0.5
    
    # Sensitivity to different inputs
    da_reward_sensitivity: float = 2.0    # how much reward affects DA
    ne_novelty_sensitivity: float = 1.5   # how much novelty affects NE
    ach_external_sensitivity: float = 1.0 # how much external input affects ACh


class NeuromodulationSystem:
    """
    Global neuromodulation system.
    
    Updates neuromodulator levels based on:
      - Reward signals → Dopamine
      - Novelty / surprise → Noradrenaline  
      - Overall activity balance → Serotonin
      - External input salience → Acetylcholine
    
    These levels then modulate the SNN engine and plasticity rules:
      DA  → STDP learning rate
      5HT → firing threshold
      NE  → neuronal gain
      ACh → external/internal balance
    
    Usage:
        neuromod = NeuromodulationSystem()
        
        # Each simulation step:
        neuromod.step(
            dt=0.1,
            reward=reward_signal,
            novelty=novelty_score,
            activity_level=mean_firing_rate,
            external_salience=sensory_magnitude,
        )
        
        # Apply to engine
        engine.state.dopamine = neuromod.state.dopamine
        engine.state.serotonin = neuromod.state.serotonin
        engine.state.noradrenaline = neuromod.state.noradrenaline
        engine.state.acetylcholine = neuromod.state.acetylcholine
    """
    
    def __init__(self, config: Optional[NeuromodConfig] = None):
        self.config = config or NeuromodConfig()
        self.state = NeuromodulatorState(
            dopamine=self.config.da_baseline,
            serotonin=self.config.ht_baseline,
            noradrenaline=self.config.ne_baseline,
            acetylcholine=self.config.ach_baseline,
        )
        
        # Internal tracking
        self._reward_history = []
        self._reward_prediction = 0.0
        self._activity_history = []
        
        # Phasic vs tonic dopamine
        self._tonic_da = self.config.da_baseline
        self._phasic_da = 0.0
        
        print(f"[Neuromodulation] Initialized: {self.state}")
    
    def step(
        self,
        dt: float,
        reward: float = 0.0,
        novelty: float = 0.0,
        activity_level: float = 5.0,
        external_salience: float = 0.0,
    ):
        """
        Update all neuromodulator levels for one timestep.
        
        Args:
            dt: Timestep in ms
            reward: Reward signal (-1 to +1)
            novelty: Novelty/surprise score (0 to 1)
            activity_level: Mean network firing rate in Hz
            external_salience: Magnitude of external sensory input
        """
        c = self.config
        
        # ── DOPAMINE ──
        self._update_dopamine(dt, reward)
        
        # ── NORADRENALINE ──
        self._update_noradrenaline(dt, novelty, activity_level)
        
        # ── SEROTONIN ──
        self._update_serotonin(dt, activity_level)
        
        # ── ACETYLCHOLINE ──
        self._update_acetylcholine(dt, external_salience)
    
    def _update_dopamine(self, dt: float, reward: float):
        """
        Dopamine: Reward Prediction Error (RPE) signal.
        
        DA = baseline + RPE
        RPE = actual_reward - predicted_reward
        
        Positive RPE (better than expected): DA burst → enhance learning
        Negative RPE (worse than expected): DA dip → suppress learning
        Zero RPE (as expected): no change
        """
        c = self.config
        
        # Compute reward prediction error
        rpe = reward - self._reward_prediction
        
        # Update prediction (slow learning)
        self._reward_prediction += 0.01 * (reward - self._reward_prediction) * (dt / 1000.0)
        
        # Phasic dopamine: transient response to RPE
        self._phasic_da = rpe * c.da_reward_sensitivity
        
        # Total dopamine = tonic + phasic
        target_da = self._tonic_da + self._phasic_da
        
        # Smooth dynamics
        alpha = dt / c.tau_dopamine
        self.state.dopamine += alpha * (target_da - self.state.dopamine)
        
        # Clamp
        self.state.dopamine = np.clip(self.state.dopamine, c.da_min, c.da_max)
    
    def _update_noradrenaline(self, dt: float, novelty: float, activity: float):
        """
        Noradrenaline: Arousal and alertness.
        
        Responds to:
          - Novelty/surprise → NE burst (orient to new stimulus)
          - Sustained high activity → sustained NE (vigilance)
        
        NE modulates neuronal gain: 
          High NE → strong signals stronger, weak signals weaker
          → Better signal discrimination but higher energy cost
        """
        c = self.config
        
        # Target NE based on novelty and activity
        novelty_drive = novelty * c.ne_novelty_sensitivity
        activity_drive = np.clip((activity - 5.0) / 10.0, -0.5, 1.0)
        
        target_ne = c.ne_baseline + novelty_drive + activity_drive * 0.3
        
        alpha = dt / c.tau_noradrenaline
        self.state.noradrenaline += alpha * (target_ne - self.state.noradrenaline)
        self.state.noradrenaline = np.clip(self.state.noradrenaline, c.ne_min, c.ne_max)
    
    def _update_serotonin(self, dt: float, activity_level: float):
        """
        Serotonin: Mood, inhibition, and overall network regulation.
        
        Provides slow, tonic modulation of the network's excitability.
        High 5HT → more inhibited, more selective, calmer processing.
        Low 5HT → more excitable, more impulsive.
        
        This is the "mood dial" of the network.
        """
        c = self.config
        
        # Track activity history for slow regulation
        self._activity_history.append(activity_level)
        if len(self._activity_history) > 1000:
            self._activity_history = self._activity_history[-1000:]
        
        # If average activity is too high → increase 5HT (more inhibition)
        # If too low → decrease 5HT (more excitability)
        avg_activity = np.mean(self._activity_history)
        target_rate = 5.0  # Hz target
        
        activity_error = (avg_activity - target_rate) / target_rate
        target_ht = c.ht_baseline + activity_error * 0.5
        
        alpha = dt / c.tau_serotonin
        self.state.serotonin += alpha * (target_ht - self.state.serotonin)
        self.state.serotonin = np.clip(self.state.serotonin, c.ht_min, c.ht_max)
    
    def _update_acetylcholine(self, dt: float, external_salience: float):
        """
        Acetylcholine: External attention and learning mode.
        
        High ACh → more weight to external (sensory) inputs
                  → good for learning new things from the environment
        Low ACh  → more weight to internal (memory) inputs
                  → good for consolidation, daydreaming, planning
        
        This is the "external vs internal" dial.
        """
        c = self.config
        
        # ACh tracks external input salience
        target_ach = c.ach_baseline + external_salience * c.ach_external_sensitivity
        
        alpha = dt / c.tau_acetylcholine
        self.state.acetylcholine += alpha * (target_ach - self.state.acetylcholine)
        self.state.acetylcholine = np.clip(self.state.acetylcholine, c.ach_min, c.ach_max)
    
    def apply_to_snn(self, engine) -> None:
        """
        Apply current neuromodulator levels to the SNN engine state.
        
        This is the integration point between neuromodulation and the engine.
        """
        engine.state.dopamine = self.state.dopamine
        engine.state.serotonin = self.state.serotonin
        engine.state.noradrenaline = self.state.noradrenaline
        engine.state.acetylcholine = self.state.acetylcholine
    
    def get_cognitive_state(self) -> str:
        """
        Infer an approximate cognitive state from neuromodulator levels.
        
        Returns a descriptive string of the current "mental state".
        """
        da = self.state.dopamine
        ht = self.state.serotonin
        ne = self.state.noradrenaline
        ach = self.state.acetylcholine
        
        if ne > 2.0 and da > 1.5:
            return "alert_engaged"      # High arousal + high reward = flow state
        elif ne > 1.5 and ach > 0.7:
            return "focused_external"   # Attending to external stimuli
        elif ne < 0.7 and ach < 0.3:
            return "consolidation"      # Low arousal, internal processing (sleep-like)
        elif da > 2.0:
            return "reward_processing"  # Strong reward signal
        elif da < 0.5:
            return "aversive"           # Punishment / disappointment
        elif ht > 1.5 and ne < 1.0:
            return "relaxed_inhibited"  # Calm, selective processing
        elif ht < 0.6:
            return "impulsive"          # Low inhibition
        elif ach > 0.7:
            return "learning"           # External attention high
        else:
            return "baseline"           # Normal resting state
    
    def set_state_preset(self, preset: str):
        """
        Set neuromodulator levels to a named preset state.
        
        Useful for testing specific cognitive states.
        """
        presets = {
            "baseline": NeuromodulatorState(1.0, 1.0, 1.0, 0.5),
            "alert": NeuromodulatorState(1.2, 0.8, 2.0, 0.7),
            "relaxed": NeuromodulatorState(0.8, 1.5, 0.5, 0.3),
            "focused": NeuromodulatorState(1.5, 1.0, 1.5, 0.8),
            "creative": NeuromodulatorState(1.3, 0.7, 0.8, 0.4),
            "sleep_nrem": NeuromodulatorState(0.3, 0.5, 0.3, 0.1),
            "sleep_rem": NeuromodulatorState(0.5, 0.2, 0.1, 0.5),
            "stress": NeuromodulatorState(0.5, 0.5, 2.5, 0.8),
            "reward": NeuromodulatorState(2.5, 1.0, 1.5, 0.5),
        }
        
        if preset in presets:
            self.state = presets[preset]
            print(f"[Neuromodulation] Set to '{preset}': {self.state}")
        else:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(presets.keys())}")

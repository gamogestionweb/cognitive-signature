"""
plasticity.py — Spike-Timing-Dependent Plasticity (STDP)
========================================================

Prioridad: 3

Input:  STDP / R-STDP configuration
Output: Updated synaptic weights based on spike timing correlations

Implements:
  - Classic STDP: Hebbian learning based on spike timing
  - R-STDP: Reward-modulated STDP (dopamine-gated learning)
  - Homeostatic plasticity: keeps firing rates in biological range
  - Structural plasticity: synapse creation/pruning

STDP Rule:
    if Δt = t_post - t_pre > 0:  → LTP (strengthen)
        Δw = +A_plus · exp(-Δt / τ_plus)
    if Δt = t_post - t_pre < 0:  → LTD (weaken)
        Δw = -A_minus · exp(+Δt / τ_minus)

Reward modulation:
    Δw_final = Δw · dopamine_signal · eligibility_trace
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class STDPParams:
    """Parameters for Spike-Timing-Dependent Plasticity."""
    
    # LTP (Long-Term Potentiation)
    A_plus: float = 0.01      # amplitude of potentiation
    tau_plus: float = 20.0     # ms - time window for LTP
    
    # LTD (Long-Term Depression)  
    A_minus: float = 0.012     # amplitude of depression (slightly > A_plus to avoid saturation)
    tau_minus: float = 20.0    # ms - time window for LTD
    
    # Weight bounds
    w_min: float = 0.0         # minimum weight (excitatory)
    w_max: float = 1.0         # maximum weight
    
    # Eligibility trace (for R-STDP)
    tau_eligibility: float = 1000.0  # ms - how long a trace persists
    
    # Learning rate modulation
    learning_rate: float = 1.0
    
    # Homeostatic target
    target_rate_hz: float = 5.0  # target firing rate per neuron
    homeostatic_tau: float = 10000.0  # ms - homeostatic time constant


class STDPRule:
    """
    Classic Spike-Timing-Dependent Plasticity.
    
    Modifies synaptic weights based on the temporal correlation between
    pre-synaptic and post-synaptic spikes.
    
    Pre before Post → strengthen (LTP)
    Post before Pre → weaken (LTD)
    
    This is the fundamental learning rule of biological neural networks.
    """
    
    def __init__(self, n_synapses: int, params: Optional[STDPParams] = None):
        self.params = params or STDPParams()
        self.n_synapses = n_synapses
        
        # Pre-synaptic and post-synaptic traces
        self.trace_pre = np.zeros(n_synapses, dtype=np.float32)
        self.trace_post = np.zeros(n_synapses, dtype=np.float32)
        
        # Eligibility traces (for R-STDP)
        self.eligibility = np.zeros(n_synapses, dtype=np.float32)
        
        print(f"[STDP] Initialized for {n_synapses:,} synapses")
        print(f"  A+={self.params.A_plus}, A-={self.params.A_minus}")
        print(f"  τ+={self.params.tau_plus}ms, τ-={self.params.tau_minus}ms")
    
    def update(
        self,
        weights: np.ndarray,
        pre_indices: np.ndarray,
        post_indices: np.ndarray,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        dt: float,
        dopamine_level: float = 1.0,
    ) -> np.ndarray:
        """
        Apply STDP weight update for one timestep.
        
        Args:
            weights: Current synaptic weights
            pre_indices: Pre-synaptic neuron indices for each synapse
            post_indices: Post-synaptic neuron indices for each synapse
            pre_spikes: Boolean array of which neurons spiked (pre)
            post_spikes: Boolean array of which neurons spiked (post)
            dt: Timestep in ms
            dopamine_level: Dopamine modulation (1.0 = normal)
            
        Returns:
            Updated weights
        """
        p = self.params
        
        # ── 1. Decay traces ──
        self.trace_pre *= np.exp(-dt / p.tau_plus)
        self.trace_post *= np.exp(-dt / p.tau_minus)
        self.eligibility *= np.exp(-dt / p.tau_eligibility)
        
        # ── 2. Update traces for spiking neurons ──
        # Pre-synaptic spikes: increment pre-trace
        pre_spiking_synapses = pre_spikes[pre_indices]
        self.trace_pre[pre_spiking_synapses] += p.A_plus
        
        # Post-synaptic spikes: increment post-trace
        post_spiking_synapses = post_spikes[post_indices]
        self.trace_post[post_spiking_synapses] += p.A_minus
        
        # ── 3. Compute weight changes ──
        dw = np.zeros(self.n_synapses, dtype=np.float32)
        
        # LTP: post-synaptic spike → read pre-trace (pre fired before post)
        if np.any(post_spiking_synapses):
            dw[post_spiking_synapses] += self.trace_pre[post_spiking_synapses]
        
        # LTD: pre-synaptic spike → read post-trace (post fired before pre)
        if np.any(pre_spiking_synapses):
            dw[pre_spiking_synapses] -= self.trace_post[pre_spiking_synapses]
        
        # ── 4. Update eligibility trace ──
        self.eligibility += dw
        
        # ── 5. Apply weight change (modulated by dopamine) ──
        effective_dw = dw * dopamine_level * p.learning_rate
        weights += effective_dw
        
        # ── 6. Enforce weight bounds ──
        # Excitatory weights: clamp to [0, w_max]
        # Inhibitory weights: clamp to [-w_max, 0]
        exc_mask = weights > 0
        inh_mask = weights < 0
        weights[exc_mask] = np.clip(weights[exc_mask], p.w_min, p.w_max)
        weights[inh_mask] = np.clip(weights[inh_mask], -p.w_max, -p.w_min)
        
        return weights
    
    def get_weight_stats(self, weights: np.ndarray) -> dict:
        """Get statistics about current weight distribution."""
        exc = weights[weights > 0]
        inh = weights[weights < 0]
        return {
            "mean_exc": float(np.mean(exc)) if len(exc) > 0 else 0.0,
            "std_exc": float(np.std(exc)) if len(exc) > 0 else 0.0,
            "mean_inh": float(np.mean(inh)) if len(inh) > 0 else 0.0,
            "mean_eligibility": float(np.mean(np.abs(self.eligibility))),
            "n_zero_weights": int(np.sum(weights == 0)),
            "n_saturated": int(np.sum(np.abs(weights) >= self.params.w_max * 0.99)),
        }


class RewardModulatedSTDP(STDPRule):
    """
    Reward-Modulated STDP (R-STDP / Three-factor learning rule).
    
    Standard STDP tracks correlations, but the actual weight change only
    happens when a reward signal (dopamine) arrives. This implements the
    biological mechanism where:
    
    1. STDP creates an eligibility trace (marks which synapses were active)
    2. Dopamine signal arrives (reward/punishment)
    3. Only eligible synapses are modified, in proportion to dopamine
    
    This is the basis of reinforcement learning in biological brains.
    
    Δw_final = eligibility_trace × dopamine_signal
    """
    
    def __init__(self, n_synapses: int, params: Optional[STDPParams] = None):
        super().__init__(n_synapses, params)
        
        # Reward prediction error (RPE)
        self.reward_baseline = 0.0
        self.rpe_tau = 5000.0  # ms - time constant for reward baseline
        
        print(f"[R-STDP] Reward-modulated STDP enabled")
    
    def update_with_reward(
        self,
        weights: np.ndarray,
        pre_indices: np.ndarray,
        post_indices: np.ndarray,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        dt: float,
        reward: float = 0.0,
        dopamine_level: float = 1.0,
    ) -> np.ndarray:
        """
        Apply R-STDP: STDP eligibility + reward modulation.
        
        Args:
            reward: Reward signal (-1 to +1). Positive = good, negative = bad.
            dopamine_level: Base dopamine from neuromodulation system.
        """
        p = self.params
        
        # ── 1. Standard STDP trace updates ──
        self.trace_pre *= np.exp(-dt / p.tau_plus)
        self.trace_post *= np.exp(-dt / p.tau_minus)
        
        pre_spiking = pre_spikes[pre_indices]
        post_spiking = post_spikes[post_indices]
        
        self.trace_pre[pre_spiking] += p.A_plus
        self.trace_post[post_spiking] += p.A_minus
        
        # ── 2. Compute STDP eligibility ──
        dw_stdp = np.zeros(self.n_synapses, dtype=np.float32)
        
        if np.any(post_spiking):
            dw_stdp[post_spiking] += self.trace_pre[post_spiking]
        if np.any(pre_spiking):
            dw_stdp[pre_spiking] -= self.trace_post[pre_spiking]
        
        # ── 3. Update eligibility traces (with decay) ──
        self.eligibility *= np.exp(-dt / p.tau_eligibility)
        self.eligibility += dw_stdp
        
        # ── 4. Compute reward prediction error ──
        rpe = reward - self.reward_baseline
        self.reward_baseline += (reward - self.reward_baseline) * (dt / self.rpe_tau)
        
        # ── 5. Apply modulated weight change ──
        # The key: weight change = eligibility × dopamine × RPE
        effective_dopamine = dopamine_level * (1.0 + rpe)
        effective_dw = self.eligibility * effective_dopamine * p.learning_rate * (dt / 1000.0)
        
        weights += effective_dw
        
        # Enforce bounds
        exc_mask = weights > 0
        inh_mask = weights < 0
        weights[exc_mask] = np.clip(weights[exc_mask], p.w_min, p.w_max)
        weights[inh_mask] = np.clip(weights[inh_mask], -p.w_max, -p.w_min)
        
        return weights


class HomeostaticPlasticity:
    """
    Homeostatic plasticity maintains stable firing rates.
    
    If a neuron fires too much → reduce its excitability
    If a neuron fires too little → increase its excitability
    
    This prevents runaway excitation or complete silence, keeping the
    network in a dynamically interesting regime.
    
    Parameterized by CSF volume from Cognitive Signature:
        Higher CSF → more aggressive homeostasis
    """
    
    def __init__(
        self,
        n_neurons: int,
        target_rate_hz: float = 5.0,
        tau_homeostatic: float = 10000.0,
        homeostatic_rate: float = 0.1,
    ):
        self.n_neurons = n_neurons
        self.target_rate = target_rate_hz
        self.tau = tau_homeostatic
        self.rate = homeostatic_rate
        
        # Running average of firing rates
        self.avg_rates = np.full(n_neurons, target_rate_hz, dtype=np.float32)
        
        # Intrinsic excitability multiplier (modifies effective threshold)
        self.excitability = np.ones(n_neurons, dtype=np.float32)
        
        print(f"[Homeostatic] Target rate: {target_rate_hz} Hz, "
              f"regulation rate: {homeostatic_rate}")
    
    def update(
        self,
        firing_rates: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Update excitability based on firing rate deviation from target.
        
        Returns:
            excitability multiplier per neuron (apply to threshold)
        """
        # Update running average
        alpha = dt / self.tau
        self.avg_rates = (1 - alpha) * self.avg_rates + alpha * firing_rates
        
        # Compute error
        rate_error = self.avg_rates - self.target_rate
        
        # Adjust excitability (negative feedback)
        # Too active → decrease excitability (raise effective threshold)
        # Too quiet → increase excitability (lower effective threshold)
        self.excitability -= rate_error * self.rate * alpha
        
        # Bound excitability
        self.excitability = np.clip(self.excitability, 0.5, 2.0)
        
        return self.excitability


class StructuralPlasticity:
    """
    Structural plasticity: create and prune synapses.
    
    Biological brains don't just change weight — they create new connections
    and prune unused ones. This is parameterized by:
        Ventricular volume → pruning_rate (from Cognitive Signature)
    """
    
    def __init__(
        self,
        pruning_rate: float = 0.01,
        growth_rate: float = 0.005,
        min_weight_for_survival: float = 0.001,
    ):
        self.pruning_rate = pruning_rate
        self.growth_rate = growth_rate
        self.min_weight = min_weight_for_survival
        
        print(f"[StructuralPlasticity] Pruning: {pruning_rate}, Growth: {growth_rate}")
    
    def prune(
        self,
        weights: np.ndarray,
        pre_indices: np.ndarray,
        post_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove weak synapses.
        
        Returns:
            Pruned (weights, pre_indices, post_indices)
        """
        # Find synapses above minimum weight
        survive_mask = np.abs(weights) > self.min_weight
        
        n_pruned = np.sum(~survive_mask)
        if n_pruned > 0:
            weights = weights[survive_mask]
            pre_indices = pre_indices[survive_mask]
            post_indices = post_indices[survive_mask]
            print(f"  [StructuralPlasticity] Pruned {n_pruned:,} synapses")
        
        return weights, pre_indices, post_indices

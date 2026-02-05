"""
cogmind_runner.py — Main Integration Script
=============================================

Orchestrates all CogMind modules into a single simulation loop:

    Cognitive Signature → Topology → SNN → [Plasticity + Thalamus + 
    Neuromodulation + Hippocampus + Environment] → CogMind Instance

This is the complete pipeline from your brain scan to a running
cognitive emulation.

Usage:
    python -m cogmind.cogmind_runner path/to/signature.json --neurons 100000
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Optional, Dict

from cogmind.topology_generator import TopologyGenerator, NetworkTopology
from cogmind.snn_engine import SNNEngine, LIFParams
from cogmind.plasticity import STDPRule, RewardModulatedSTDP, HomeostaticPlasticity, STDPParams
from cogmind.thalamus import Thalamus, ThalamusConfig
from cogmind.neuromodulation import NeuromodulationSystem, NeuromodConfig
from cogmind.hippocampus import Hippocampus, HippocampusConfig
from cogmind.environment import VirtualEnvironment


class CogMindInstance:
    """
    A complete CogMind cognitive emulation instance.
    
    Integrates all 7 architectural layers:
        Layer 1-2: Cognitive Signature → Topology (structural seed)
        Layer 3-5: SNN Engine with Plasticity (cortical processing)
        Layer 4:   Thalamus (oscillations, routing, attention)
        Layer 5:   Cortex (SNN modules with recurrence)
        Layer 6:   Neuromodulation (global state modulation)
        Layer 7:   Environment (sensory I/O)
        + Hippocampus (episodic memory, replay, consolidation)
    
    Usage:
        instance = CogMindInstance.from_signature("signature.json", n_neurons=100_000)
        instance.run(duration_ms=10_000)
        instance.save_state("checkpoint.npz")
    """
    
    def __init__(
        self,
        topology: NetworkTopology,
        env_type: str = "pattern_recognition",
        lif_params: Optional[LIFParams] = None,
    ):
        self.topology = topology
        
        module_names = topology.parameters.get("module_names", [])
        
        # ── Initialize all subsystems ──
        print("\n" + "="*60)
        print("  CogMind Instance Initialization")
        print("="*60)
        
        # Layer 3-5: SNN Engine
        self.engine = SNNEngine(topology, params=lif_params)
        
        # Layer 3: Plasticity
        n_synapses = len(topology.connections_pre)
        self.stdp = RewardModulatedSTDP(n_synapses, STDPParams())
        self.homeostatic = HomeostaticPlasticity(
            topology.n_total,
            target_rate_hz=5.0,
            homeostatic_rate=topology.parameters.get("homeostatic_rate", 0.1),
        )
        
        # Layer 4: Thalamus
        self.thalamus = Thalamus(
            n_cortical=topology.n_total,
            module_names=module_names,
            neuron_modules=topology.neuron_modules,
        )
        
        # Layer 6: Neuromodulation
        self.neuromod = NeuromodulationSystem()
        
        # Hippocampus
        self.hippocampus = Hippocampus(
            n_cortical=topology.n_total,
            config=HippocampusConfig(pattern_dim=min(256, topology.n_total // 100)),
        )
        
        # Layer 7: Environment
        self.environment = VirtualEnvironment(
            n_cortical=topology.n_total,
            env_type=env_type,
            module_names=module_names,
            neuron_modules=topology.neuron_modules,
        )
        
        # Simulation state
        self.total_time_ms = 0.0
        self.total_steps = 0
        
        # Metrics history
        self.metrics_history = []
        
        print(f"\n{'='*60}")
        print(f"  ✓ CogMind Instance ready")
        print(f"  Signature: {topology.signature_id}")
        print(f"  Neurons: {topology.n_total:,}")
        print(f"  Synapses: {n_synapses:,}")
        print(f"  Environment: {env_type}")
        print(f"{'='*60}\n")
    
    @classmethod
    def from_signature(
        cls,
        signature_path: str,
        n_neurons: int = 100_000,
        env_type: str = "pattern_recognition",
        seed: int = 42,
    ) -> 'CogMindInstance':
        """
        Create a CogMind instance from a Cognitive Signature JSON file.
        
        This is the main entry point: signature.json → running mind emulation.
        """
        print("╔══════════════════════════════════════════════╗")
        print("║        CogMind — Cognitive Emulation         ║")
        print("║     From Cognitive Signature to Mind          ║")
        print("╚══════════════════════════════════════════════╝")
        
        # Generate topology from signature
        gen = TopologyGenerator(signature_path)
        topology = gen.generate(n_total=n_neurons, seed=seed)
        
        return cls(topology, env_type=env_type)
    
    def step(self, dt: float = 0.1):
        """
        Execute one complete simulation timestep.
        
        This is the core loop that integrates all subsystems:
        
        1. Environment → sensory input
        2. Thalamus → oscillation + gating
        3. SNN Engine → neural dynamics
        4. Plasticity → weight updates  
        5. Neuromodulation → state update
        6. Hippocampus → memory encoding
        7. Environment ← motor output
        """
        t = self.total_time_ms
        
        # ── 1. Get sensory input from environment ──
        sensory_input = self.environment.get_sensory_input(t)
        
        # ── 2. Thalamic processing ──
        I_thalamic = self.thalamus.step(
            dt=dt,
            cortical_activity=self.engine.get_firing_rates() if self.total_steps > 0 else None,
            sensory_input=sensory_input,
            arousal=self.neuromod.state.noradrenaline,
        )
        self.engine.state.I_thalamic = I_thalamic
        
        # ── 3. SNN engine step ──
        self.neuromod.apply_to_snn(self.engine)
        self.engine.step(dt=dt, external_current=sensory_input)
        
        # ── 4. Plasticity (every 1ms for efficiency) ──
        if self.total_steps % 10 == 0:  # every 10 steps = 1ms at dt=0.1
            spike_mask = np.zeros(self.topology.n_total, dtype=bool)
            if self.engine.state.spike_indices:
                last_spikes = self.engine.state.spike_indices[-1]
                spike_mask[last_spikes] = True
            
            # R-STDP update
            self.topology.connection_weights = self.stdp.update_with_reward(
                weights=self.topology.connection_weights,
                pre_indices=self.topology.connections_pre,
                post_indices=self.topology.connections_post,
                pre_spikes=spike_mask,
                post_spikes=spike_mask,
                dt=dt * 10,
                reward=self.environment.state.reward,
                dopamine_level=self.neuromod.state.dopamine,
            )
            
            # Homeostatic plasticity (every 100ms)
            if self.total_steps % 1000 == 0:
                rates = self.engine.get_firing_rates()
                excitability = self.homeostatic.update(rates, dt=dt * 1000)
        
        # ── 5. Neuromodulation update ──
        novelty = self.environment.get_novelty_signal()
        avg_rate = 0.0
        if self.total_steps > 0:
            rates = self.engine.get_firing_rates()
            avg_rate = float(np.mean(rates))
        
        self.neuromod.step(
            dt=dt,
            reward=self.environment.state.reward,
            novelty=novelty,
            activity_level=avg_rate,
            external_salience=float(np.mean(np.abs(sensory_input))),
        )
        
        # ── 6. Hippocampal processing (every 100ms) ──
        if self.total_steps % 1000 == 0 and self.total_steps > 0:
            rates = self.engine.get_firing_rates()
            hip_novelty = self.hippocampus.encode(
                rates, timestamp=t,
                neuromod_state=self.neuromod.state.as_dict(),
            )
            self.hippocampus.consolidate(t)
        
        # ── 7. Motor output → Environment ──
        if self.total_steps % 100 == 0:  # decode motor every 10ms
            rates = self.engine.get_firing_rates()
            # Decode from frontal module activity
            motor = self._decode_motor(rates)
            reward = self.environment.process_action(motor)
        
        # ── Advance time ──
        self.total_time_ms += dt
        self.total_steps += 1
    
    def run(
        self,
        duration_ms: float,
        dt: float = 0.1,
        report_interval_ms: float = 1000.0,
    ):
        """
        Run the full simulation for a given duration.
        
        Args:
            duration_ms: Total simulation time in milliseconds
            dt: Timestep (default 0.1ms)
            report_interval_ms: Print status every N ms
        """
        n_steps = int(duration_ms / dt)
        last_report = self.total_time_ms
        start_wall = time.time()
        
        print(f"\n[CogMind] Running {duration_ms:.0f}ms simulation...")
        print(f"  dt={dt}ms, steps={n_steps:,}")
        
        for i in range(n_steps):
            self.step(dt=dt)
            
            # Status report
            if self.total_time_ms - last_report >= report_interval_ms:
                wall_elapsed = time.time() - start_wall
                sim_speed = self.total_time_ms / (wall_elapsed * 1000)
                
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                state_name = self.neuromod.get_cognitive_state()
                
                print(f"\n  ─── t={self.total_time_ms:.0f}ms ───")
                print(f"  Speed: {sim_speed:.3f}x realtime")
                print(f"  Firing rate: {metrics['avg_firing_rate']:.1f} Hz")
                print(f"  Cognitive state: {state_name}")
                print(f"  Neuromod: {self.neuromod.state}")
                print(f"  Episodes stored: {metrics['hippocampus_episodes']}")
                print(f"  Reward: {metrics['cumulative_reward']:.2f}")
                
                last_report = self.total_time_ms
        
        total_wall = time.time() - start_wall
        print(f"\n[CogMind] ✓ Complete in {total_wall:.1f}s "
              f"({duration_ms/1000:.1f}s sim / {total_wall:.1f}s wall = "
              f"{duration_ms/(total_wall*1000):.3f}x realtime)")
    
    def _decode_motor(self, rates: np.ndarray) -> np.ndarray:
        """Decode motor output from frontal module activity."""
        module_names = self.topology.parameters.get("module_names", [])
        
        # Find frontal_R module (motor planning)
        for i, name in enumerate(module_names):
            if "frontal_R" in name:
                mask = self.topology.neuron_modules == i
                frontal_rates = rates[mask]
                
                # Simple decoding: average activity in bins
                n_bins = self.environment.n_motor
                bin_size = max(1, len(frontal_rates) // n_bins)
                motor = np.array([
                    np.mean(frontal_rates[j*bin_size:(j+1)*bin_size])
                    for j in range(n_bins)
                ])
                return motor
        
        return np.zeros(self.environment.n_motor)
    
    def _collect_metrics(self) -> Dict:
        """Collect current simulation metrics."""
        rates = self.engine.get_firing_rates()
        module_activity = self.engine.get_module_activity()
        hippo_stats = self.hippocampus.get_stats()
        osc_state = self.thalamus.get_oscillation_state()
        
        return {
            "time_ms": self.total_time_ms,
            "avg_firing_rate": float(np.mean(rates)),
            "std_firing_rate": float(np.std(rates)),
            "module_activity": module_activity,
            "neuromod_state": self.neuromod.state.as_dict(),
            "cognitive_state": self.neuromod.get_cognitive_state(),
            "oscillation_state": osc_state,
            "hippocampus_episodes": hippo_stats.get("n_episodes", 0),
            "cumulative_reward": self.environment.state.cumulative_reward,
            "weight_stats": self.stdp.get_weight_stats(self.topology.connection_weights),
        }
    
    def save_state(self, path: str):
        """Save complete instance state to disk."""
        self.topology.save(path.replace('.npz', '_topology.npz'))
        
        state = {
            "total_time_ms": self.total_time_ms,
            "total_steps": self.total_steps,
            "neuromod_state": self.neuromod.state.as_dict(),
            "metrics_history": self.metrics_history,
            "hippo_stats": self.hippocampus.get_stats(),
        }
        
        with open(path.replace('.npz', '_state.json'), 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        print(f"[CogMind] State saved to {path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CogMind — Cognitive Emulation Runner")
    parser.add_argument("signature", help="Path to signature.json")
    parser.add_argument("-n", "--neurons", type=int, default=100_000,
                        help="Total neurons (default: 100K)")
    parser.add_argument("-d", "--duration", type=float, default=5000.0,
                        help="Simulation duration in ms (default: 5000)")
    parser.add_argument("-e", "--env", default="pattern_recognition",
                        choices=["pattern_recognition", "sequence_learning", "sensorimotor"],
                        help="Environment type")
    parser.add_argument("-o", "--output", default="cogmind_checkpoint",
                        help="Output checkpoint path")
    parser.add_argument("-s", "--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    instance = CogMindInstance.from_signature(
        args.signature,
        n_neurons=args.neurons,
        env_type=args.env,
        seed=args.seed,
    )
    
    instance.run(duration_ms=args.duration)
    instance.save_state(args.output)


if __name__ == "__main__":
    main()

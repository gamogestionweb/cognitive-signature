"""
topology_generator.py — De Cognitive Signature a Topología de Red
=================================================================

Prioridad: 1 — EMPIEZA AQUÍ

Input:  signature.json (output de brain_analyzer.py)
Output: Grafo de conectividad con N nodos, pesos iniciales, tipos E/I,
        asignación a módulos corticales.

Convierte las métricas cerebrales reales en parámetros de red:
  - Distribución regional → module_sizes[]
  - Ratio gris/blanca → local_vs_long_connectivity
  - Girificación → neurons_per_column, columns_per_module
  - Asimetría hemisférica → hemisphere_ratio
  - Volumen ventricular → pruning_rate
  - CSF → homeostatic_rate
"""

import json
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# ──────────────────────────────────────────────────────────────────
# Population norms (from peer-reviewed neuroimaging studies)
# ──────────────────────────────────────────────────────────────────
POPULATION_NORMS = {
    "gray_white_ratio": 1.45,
    "gyrification_index": 2.55,
    "hemispheric_asymmetry": 2.0,  # % 
    "ventricular_volume_ml": 25.0,
    "csf_volume_ml": 140.0,
}

# Default regional distribution (population average)
DEFAULT_REGIONAL_DISTRIBUTION = {
    "frontal": 0.321,
    "parietal": 0.248,
    "temporal": 0.215,
    "occipital": 0.142,
    "central": 0.074,
}

# Inter-module connectivity weights (based on tract-tracing literature)
# Higher = stronger baseline connection between modules
INTER_MODULE_CONNECTIVITY = {
    ("frontal", "parietal"): 0.8,
    ("frontal", "temporal"): 0.6,
    ("frontal", "occipital"): 0.3,
    ("frontal", "central"): 0.9,
    ("parietal", "temporal"): 0.7,
    ("parietal", "occipital"): 0.6,
    ("parietal", "central"): 0.7,
    ("temporal", "occipital"): 0.5,
    ("temporal", "central"): 0.5,
    ("occipital", "central"): 0.4,
}


@dataclass
class NeuronInfo:
    """Information about a single neuron in the network."""
    neuron_id: int
    module: str           # e.g., "frontal_L", "temporal_R"
    hemisphere: str       # "L" or "R" 
    neuron_type: str      # "excitatory" or "inhibitory"
    column_id: int        # minicolumn assignment
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class NetworkTopology:
    """Complete network topology generated from a Cognitive Signature."""
    
    # Metadata
    signature_id: str
    n_total: int
    timestamp: str
    
    # Module assignments
    module_sizes: Dict[str, int] = field(default_factory=dict)
    
    # Neuron properties
    neuron_modules: np.ndarray = field(default_factory=lambda: np.array([]))
    neuron_types: np.ndarray = field(default_factory=lambda: np.array([]))  # 1=E, -1=I
    neuron_hemispheres: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Connectivity (sparse representation)
    connections_pre: np.ndarray = field(default_factory=lambda: np.array([]))  # presynaptic indices
    connections_post: np.ndarray = field(default_factory=lambda: np.array([]))  # postsynaptic indices
    connection_weights: np.ndarray = field(default_factory=lambda: np.array([]))  # initial weights
    connection_delays: np.ndarray = field(default_factory=lambda: np.array([]))  # ms
    
    # Derived parameters
    parameters: Dict = field(default_factory=dict)
    
    def summary(self) -> str:
        n_exc = int(np.sum(self.neuron_types == 1))
        n_inh = int(np.sum(self.neuron_types == -1))
        n_conn = len(self.connections_pre)
        return (
            f"NetworkTopology(signature={self.signature_id}, "
            f"neurons={self.n_total} [E:{n_exc}/I:{n_inh}], "
            f"synapses={n_conn:,}, "
            f"modules={list(self.module_sizes.keys())})"
        )
    
    def save(self, path: str):
        """Save topology to disk (numpy compressed format)."""
        np.savez_compressed(
            path,
            signature_id=self.signature_id,
            n_total=self.n_total,
            module_sizes=json.dumps(self.module_sizes),
            neuron_modules=self.neuron_modules,
            neuron_types=self.neuron_types,
            neuron_hemispheres=self.neuron_hemispheres,
            connections_pre=self.connections_pre,
            connections_post=self.connections_post,
            connection_weights=self.connection_weights,
            connection_delays=self.connection_delays,
            parameters=json.dumps(self.parameters),
        )
        print(f"[TopologyGenerator] Saved topology to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'NetworkTopology':
        """Load topology from disk."""
        data = np.load(path, allow_pickle=True)
        topo = cls(
            signature_id=str(data['signature_id']),
            n_total=int(data['n_total']),
            timestamp="",
        )
        topo.module_sizes = json.loads(str(data['module_sizes']))
        topo.neuron_modules = data['neuron_modules']
        topo.neuron_types = data['neuron_types']
        topo.neuron_hemispheres = data['neuron_hemispheres']
        topo.connections_pre = data['connections_pre']
        topo.connections_post = data['connections_post']
        topo.connection_weights = data['connection_weights']
        topo.connection_delays = data['connection_delays']
        topo.parameters = json.loads(str(data['parameters']))
        return topo


class TopologyGenerator:
    """
    Converts a Cognitive Signature (signature.json) into a network topology
    parameterized by real brain structural data.
    
    The key insight: we don't copy every synapse. We create a system that,
    given the same macro structure (your Cognitive Signature), the same
    learning rules (biological plasticity), and sufficient environmental 
    exposure, converges toward similar processing patterns.
    
    Usage:
        gen = TopologyGenerator("path/to/signature.json")
        topology = gen.generate(n_total=100_000)
        topology.save("my_brain_topology.npz")
    """
    
    def __init__(self, signature_path: str):
        """
        Args:
            signature_path: Path to signature.json from brain_analyzer.py
        """
        self.signature_path = Path(signature_path)
        self.signature = self._load_signature()
        self.params = self._extract_network_params()
    
    def _load_signature(self) -> dict:
        """Load and validate the cognitive signature JSON."""
        with open(self.signature_path) as f:
            sig = json.load(f)
        
        # Validate required fields
        required = ["signature_id"]
        for field_name in required:
            if field_name not in sig:
                raise ValueError(f"signature.json missing required field: {field_name}")
        
        print(f"[TopologyGenerator] Loaded signature: {sig.get('signature_id', 'unknown')}")
        return sig
    
    def _extract_network_params(self) -> dict:
        """
        Convert brain metrics from signature.json into network parameters.
        
        Each metric maps to a specific network property:
            Regional distribution → module_sizes
            Gray/white ratio → local_vs_long connectivity balance
            Gyrification → columns per module
            Hemispheric asymmetry → L/R neuron allocation
            Ventricular volume → synaptic pruning rate
            CSF volume → homeostatic regulation rate
        """
        sig = self.signature
        metrics = sig.get("metrics", sig)  # handle both nested and flat formats
        
        # ── Regional Distribution ──
        regional = metrics.get("regional_distribution", {})
        if isinstance(regional, dict) and "frontal" in regional:
            # Direct percentages
            region_dist = {
                "frontal": regional.get("frontal", 32.1) / 100.0,
                "parietal": regional.get("parietal", 24.8) / 100.0,
                "temporal": regional.get("temporal", 21.5) / 100.0,
                "occipital": regional.get("occipital", 14.2) / 100.0,
                "central": regional.get("central", 7.4) / 100.0,
            }
        else:
            region_dist = DEFAULT_REGIONAL_DISTRIBUTION.copy()
        
        # Normalize to sum to 1.0
        total = sum(region_dist.values())
        region_dist = {k: v / total for k, v in region_dist.items()}
        
        # ── Gray/White Ratio → Connectivity Balance ──
        gw_ratio = float(metrics.get("gray_white_ratio", 
                         metrics.get("gw_ratio", POPULATION_NORMS["gray_white_ratio"])))
        
        # Higher gray/white = more local processing, less long-range
        # Normal (1.45) → 60/40 local/long. High (3.03) → 75/25
        norm_gw = POPULATION_NORMS["gray_white_ratio"]
        local_fraction = np.clip(0.60 + 0.15 * (gw_ratio - norm_gw) / norm_gw, 0.50, 0.90)
        
        # ── Gyrification → Columns per Module ──
        gyrification = float(metrics.get("gyrification_index",
                            metrics.get("gyrification", POPULATION_NORMS["gyrification_index"])))
        norm_gyri = POPULATION_NORMS["gyrification_index"]
        
        # More gyrification → more minicolumns → more parallel processing
        base_neurons_per_column = 80  # biological: ~80-100 per minicolumn
        column_multiplier = gyrification / norm_gyri
        
        # ── Hemispheric Asymmetry ──
        asymmetry_pct = float(metrics.get("hemispheric_asymmetry",
                              metrics.get("asymmetry", POPULATION_NORMS["hemispheric_asymmetry"])))
        # Positive = left dominant
        left_fraction = 0.50 + (asymmetry_pct / 100.0) / 2.0
        
        # ── Ventricular Volume → Pruning Rate ──
        ventricle_ml = float(metrics.get("ventricular_volume",
                             metrics.get("ventricles_ml", POPULATION_NORMS["ventricular_volume_ml"])))
        norm_vent = POPULATION_NORMS["ventricular_volume_ml"]
        # Higher ventricles → more aggressive pruning
        pruning_rate = np.clip(0.01 * (ventricle_ml / norm_vent), 0.005, 0.05)
        
        # ── CSF → Homeostatic Rate ──
        csf_ml = float(metrics.get("csf_volume",
                       metrics.get("csf_ml", POPULATION_NORMS["csf_volume_ml"])))
        norm_csf = POPULATION_NORMS["csf_volume_ml"]
        # Higher CSF → more aggressive homeostatic regulation
        homeostatic_rate = np.clip(0.1 * (csf_ml / norm_csf), 0.05, 0.30)
        
        params = {
            "regional_distribution": region_dist,
            "local_fraction": float(local_fraction),
            "long_range_fraction": float(1.0 - local_fraction),
            "gyrification_multiplier": float(column_multiplier),
            "neurons_per_column": base_neurons_per_column,
            "left_hemisphere_fraction": float(left_fraction),
            "right_hemisphere_fraction": float(1.0 - left_fraction),
            "pruning_rate": float(pruning_rate),
            "homeostatic_rate": float(homeostatic_rate),
            "ei_ratio": 0.80,  # 80% excitatory, 20% inhibitory (Dale's law)
            "p_local": 0.10,   # intra-module connection probability
            "p_long": 0.02,    # inter-module connection probability
            "rewire_prob": 0.15,  # Watts-Strogatz small-world rewiring
            # Source metrics for reference
            "_source_gw_ratio": float(gw_ratio),
            "_source_gyrification": float(gyrification),
            "_source_asymmetry_pct": float(asymmetry_pct),
            "_source_ventricle_ml": float(ventricle_ml),
            "_source_csf_ml": float(csf_ml),
        }
        
        print(f"[TopologyGenerator] Extracted network parameters:")
        print(f"  Local connectivity:  {local_fraction:.0%}")
        print(f"  Column multiplier:   {column_multiplier:.2f}x")
        print(f"  L/R hemisphere:      {left_fraction:.1%} / {1-left_fraction:.1%}")
        print(f"  Pruning rate:        {pruning_rate:.4f}")
        print(f"  Homeostatic rate:    {homeostatic_rate:.3f}")
        
        return params
    
    def generate(self, n_total: int = 100_000, seed: int = 42) -> NetworkTopology:
        """
        Generate the complete network topology.
        
        Args:
            n_total: Total number of neurons (min 10K for meaningful dynamics)
            seed: Random seed for reproducibility
            
        Returns:
            NetworkTopology with all neuron assignments and connectivity
        """
        rng = np.random.default_rng(seed)
        
        print(f"\n[TopologyGenerator] Generating topology with N={n_total:,} neurons...")
        
        # ──────────────────────────────────────────
        # Step 1: Assign neurons to modules
        # ──────────────────────────────────────────
        module_sizes, neuron_modules, neuron_hemispheres = self._assign_modules(
            n_total, rng
        )
        
        # ──────────────────────────────────────────
        # Step 2: Assign E/I types (Dale's law)
        # ──────────────────────────────────────────
        neuron_types = self._assign_ei_types(n_total, rng)
        
        # ──────────────────────────────────────────
        # Step 3: Generate connectivity
        # ──────────────────────────────────────────
        pre, post, weights, delays = self._generate_connectivity(
            n_total, neuron_modules, neuron_types, module_sizes, rng
        )
        
        # ──────────────────────────────────────────
        # Step 4: Apply small-world rewiring
        # ──────────────────────────────────────────
        pre, post, weights, delays = self._apply_small_world_rewiring(
            pre, post, weights, delays, neuron_modules, n_total, rng
        )
        
        # Build topology object
        from datetime import datetime
        topology = NetworkTopology(
            signature_id=self.signature.get("signature_id", "unknown"),
            n_total=n_total,
            timestamp=datetime.now().isoformat(),
            module_sizes=module_sizes,
            neuron_modules=neuron_modules,
            neuron_types=neuron_types,
            neuron_hemispheres=neuron_hemispheres,
            connections_pre=pre,
            connections_post=post,
            connection_weights=weights,
            connection_delays=delays,
            parameters=self.params,
        )
        
        print(f"\n[TopologyGenerator] ✓ Generated: {topology.summary()}")
        return topology
    
    def _assign_modules(
        self, n_total: int, rng: np.random.Generator
    ) -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
        """
        Assign each neuron to a cortical module and hemisphere.
        
        Uses regional distribution from Cognitive Signature and hemispheric
        asymmetry to determine the exact count per module.
        """
        region_dist = self.params["regional_distribution"]
        left_frac = self.params["left_hemisphere_fraction"]
        right_frac = self.params["right_hemisphere_fraction"]
        
        module_sizes = {}
        assignments = []  # (module_name, hemisphere)
        
        for region, fraction in region_dist.items():
            n_region = int(n_total * fraction)
            
            if region == "central":
                # Central region: no strong lateralization
                n_left = n_region // 2
                n_right = n_region - n_left
            else:
                n_left = int(n_region * left_frac)
                n_right = n_region - n_left
            
            module_sizes[f"{region}_L"] = n_left
            module_sizes[f"{region}_R"] = n_right
            
            assignments.extend([(f"{region}_L", "L")] * n_left)
            assignments.extend([(f"{region}_R", "R")] * n_right)
        
        # Adjust to exact n_total
        while len(assignments) < n_total:
            assignments.append(("central_L", "L"))
            module_sizes["central_L"] += 1
        while len(assignments) > n_total:
            assignments.pop()
            module_sizes["central_L"] -= 1
        
        # Encode as numpy arrays
        module_names = sorted(module_sizes.keys())
        module_to_id = {name: i for i, name in enumerate(module_names)}
        
        neuron_modules = np.array([module_to_id[a[0]] for a in assignments], dtype=np.int32)
        neuron_hemispheres = np.array([0 if a[1] == "L" else 1 for a in assignments], dtype=np.int8)
        
        # Store module name mapping in params
        self.params["module_names"] = module_names
        self.params["module_to_id"] = module_to_id
        
        print(f"  Modules: {len(module_sizes)} ({len(module_names)} with hemispheres)")
        for name, count in sorted(module_sizes.items(), key=lambda x: -x[1]):
            print(f"    {name:>15}: {count:>7,} neurons ({count/n_total:.1%})")
        
        return module_sizes, neuron_modules, neuron_hemispheres
    
    def _assign_ei_types(
        self, n_total: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Assign excitatory (1) or inhibitory (-1) type to each neuron.
        Follows Dale's law: ~80% excitatory, ~20% inhibitory.
        """
        ei_ratio = self.params["ei_ratio"]
        types = np.ones(n_total, dtype=np.int8)
        n_inhibitory = int(n_total * (1 - ei_ratio))
        inhibitory_indices = rng.choice(n_total, size=n_inhibitory, replace=False)
        types[inhibitory_indices] = -1
        
        n_exc = int(np.sum(types == 1))
        n_inh = int(np.sum(types == -1))
        print(f"  E/I types: {n_exc:,} excitatory ({n_exc/n_total:.0%}), "
              f"{n_inh:,} inhibitory ({n_inh/n_total:.0%})")
        
        return types
    
    def _generate_connectivity(
        self,
        n_total: int,
        neuron_modules: np.ndarray,
        neuron_types: np.ndarray,
        module_sizes: Dict[str, int],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate sparse connectivity matrix.
        
        Intra-module: p=0.10, small delays (1-3ms)
        Inter-module: p=0.02, larger delays (3-15ms), weighted by tract strength
        
        Uses batched generation for memory efficiency.
        """
        p_local = self.params["p_local"]
        p_long = self.params["p_long"]
        local_frac = self.params["local_fraction"]
        module_names = self.params["module_names"]
        
        pre_list = []
        post_list = []
        weight_list = []
        delay_list = []
        
        # Get neuron indices per module
        module_indices = {}
        for mid, name in enumerate(module_names):
            module_indices[mid] = np.where(neuron_modules == mid)[0]
        
        total_synapses = 0
        
        # ── Intra-module connections (local) ──
        print(f"  Generating intra-module connections (p={p_local})...")
        for mid, indices in module_indices.items():
            n_mod = len(indices)
            if n_mod < 2:
                continue
            
            # Expected number of connections
            n_expected = int(n_mod * n_mod * p_local)
            
            # Generate random connections
            pre_local = rng.choice(indices, size=n_expected)
            post_local = rng.choice(indices, size=n_expected)
            
            # Remove self-connections
            valid = pre_local != post_local
            pre_local = pre_local[valid]
            post_local = post_local[valid]
            
            # Weights: excitatory neurons have positive weights, inhibitory negative
            pre_types = neuron_types[pre_local]
            w = rng.lognormal(mean=-2.0, sigma=0.5, size=len(pre_local))
            w = w * pre_types  # sign matches neuron type
            
            # Delays: local connections are fast (1-3 ms)
            d = rng.uniform(1.0, 3.0, size=len(pre_local))
            
            pre_list.append(pre_local)
            post_list.append(post_local)
            weight_list.append(w.astype(np.float32))
            delay_list.append(d.astype(np.float32))
            total_synapses += len(pre_local)
        
        # ── Inter-module connections (long-range) ──
        print(f"  Generating inter-module connections (p={p_long})...")
        for i, name_i in enumerate(module_names):
            for j, name_j in enumerate(module_names):
                if i >= j:
                    continue
                
                indices_i = module_indices[i]
                indices_j = module_indices[j]
                
                if len(indices_i) < 1 or len(indices_j) < 1:
                    continue
                
                # Get base region names (strip hemisphere suffix)
                region_i = name_i.rsplit("_", 1)[0]
                region_j = name_j.rsplit("_", 1)[0]
                
                # Lookup tract strength
                key = tuple(sorted([region_i, region_j]))
                tract_strength = INTER_MODULE_CONNECTIVITY.get(key, 0.3)
                
                # Same hemisphere = stronger connection
                hemi_i = name_i.rsplit("_", 1)[1] if "_" in name_i else ""
                hemi_j = name_j.rsplit("_", 1)[1] if "_" in name_j else ""
                if hemi_i == hemi_j:
                    hemi_factor = 1.0
                else:
                    hemi_factor = 0.5  # commissural connections are weaker
                
                effective_p = p_long * tract_strength * hemi_factor
                n_expected = int(len(indices_i) * len(indices_j) * effective_p)
                n_expected = min(n_expected, 500_000)  # cap for memory
                
                if n_expected < 1:
                    continue
                
                pre_long = rng.choice(indices_i, size=n_expected)
                post_long = rng.choice(indices_j, size=n_expected)
                
                # Bidirectional: also add reverse
                pre_rev = rng.choice(indices_j, size=n_expected)
                post_rev = rng.choice(indices_i, size=n_expected)
                
                pre_long = np.concatenate([pre_long, pre_rev])
                post_long = np.concatenate([post_long, post_rev])
                
                # Long-range weights: slightly stronger initial weights
                pre_types_long = neuron_types[pre_long]
                w = rng.lognormal(mean=-1.5, sigma=0.5, size=len(pre_long))
                w = w * pre_types_long * float(tract_strength)
                
                # Long-range delays: 3-15ms (conduction through white matter)
                d = rng.uniform(3.0, 15.0, size=len(pre_long))
                
                pre_list.append(pre_long)
                post_list.append(post_long)
                weight_list.append(w.astype(np.float32))
                delay_list.append(d.astype(np.float32))
                total_synapses += len(pre_long)
        
        # Concatenate all connections
        all_pre = np.concatenate(pre_list)
        all_post = np.concatenate(post_list)
        all_weights = np.concatenate(weight_list)
        all_delays = np.concatenate(delay_list)
        
        print(f"  Total synapses: {total_synapses:,} "
              f"(~{total_synapses/n_total:.0f} per neuron)")
        
        return all_pre, all_post, all_weights, all_delays
    
    def _apply_small_world_rewiring(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        weights: np.ndarray,
        delays: np.ndarray,
        neuron_modules: np.ndarray,
        n_total: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Watts-Strogatz-style rewiring to create small-world topology.
        
        High clustering (local circuits) + short path lengths (long-range shortcuts).
        This is a key feature of biological cortical networks.
        """
        rewire_prob = self.params["rewire_prob"]
        n_connections = len(pre)
        n_rewire = int(n_connections * rewire_prob)
        
        print(f"  Small-world rewiring: {n_rewire:,} connections ({rewire_prob:.0%})...")
        
        # Select random connections to rewire
        rewire_indices = rng.choice(n_connections, size=n_rewire, replace=False)
        
        # Rewire post-synaptic targets to random neurons
        new_targets = rng.choice(n_total, size=n_rewire)
        
        # Ensure no self-connections
        for i, idx in enumerate(rewire_indices):
            while new_targets[i] == pre[idx]:
                new_targets[i] = rng.integers(0, n_total)
        
        post[rewire_indices] = new_targets
        
        # Update delays for rewired connections (they become long-range)
        delays[rewire_indices] = rng.uniform(3.0, 15.0, size=n_rewire).astype(np.float32)
        
        return pre, post, weights, delays


# ──────────────────────────────────────────────────────────────────
# CLI interface
# ──────────────────────────────────────────────────────────────────
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate network topology from Cognitive Signature"
    )
    parser.add_argument("signature", help="Path to signature.json")
    parser.add_argument("-n", "--neurons", type=int, default=100_000,
                        help="Total number of neurons (default: 100K)")
    parser.add_argument("-o", "--output", default="topology.npz",
                        help="Output path for topology file")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    gen = TopologyGenerator(args.signature)
    topology = gen.generate(n_total=args.neurons, seed=args.seed)
    topology.save(args.output)
    
    print(f"\n✓ Topology saved to {args.output}")
    print(f"  Next step: snn_engine.py to instantiate this as a Spiking Neural Network")


if __name__ == "__main__":
    main()

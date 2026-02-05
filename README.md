# 🧠 Cognitive Signature + CogMind

**Extract cognitive fingerprints from brain scans → Generate cognitive emulations.**

This project has two integrated layers:

1. **Cognitive Signature** — Analyze brain CT/MRI DICOM scans to extract structural metrics (volumes, ratios, gyrification, asymmetry)
2. **CogMind** — Use those metrics as structural seeds to generate Spiking Neural Networks that emulate cognitive processing

## Architecture Overview

```
╔══════════════════════════════════════════════════════════════════╗
║                    CogMind Architecture                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │          LAYER 7: VIRTUAL ENVIRONMENT (I/O)                │  ║
║  │   Simulated sensors ←→ Simulated body interface            │  ║
║  └──────────────────────┬─────────────────────────────────────┘  ║
║                         │                                        ║
║  ┌──────────────────────┴─────────────────────────────────────┐  ║
║  │          LAYER 6: NEUROMODULATION                          │  ║
║  │   Dopamine · Serotonin · Noradrenaline · Acetylcholine     │  ║
║  └──────────┬───────────────────────────────┬─────────────────┘  ║
║             │                               │                    ║
║  ┌──────────┴───────────────────────────────┴─────────────────┐  ║
║  │          LAYER 5: CORTEX (Recurrent SNN Modules)           │  ║
║  │   FRONTAL ←→ PARIETAL ←→ TEMPORAL ←→ OCCIPITAL            │  ║
║  │   + CENTRAL hub · Parametrized by Cognitive Signature      │  ║
║  └──────────┬───────────────────────────────┬─────────────────┘  ║
║             │                               │                    ║
║  ┌──────────┴───────────────────────────────┴─────────────────┐  ║
║  │          LAYER 4: THALAMUS (Router/Oscillator/Gate)        │  ║
║  │   Theta · Alpha · Gamma oscillations + Attention gating    │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │   LAYER 3: HIPPOCAMPUS (Episodic memory + Replay)          │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │   LAYERS 1-2: COGNITIVE SIGNATURE (Structural Seed)        │  ║
║  │   Volumes + Ratios + Gyrification + Asymmetry → Topology   │  ║
║  └────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════╝
```

## The Key Insight

We don't replicate every synapse. We create a system that, given the **same macro structure** (your Cognitive Signature), the **same learning rules** (biological plasticity), and **sufficient environmental exposure**, converges toward similar processing patterns. Like giving a musician the same instrument, training, and influences — they won't play identically, but they'll play similarly.

## Project Structure

```
cognitive-signature/
├── src/
│   ├── brain_analyzer.py          # Original: DICOM → signature.json
│   └── cogmind/                   # NEW: CogMind modules
│       ├── __init__.py
│       ├── topology_generator.py  # signature.json → Network graph
│       ├── snn_engine.py          # Graph → Running SNN (LIF neurons)
│       ├── plasticity.py          # STDP + R-STDP learning rules
│       ├── thalamus.py            # Oscillator + Router + Attention
│       ├── neuromodulation.py     # DA / 5HT / NE / ACh modulation
│       ├── hippocampus.py         # Episodic memory + Replay
│       ├── environment.py         # Virtual sensory I/O
│       └── cogmind_runner.py      # Main integration runner
├── examples/
│   └── example_signature.json     # Example signature for testing
├── docs/
│   └── cogmind_architecture.html  # Full architecture documentation
├── requirements.txt
├── requirements_cogmind.txt
└── setup.py
```

## Quick Start

### 1. Generate Cognitive Signature (existing)

```python
from src.brain_analyzer import CognitiveSignatureAnalyzer

analyzer = CognitiveSignatureAnalyzer("path/to/dicom/folder")
analyzer.run_analysis()
analyzer.generate_report("./output")
# → produces signature.json
```

### 2. Generate Network Topology (new)

```python
from cogmind.topology_generator import TopologyGenerator

gen = TopologyGenerator("output/signature.json")
topology = gen.generate(n_total=100_000)
topology.save("my_brain_topology.npz")
```

### 3. Run CogMind Emulation (new)

```python
from cogmind.cogmind_runner import CogMindInstance

# One-line: signature → running emulation
instance = CogMindInstance.from_signature(
    "output/signature.json",
    n_neurons=100_000,
    env_type="pattern_recognition"
)

# Run 5 seconds of simulation
instance.run(duration_ms=5000)
```

### CLI Usage

```bash
# Generate topology only
python -m cogmind.topology_generator signature.json -n 100000 -o topology.npz

# Full emulation
python -m cogmind.cogmind_runner signature.json -n 100000 -d 5000 -e pattern_recognition
```

## Module Reference

| Module | Input | Output | Priority |
|--------|-------|--------|----------|
| `topology_generator.py` | signature.json | Network graph (N nodes, weights, E/I types) | 1 |
| `snn_engine.py` | Network graph | Running SNN with LIF adaptive neurons | 2 |
| `plasticity.py` | STDP config | Weight updates via spike-timing correlations | 3 |
| `thalamus.py` | Oscillation config | Sync signals (theta/alpha/gamma) → cortex | 4 |
| `neuromodulation.py` | Environment state | DA/5HT/NE/ACh levels → global modulation | 5 |
| `hippocampus.py` | Cortical patterns | Episodic memory, replay, consolidation | 6 |
| `environment.py` | Sensor/actuator interface | Sensory streams + motor feedback | 7 |

## How Signature Metrics Map to Network Parameters

| Your Metric | Example Value | Network Parameter | Effect |
|-------------|---------------|-------------------|--------|
| Regional distribution | F:32% P:25% T:21% O:14% C:7% | `module_sizes[]` | Neurons per cortical module |
| Gray/white ratio | 3.03 | `local_vs_long_connectivity` | 75% local / 25% long-range |
| Gyrification index | 5.33 | `columns_per_module` | More parallel processing |
| Hemispheric asymmetry | 3.30% left | `hemisphere_ratio` | Left hemisphere +3.3% neurons |
| Ventricular volume | 27.48 ml (p57) | `pruning_rate` | Standard synaptic pruning |
| CSF volume | 160.58 ml (p75) | `homeostatic_rate` | Slightly aggressive regulation |

## What CogMind Has That LLMs Don't

1. **Genuine recurrence** — Thalamo-cortical loops, not just residual connections
2. **Real-time plasticity** — STDP modifies weights during processing
3. **Temporal dynamics** — Oscillations create integration windows
4. **Global neuromodulation** — States (alert, relaxed, focused, creative)
5. **Intrinsic causality** — The system modifies itself

These are the 5 ingredients that IIT (Integrated Information Theory) identifies as necessary for generating Φ (integrated information).

## Scaling

| Scale | Neurons | Synapses | Hardware | Real-time? | Equivalence |
|-------|---------|----------|----------|------------|-------------|
| **Prototype** | 100K | 100M | 1× GPU A100 | ~10-100× slower | ~minicolumn |
| **Alpha** | 1M | 1B | 8× GPU cluster | ~100-1000× slower | ~small region |
| **Beta** | 100M | 100B | Supercomputer | ~1000× slower | ~mouse brain |
| **Full** | 86B | ~100T | Exascale + neuromorphic | Unknown | ~human brain |

## Installation

```bash
git clone https://github.com/gamogestionweb/cognitive-signature.git
cd cognitive-signature

# Core dependencies
pip install -r requirements.txt
pip install -r requirements_cogmind.txt

# Or install as package
pip install -e ".[full]"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

Priority areas for contribution:
- Brian2/NEST backend integration for `snn_engine.py`
- More environment types in `environment.py`
- Visualization dashboards for real-time simulation monitoring
- Benchmarking against biological data (spike statistics, oscillation power spectra)

## License

MIT License — See [LICENSE](LICENSE) for details.

## Author

**Daniel Gamo** ([@gamogestionweb](https://github.com/gamogestionweb))

---

*"Your brain's structure is the seed. The architecture does the rest."*

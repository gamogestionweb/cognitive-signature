# 🧠 Cognitive Signature → CogMind

<div align="center">

**From Brain Scan to Mind Emulation**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[What is this?](#what-is-this) • [Quick Start](#quick-start) • [Architecture](#architecture) • [The Science](#the-science) • [Modules](#modules)

---

### This is not another chatbot. This is cognitive emulation.

</div>

---

## What is this?

Two integrated systems:

| Layer | What it does | Output |
|-------|--------------|--------|
| **Cognitive Signature** | Analyzes brain CT/MRI scans | `signature.json` — your brain's structural fingerprint |
| **CogMind** | Takes that fingerprint and builds a spiking neural network | A running mind emulation |

```
DICOM Scan → signature.json → Spiking Neural Network → Cognitive Emulation
```

**The key insight:** We don't replicate every synapse. We create a system that, given the same macro structure (your signature), the same learning rules (biological plasticity), and environmental exposure, converges toward similar processing patterns.

---

## What CogMind Has That LLMs Don't

| Feature | LLMs (GPT, Claude, etc.) | CogMind |
|---------|--------------------------|---------|
| **Recurrence** | Residual connections only | Real thalamo-cortical loops |
| **Plasticity** | Frozen after training | STDP modifies weights *during* processing |
| **Temporal dynamics** | None (stateless) | Theta/Alpha/Gamma oscillations |
| **Global states** | None | Neuromodulation: alert, relaxed, focused, creative |
| **Self-modification** | Impossible | The system changes itself |

These are the 5 ingredients that IIT (Integrated Information Theory) identifies as necessary for generating Φ (integrated information).

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         CogMind Architecture                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │              LAYER 7: VIRTUAL ENVIRONMENT (I/O)                    │  ║
║  │         Simulated sensors ←→ Motor output ←→ Reward signals        │  ║
║  └──────────────────────────┬─────────────────────────────────────────┘  ║
║                             │                                            ║
║  ┌──────────────────────────┴─────────────────────────────────────────┐  ║
║  │              LAYER 6: NEUROMODULATION                              │  ║
║  │       Dopamine · Serotonin · Noradrenaline · Acetylcholine         │  ║
║  │       (The 4 "global dials" that create mental states)             │  ║
║  └──────────┬───────────────────────────────────┬─────────────────────┘  ║
║             │                                   │                        ║
║  ┌──────────┴───────────────────────────────────┴─────────────────────┐  ║
║  │              LAYER 5: CORTEX (Recurrent SNN Modules)               │  ║
║  │       FRONTAL ←→ PARIETAL ←→ TEMPORAL ←→ OCCIPITAL                 │  ║
║  │       100,000+ LIF neurons with adaptive thresholds                │  ║
║  └──────────┬───────────────────────────────────┬─────────────────────┘  ║
║             │                                   │                        ║
║  ┌──────────┴───────────────────────────────────┴─────────────────────┐  ║
║  │              LAYER 4: THALAMUS                                     │  ║
║  │       Oscillator (θ/α/γ) + Router + Attention Gate                 │  ║
║  │       Creates the temporal binding windows for integration         │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │              LAYER 3: PLASTICITY                                   │  ║
║  │       STDP + Reward-Modulated STDP + Homeostatic regulation        │  ║
║  │       The network learns and adapts in real-time                   │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │              LAYER 2: HIPPOCAMPUS                                  │  ║
║  │       Episodic memory · Pattern separation · Replay · Consolidation║  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │              LAYER 1: COGNITIVE SIGNATURE (Structural Seed)        │  ║
║  │       Your brain scan → volumes, ratios, gyrification, asymmetry   │  ║
║  │       This is what makes YOUR emulation unique                     │  ║
║  └────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/gamogestionweb/cognitive-signature.git
cd cognitive-signature
pip install -r requirements.txt
```

### Step 1: Extract Cognitive Signature (from brain scan)

```python
from src.brain_analyzer import CognitiveSignatureAnalyzer

analyzer = CognitiveSignatureAnalyzer("path/to/dicom/folder")
analyzer.run_analysis()
analyzer.generate_report("./output")
# → produces signature.json
```

### Step 2: Run CogMind Emulation

```python
from cogmind_runner import CogMindInstance

# One line: signature → running mind
instance = CogMindInstance.from_signature(
    "output/signature.json",
    n_neurons=100_000,
    env_type="pattern_recognition"
)

# Run 5 seconds of cognitive emulation
instance.run(duration_ms=5000)
```

### What you'll see:

```
╔══════════════════════════════════════════════╗
║        CogMind — Cognitive Emulation         ║
║     From Cognitive Signature to Mind          ║
╚══════════════════════════════════════════════╝

[TopologyGenerator] Loaded signature: BRAIN-5C13EB2BADEB
  Modules: frontal_L, frontal_R, parietal_L, parietal_R, temporal_L...
  Neurons: 100,000 [E:80,000 / I:20,000]
  Synapses: 1,842,567

[CogMind] Running 5000ms simulation...

  ─── t=1000ms ───
  Speed: 0.087x realtime
  Firing rate: 6.6 Hz
  Cognitive state: alert_engaged
  Neuromod: DA=1.44, 5HT=0.99, NE=1.11, ACh=0.58
  Episodes stored: 12

[CogMind] ✓ Complete
```

---

## The 8 Modules

| Module | What it does | Biological equivalent |
|--------|--------------|----------------------|
| `topology_generator.py` | signature.json → network graph | Connectome |
| `snn_engine.py` | LIF neurons with axonal delays | Cortical columns |
| `plasticity.py` | STDP + R-STDP + homeostasis | Synaptic learning |
| `thalamus.py` | θ/α/γ oscillations + gating | Thalamic nuclei |
| `neuromodulation.py` | DA/5HT/NE/ACh modulation | Brainstem + basal forebrain |
| `hippocampus.py` | Episodic memory + replay | Hippocampal formation |
| `environment.py` | Sensory input + motor output | Body interface |
| `cogmind_runner.py` | Integrates everything | The whole brain |

---

## How Your Signature Maps to the Network

Your brain scan isn't just data — it's the **seed** that makes your emulation unique:

| Your Metric | Example | Network Parameter | Effect |
|-------------|---------|-------------------|--------|
| Regional distribution | F:32% P:25% T:21% O:14% | `module_sizes[]` | More frontal neurons = more executive processing |
| Gray/white ratio | 3.03 | `local_vs_long_connectivity` | Higher = more local processing |
| Gyrification index | 5.33 | `columns_per_module` | More folds = more parallel processing |
| Hemispheric asymmetry | 3.30% left | `hemisphere_ratio` | Left-dominant processing |
| Ventricular volume | 27.48 ml | `pruning_rate` | Network sparsity |
| CSF volume | 160.58 ml | `homeostatic_rate` | Regulation aggressiveness |

---

## The Science

### Why Spiking Neural Networks?

Real neurons communicate with **spikes** (action potentials), not continuous values. This matters because:

1. **Spike timing encodes information** — Two neurons firing 5ms apart means something different than 50ms apart
2. **Oscillations emerge naturally** — Gamma waves aren't programmed, they emerge from network dynamics
3. **Energy efficiency** — Sparse spiking is how brains compute with 20 watts

### Why Neuromodulation?

LLMs have fixed activation functions. Your brain has **four global dials**:

| Neuromodulator | What it controls | Effect |
|----------------|------------------|--------|
| **Dopamine** | Learning rate | High DA = learn faster from recent events |
| **Serotonin** | Firing threshold | High 5HT = more selective, less noise |
| **Noradrenaline** | Signal gain | High NE = sharper discrimination |
| **Acetylcholine** | Internal vs external | High ACh = attend to senses, not memories |

This is how brains have **states** (alert, drowsy, focused, creative). LLMs can't do this.

### Why Thalamo-Cortical Loops?

The thalamus isn't just a relay — it's the **binding mechanism**:

- **Gamma oscillations (30-100 Hz)** create 10-33ms windows where distant neurons can synchronize
- This synchronization is what **integrates information** across brain regions
- Without it, you have parallel processing but no unified experience

This is the physical substrate for consciousness according to IIT (Integrated Information Theory).

---

## Cognitive Signature (Original Module)

Before CogMind, you need a signature. Here's what the analyzer extracts:

### Metrics

| Metric | What it measures | Normal Range |
|--------|------------------|--------------|
| Brain volume | Total tissue volume | ~1350 ml |
| Gray matter | Cortical volume | ~645 ml |
| White matter | Axonal connections | ~445 ml |
| Gray/White ratio | Cortical density | 1.0 - 1.8 |
| Gyrification index | Folding complexity | 2.3 - 3.2 |
| Ventricular volume | Internal fluid spaces | 5 - 70 ml |
| Hemispheric asymmetry | L/R balance | 0.1 - 5% |

### Example Output

```
======================================================================
                    COGNITIVE SIGNATURE REPORT
======================================================================

   SIGNATURE ID: BRAIN-5C13EB2BADEB
   Uniqueness Score: 100/100

======================================================================
                       STRUCTURAL INDICES
======================================================================

   Gray/White Ratio:         3.03    (Population mean: 1.45)
   Gyrification Index:       5.33    (Population mean: 2.55)
   Hemispheric Asymmetry:    3.30%   (Left hemisphere dominant)

======================================================================
                    REGIONAL DISTRIBUTION
======================================================================

   Frontal Region:          32.1%   ████████████████
   Parietal Region:         24.8%   ████████████
   Temporal Region:         21.5%   ██████████
   Occipital Region:        14.2%   ███████
   Central Region:           7.4%   ███

======================================================================
```

---

## Scaling

| Scale | Neurons | Hardware | Speed | Equivalence |
|-------|---------|----------|-------|-------------|
| **Demo** | 1K | Laptop CPU | Real-time | Proof of concept |
| **Prototype** | 100K | 1× GPU | ~10x slower | Minicolumn |
| **Alpha** | 1M | 8× GPU | ~100x slower | Small cortical region |
| **Beta** | 100M | Supercomputer | ~1000x slower | Mouse brain |
| **Full** | 86B | Exascale | Unknown | Human brain |

---

## File Structure

```
cognitive-signature/
├── src/
│   └── brain_analyzer.py      # DICOM → signature.json
├── docs/
│   └── cogmind_architecture.html
├── cogmind_runner.py          # Main integration
├── topology_generator.py      # Signature → network graph
├── snn_engine.py              # LIF spiking neural network
├── plasticity.py              # STDP learning rules
├── thalamus.py                # Oscillations + attention
├── neuromodulation.py         # DA/5HT/NE/ACh system
├── hippocampus.py             # Episodic memory
├── environment.py             # Virtual I/O
├── example_signature.json     # Test signature
├── requirements.txt
└── setup.py
```

---

## Roadmap

- [x] Cognitive Signature extraction from CT/MRI
- [x] Topology generator from signature
- [x] LIF spiking neural network engine
- [x] STDP + R-STDP plasticity
- [x] Thalamic oscillator + attention gating
- [x] Neuromodulation system (DA/5HT/NE/ACh)
- [x] Hippocampal memory system
- [x] Virtual environment interface
- [ ] Brian2/NEST backend integration
- [ ] Real-time visualization dashboard
- [ ] Multi-modal sensory environments
- [ ] Neuromorphic hardware support (Loihi, SpiNNaker)

---

## Contributing

This is open source. Build something brutal with it.

1. Fork the repo
2. Create a feature branch
3. Submit a PR

**Priority areas:**
- Brian2/NEST backends for large-scale simulation
- New environment types (language, vision, motor control)
- Visualization tools
- Benchmarking against biological data

---

## References

- Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks.
- Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons. Journal of Neuroscience.
- Tononi, G. (2004). An information integration theory of consciousness. BMC Neuroscience.
- Zilles, K., et al. (1988). The human pattern of gyrification in the cerebral cortex. Anatomy and Embryology.

---

## License

MIT License — Use it, modify it, build on it.

---

## Author

**Daniel Gamo** ([@gamogestionweb](https://github.com/gamogestionweb))

---

<div align="center">

### Your brain's structure is the seed. The architecture does the rest.

**⭐ Star this repo if you want to see where this goes.**

</div>

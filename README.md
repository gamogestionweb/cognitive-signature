# 🧬 Cognitive Signature Platform

### Mapping the complete biological identity of a human being.

---

## What is this?

Every human is unique. Not just because of their brain — because of their **entire body**. Your blood chemistry, your hormones, your immune system, the physical structure of your brain — all of it together is what makes you *you*.

This project creates a **biological fingerprint** by combining:

- 🧠 **Brain structure** from CT/MRI scans
- 🩸 **Blood chemistry** across multiple time points
- ⚗️ **Hormones** (thyroid, metabolic)
- 🦠 **Immunological history** (what infections you've encountered)
- 🧪 **Toxicology** (what chemicals your body has been exposed to)
- 💧 **Urinalysis**
- 📈 **Temporal evolution** (how your biology changes over time)

The result is a unique **Biological Signature ID** — a hash-based fingerprint derived from your complete biological data.

> **This is not a medical tool.** This doesn't tell you if you're sick. It tells you *who you are*, biologically.

---

## The Philosophy

Traditional neuroscience looks at the brain in isolation. But that's like reading one chapter of a book and claiming you know the whole story.

Your mind doesn't live only in your skull. It lives in:

- The **hormones** that regulate your mood and cognitive speed
- The **immune cells** that communicate with your brain via cytokines
- The **gut bacteria** that produce serotonin and GABA
- The **blood** that carries oxygen and nutrients to every neuron
- The **peripheral nerves** that define how you physically feel the world

**If you want to truly map a human identity, you need to map the whole body.**

This project is step one.

---

## Case Study #1: The Creator

The first subject is the creator of this project — **Daniel Aparicio Gamo** (born 1992, Madrid, Spain). Real medical data. Real brain scan. Real person.

```
╔══════════════════════════════════════════════════════════════╗
║  SIGNATURE ID: BIO-387E79F9997E                             ║
║  BRAIN ID:     BRAIN-5C43EB2BADwB                           ║
║  Subject:      GAMO, DANIEL                        ║
║  Age:          34  |  Sex: M  |  Location: Spain    ║
║  Data:         Aug 2023 → Oct 2025 (2+ years tracked)       ║
║  Completeness: 50% (8 of 16 biological systems mapped)      ║
╚══════════════════════════════════════════════════════════════╝
```

### What makes this individual biologically distinctive:

| Marker | Value | Population Norm | Z-Score | What it means |
|--------|-------|----------------|---------|---------------|
| Eosinophils | 0.7 x10³/µl | 0.0 - 0.5 | +3.60 | Unique immune/allergic phenotype |
| Eosinophils % | 9.2% | 0.4 - 6.5% | +3.77 | Top 1% of population |
| Neutrophils % | 39.6% | 41.5 - 72% | -2.25 | Below range — lymphocyte-dominant |
| LDL Cholesterol | 121 mg/dl | < 100 | +2.84 | Slightly elevated |
| Total Cholesterol | 205 mg/dl | < 200 | +2.10 | Borderline |

### Immune Profile: **Lymphocyte-Dominant**

Most adults have more neutrophils than lymphocytes. This subject has the opposite (lymphocytes 42.1% vs neutrophils 39.6%). Only about 15-20% of healthy adult males show this pattern. It suggests a strong adaptive immune system.

### How the body changed over 2 years (2023 → 2025):

| Metric | Change | Direction | Interpretation |
|--------|--------|-----------|----------------|
| Eosinophils | +75.0% | ↑ | Immune system shift |
| Triglycerides | -37.9% | ↓ | Significant metabolic improvement |
| HDL Cholesterol | +29.1% | ↑ | Better cardiovascular protection |
| Creatinine | -17.2% | ↓ | Improved kidney function |
| Hemoglobin | +7.9% | ↑ | Better oxygen carrying capacity |

---

## How It Works

### 1. Collect Data

You need medical lab results. In most countries, you can request your blood test results from your doctor. Brain scans (CT or MRI) require a medical referral.

### 2. Run the Analysis

```bash
git clone https://github.com/gamogestionweb/cognitive-signature.git
cd cognitive-signature
pip install -r requirements.txt

# Run the platform with the built-in case study
python src/cognitive_signature_platform.py ./output

# Or import and build your own
python -c "
from src.cognitive_signature_platform import *
subject = Subject('YOUR NAME', '1990-01-01', 'M')
platform = CognitiveSignaturePlatform(subject)
# Add your data...
report = platform.generate_report()
print(report['signature_id'])
"
```

### 3. Get Your Signature

The platform outputs:
- A **unique Signature ID** (SHA-256 hash of all your biological data)
- **Z-scores** for every metric (how you compare to population norms)
- **Computed indices** (inflammation, cardiovascular risk, liver health, immune profile)
- **Temporal evolution** (how your biology changes between measurements)
- **Distinctive markers** (what makes you biologically unique)
- **Interactive dashboard** (HTML file you can open in any browser)

---

## Output Files

| File | What it is |
|------|------------|
| `output/signature_report.json` | Complete raw data — every metric, Z-score, index, and analysis |
| `output/cognitive_signature_platform.html` | Interactive visual dashboard — open in any browser |
| `src/cognitive_signature_platform.py` | The platform source code |
| `src/brain_analyzer.py` | Original brain CT/MRI analyzer (v1) |

---

## Computed Biological Indices

The platform doesn't just store numbers — it computes derived indices that reveal systemic patterns:

| Index | Formula | What it reveals |
|-------|---------|-----------------|
| **NLR** | Neutrophils ÷ Lymphocytes | Systemic inflammation (lower = less inflammation) |
| **PLR** | Platelets ÷ Lymphocytes | Thromboinflammatory state |
| **Atherogenic Index** | (Total Chol - HDL) ÷ HDL | Cardiovascular risk |
| **De Ritis Ratio** | AST ÷ ALT | Liver integrity |
| **TSH Index** | TSH × Free T4 | Thyroid hormone sensitivity |
| **Immune Profile** | Lymphocyte% vs Neutrophil% | Adaptive vs innate immunity dominance |

---

## What We've Captured (and What's Still Missing)

### ✅ Mapped Systems (50%)

- Brain structural analysis (CT)
- Complete blood chemistry (2 time points)
- Hematology (red cells, white cells, platelets)
- Lipid profile
- Hepatic function
- Renal function
- Electrolyte balance
- Thyroid hormones
- Vitamins (B12, folic acid)
- Iron metabolism
- Glycated hemoglobin
- Urinalysis
- 10-panel toxicology
- Serology (Hep B/C, HIV, syphilis, EBV)

### ❌ Missing Systems (50%)

| System | What's needed | Why it matters |
|--------|---------------|----------------|
| **Connectome** | DTI + resting-state fMRI | Maps the brain's wiring — how regions communicate |
| **Peripheral Nervous System** | Nerve conduction studies | How your body sends signals to the brain |
| **Sensory Receptor Map** | Somatosensory testing | How you physically experience touch, pain, temperature |
| **Microbiome** | 16S rRNA gut sequencing | Your gut bacteria produce neurotransmitters |
| **Full Endocrine** | Cortisol, testosterone, insulin, IGF-1 | Hormones drive behavior and cognition |
| **Inflammatory Cytokines** | CRP, IL-6, TNF-alpha | Chronic inflammation affects brain function |
| **Genome** | Whole genome sequencing | Your biological source code |
| **Epigenetics** | DNA methylation array | How life experience modified your gene expression |

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Done | Brain structural signature from CT scan |
| 2 | ✅ Done | Blood chemistry + hematology + lipids + hepatic + renal |
| 2b | ✅ Done | Hormones + vitamins + serology + toxicology + urinalysis |
| 3 | 🔜 Next | Full endocrine panel (cortisol, testosterone, insulin) |
| 4 | 📋 Planned | Brain connectome (DTI + fMRI) |
| 5 | 📋 Planned | Microbiome + inflammatory cytokines |
| 6 | 📋 Planned | Genome + epigenetics |
| 7 | 📋 Planned | Peripheral nervous system + sensory mapping |
| 8 | 🌅 Horizon | Complete integrated biological model |

---

## Want to Add Your Own Data?

1. Fork this repository
2. Copy the data structure from `src/cognitive_signature_platform.py`
3. Replace the values with your own lab results
4. Run the analysis
5. Submit a pull request if you want to contribute to the project

**Privacy note:** This project deals with medical data. Only share what you're comfortable making public. The case study subject chose to share his data openly for research purposes.

---

## Understanding the Metrics

### Z-Score
A Z-score tells you how far a value is from the population average:
- **-1 to +1**: Normal range (68% of people fall here)
- **-2 to +2**: Slight deviation (95% of people)
- **Beyond ±2**: Significantly different from most people

### Percentile
If your percentile is 85%, it means 85% of the population has a lower value than you.

---

## Technical Stack

- **Python 3.8+** (no external dependencies for core analysis)
- **HTML/CSS/JS** (interactive dashboard — no build tools needed)
- Standard library only: `json`, `hashlib`, `math`, `dataclasses`
- Optional: `pydicom`, `nibabel`, `numpy` for brain scan processing

---

## Contributing

This is an open research project. Contributions welcome in any form:

- Add new biological data layers
- Improve the analysis algorithms
- Create better visualizations
- Add your own data as a case study
- Translate documentation
- Write about the philosophy

---

## References

- Allen, J.S., et al. (2002). *Normal neuroanatomical variation in the human brain*
- Sporns, O. (2011). *The human connectome: a complex network*
- Cryan, J.F. & Dinan, T.G. (2012). *Mind-altering microorganisms: the impact of the gut microbiota on brain and behaviour*
- Slavich, G.M. & Irwin, M.R. (2014). *From stress to inflammation and major depressive disorder*
- Sled, J.G., et al. (2010). *Regional variations in gray matter morphometry*

---

## License

MIT License — See [LICENSE](LICENSE) for details.

## Author

**Daniel Aparicio Gamo** ([@gamogestionweb](https://github.com/gamogestionweb))

---

<p align="center">
  <em>"Every body is a universe. This tool helps you map yours."</em>
</p>

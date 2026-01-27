# 🧠 Cognitive Signature

**Extract unique cognitive fingerprints from brain CT/MRI scans.**

This tool analyzes brain medical images (DICOM format) and generates a unique "cognitive signature" - a fingerprint based on structural brain features that can potentially identify individual differences in brain organization.

![Brain 3D Visualization](docs/brain_3d_example.png)

## What is a Cognitive Signature?

A cognitive signature is a unique identifier derived from your brain's structural features:

- **Brain volume** and tissue composition
- **Gray/White matter ratio** - indicator of cortical density
- **Gyrification index** - complexity of cortical folding
- **Hemispheric asymmetry** - balance between brain hemispheres
- **Ventricular system** - internal fluid spaces
- **Regional patterns** - distribution across brain lobes

Each brain has a unique combination of these features, like a fingerprint.

## Features

✅ **DICOM Processing** - Load CT or MRI brain scans
✅ **Automatic Segmentation** - Skull, gray matter, white matter, CSF, ventricles
✅ **3D Visualization** - Interactive brain model you can rotate and explore
✅ **Normative Comparison** - Compare your metrics with population averages
✅ **Unique Signature ID** - Generate a hash-based brain fingerprint
✅ **Scientific Reports** - Export data in JSON format for further analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/gamogestionweb/cognitive-signature.git
cd cognitive-signature

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.brain_analyzer import CognitiveSignatureAnalyzer

# Point to your DICOM folder
analyzer = CognitiveSignatureAnalyzer("path/to/your/dicom/folder")

# Run analysis
analyzer.run_analysis()

# Generate reports
analyzer.generate_report("./output")

# Print summary
analyzer.print_summary()
```

Or from command line:

```bash
python src/brain_analyzer.py ./my_brain_scan ./results
```

## Output

The tool generates:

| File | Description |
|------|-------------|
| `brain_3d.html` | Interactive 3D visualization |
| `comparison.html` | Dashboard comparing your brain with population norms |
| `signature.json` | Raw data with all metrics and your unique signature ID |

## Example Analysis

Here's an example output from a real brain CT scan:

```
======================================================================
   ANALYSIS SUMMARY
======================================================================

   Signature ID: BRAIN-5C13EB2BADEB

   SUBJECT vs POPULATION:
   ----------------------------------------------------------------
   Metric                    Subject     Normal    Z-Score       Status
   ----------------------------------------------------------------
   Brain Volume              1420.00    1380.00      +0.29       Normal
   Gray Matter                680.50     645.00      +0.55       Normal
   White Matter               456.20     445.00      +0.22       Normal
   Ventricles                  22.30      25.00      -0.18       Normal
   Gray/White Ratio             1.49       1.45      +0.20       Normal
   Gyrification                 2.67       2.55      +0.40       Normal
   ----------------------------------------------------------------
```

## Understanding the Metrics

### Z-Score
- **-1 to +1**: Normal range (68% of population)
- **-2 to +2**: Slight deviation (95% of population)
- **Beyond ±2**: Significant deviation

### Key Metrics Explained

| Metric | What it measures | Clinical relevance |
|--------|------------------|-------------------|
| **Gray/White Ratio** | Proportion of cortex to white matter | Higher values may indicate preserved cognitive function |
| **Gyrification Index** | Cortical folding complexity | Higher = more cortical surface area |
| **Ventricular Volume** | Size of fluid-filled spaces | Enlargement may indicate atrophy |
| **Hemispheric Asymmetry** | Balance between hemispheres | Normal brains have slight asymmetry |

## Normative Data Sources

The comparison values are based on peer-reviewed neuroimaging studies:

- Allen, J. S., et al. (2002). *Normal neuroanatomical variation in the human brain*
- Sled, J. G., et al. (2010). *Regional variations in gray matter morphometry*
- Courchesne, E., et al. (2000). *Normal brain development and aging*

## Limitations

⚠️ **This tool is for research and educational purposes only.**

- CT scans have lower tissue contrast than MRI
- Volume measurements include some non-brain tissue
- **Ratios and proportions are more reliable than absolute volumes**
- Cannot detect pathologies - consult a medical professional for diagnosis

## Research Applications

Potential uses for cognitive signatures:

- **Longitudinal studies** - Track brain changes over time
- **Population studies** - Compare groups
- **Biometric research** - Brain-based identification
- **AI training** - Structural brain feature extraction
- **Cognitive modeling** - Correlate structure with function

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See [LICENSE](LICENSE) for details.

## Author

**Daniel Gamo** ([@gamogestionweb](https://github.com/gamogestionweb))

---

*"Every brain is unique. This tool helps you discover yours."*

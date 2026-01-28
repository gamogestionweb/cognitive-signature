# 🧠 Cognitive Signature

**Extract unique cognitive fingerprints from brain CT/MRI scans.**

This tool analyzes brain medical images (DICOM format) and generates a unique "cognitive signature" - a fingerprint based on structural brain features that can potentially identify individual differences in brain organization.

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

Here's an example output from a real brain CT scan (Subject #1):

```
======================================================================
                        COGNITIVE SIGNATURE REPORT
======================================================================

   SIGNATURE ID: BRAIN-5C13EB2BADEB
   Uniqueness Score: 100/100

======================================================================
                          VOLUMETRIC DATA
======================================================================

   Total Brain Volume:     5294.31 ml (includes surrounding tissue)
   Gray Matter:            1717.47 ml
   White Matter:            566.54 ml
   Cerebrospinal Fluid:     160.58 ml
   Ventricular System:       27.48 ml

======================================================================
                      STRUCTURAL INDICES
======================================================================

   Gray/White Ratio:         3.03    (Population mean: 1.45)
   Gyrification Index:       5.33    (Population mean: 2.55)
   Hemispheric Asymmetry:    3.30%   (Left hemisphere dominant)

======================================================================
                   COMPARISON WITH POPULATION NORMS
======================================================================

   Metric                    Subject     Normal    Z-Score    Percentile
   ----------------------------------------------------------------------
   Brain Volume (ml)        5294.31    1350.00     +28.17        >99%
   Gray Matter (ml)         1717.47     645.00     +16.56        >99%
   White Matter (ml)         566.54     445.00      +2.43         99%
   Ventricles (ml)            27.48      25.00      +0.17         57%
   CSF (ml)                  160.58     140.00      +0.69         75%
   Gray/White Ratio           3.03       1.45     +10.53        >99%
   Gyrification               5.33       2.55      +9.27        >99%
   Asymmetry (%)              3.30       2.00      +0.43         67%
   ----------------------------------------------------------------------

   Note: High absolute volumes are due to CT scan including soft tissue.
   Ratios and indices are more reliable for inter-subject comparison.

======================================================================
                       REGIONAL DISTRIBUTION
======================================================================

   Frontal Region:          32.1%
   Parietal Region:         24.8%
   Temporal Region:         21.5%
   Occipital Region:        14.2%
   Central Region:           7.4%

======================================================================
```

## Understanding the Metrics

### Z-Score
- **-1 to +1**: Normal range (68% of population)
- **-2 to +2**: Slight deviation (95% of population)
- **Beyond ±2**: Significant deviation

### Key Metrics Explained

| Metric | What it measures | Normal Range | Method |
|--------|------------------|--------------|--------|
| **Gray/White Ratio** | Proportion of cortex to white matter | 1.0 - 1.8 | HU thresholding (GM: 37-45, WM: 20-32) |
| **Gyrification Index** | Cortical folding complexity | 2.3 - 3.2 | Pial/Hull surface ratio (Zilles method) |
| **Ventricular Volume** | Size of fluid-filled spaces | 5 - 70 ml | Central CSF segmentation |
| **Hemispheric Asymmetry** | Balance between hemispheres | 0.1 - 5% | L-R volume difference |

### Hounsfield Unit (HU) Ranges for CT Segmentation

| Tissue | HU Range | Reference |
|--------|----------|-----------|
| Gray Matter | 37 - 45 HU | Mean ~40 HU |
| White Matter | 20 - 32 HU | Mean ~25-30 HU |
| CSF | 0 - 15 HU | Near water density |
| Bone/Skull | > 300 HU | Cortical bone |

## Normative Data Sources

The comparison values are based on peer-reviewed neuroimaging studies:

- Allen, J. S., et al. (2002). *Normal neuroanatomical variation in the human brain*
- Sled, J. G., et al. (2010). *Regional variations in gray matter morphometry*
- [Lifespan Gyrification Trajectories](https://www.nature.com/articles/s41598-017-00582-1) - Nature Scientific Reports
- [Inter-scanner HU variability](https://pubmed.ncbi.nlm.nih.gov/30017694/) - AJEM 2018
- [CT-determined intracranial volume](https://pubmed.ncbi.nlm.nih.gov/11314299/) - Normal population values

## Methodology

### Gyrification Index Calculation

This tool uses the **Zilles method** (pial/hull ratio) for gyrification calculation:

```
GI = Pial Surface Area / Outer Hull Surface Area
```

- **Pial surface**: The actual brain surface including all gyri and sulci
- **Outer hull**: A smooth convex envelope that wraps the brain
- **Normal human range**: 2.3 - 3.2 (decreases with age: GI ≈ 3.4 - 0.17×ln(age))

Reference: [How to Measure Cortical Folding from MR Images](https://pmc.ncbi.nlm.nih.gov/articles/PMC3369773/)

### CT vs MRI Considerations

| Aspect | CT | MRI |
|--------|-----|-----|
| Tissue contrast | Lower | Higher |
| Absolute volumes | Include extra-cranial tissue | More accurate segmentation |
| Recommended metrics | **Ratios and indices** | Both volumes and ratios |
| Speed | Fast | Slower |

## Limitations

⚠️ **This tool is for research and educational purposes only.**

- CT scans have lower tissue contrast than MRI
- Volume measurements include some non-brain tissue
- **Ratios and proportions are more reliable than absolute volumes**
- Gyrification calculated from CT is an approximation (MRI preferred for precise measurement)
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

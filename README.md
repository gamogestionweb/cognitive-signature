# Cognitive Signature

**Extract unique cognitive fingerprints from brain CT/MRI scans.**

This tool analyzes brain medical images (DICOM format) and generates a unique "cognitive signature" - a fingerprint based on structural brain features that can potentially identify individual differences in brain organization.

## What is a Cognitive Signature?

A cognitive signature is a unique identifier derived from your brain's structural features:

- **Brain volume** and tissue composition
- **Gray/White matter ratio** - indicator of cortical density
- **Gyrification index** - complexity of cortical folding (Zilles method)
- **Hemispheric asymmetry** - balance between brain hemispheres
- **Ventricular system** - internal fluid spaces
- **Regional patterns** - distribution across brain lobes

Each brain has a unique combination of these features, like a fingerprint.

## Features

- **DICOM Processing** - Load CT or MRI brain scans
- **Automatic Segmentation** - Skull, gray matter, white matter, CSF, ventricles
- **3D Visualization** - Interactive brain model you can rotate and explore
- **Normative Comparison** - Compare your metrics with population averages
- **Unique Signature ID** - Generate a hash-based brain fingerprint
- **Scientific Reports** - Export data in JSON format for further analysis

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

Example output from a brain CT scan analysis:

```
======================================================================
                        COGNITIVE SIGNATURE REPORT
======================================================================

   SIGNATURE ID: BRAIN-5C13EB2BADEB
   Source: CT Scan (Hospital 12 de Octubre, 2021)

======================================================================
                      RELIABLE METRICS (CT)
======================================================================

   Metric                    Subject     Normal     Z-Score    Status
   ----------------------------------------------------------------------
   Ventricles (ml)            27.48      25.00      +0.17      Normal
   CSF (ml)                  160.58     165.00      -0.08      Normal
   Hemispheric Asymmetry      3.30%      2.00%     +0.87      Normal
   Hemisphere Dominance:      Left

======================================================================
                   CT-SPECIFIC CONSIDERATIONS
======================================================================

   The following metrics require MRI for accurate measurement:

   - Gyrification Index: CT approximation only (MRI needed for Zilles method)
   - Gray/White Ratio: HU overlap limits precision
   - Absolute volumes: CT includes extra-cranial tissue

======================================================================
                       REGIONAL DISTRIBUTION
======================================================================

   Frontal Region:          32.1%   [Higher than average]
   Parietal Region:         24.8%
   Temporal Region:         21.5%
   Occipital Region:        14.2%
   Central Region:           7.4%

======================================================================
                    SUBJECT CHARACTERISTICS
======================================================================

   - Left hemisphere dominant (3.30% asymmetry)
   - Ventricular system: Normal size (percentile 57)
   - Frontal lobe proportion: Above average
   - No signs of atrophy or hydrocephalus

======================================================================
```

## Understanding the Metrics

### Z-Score Interpretation
- **-1 to +1**: Normal range (68% of population)
- **-2 to +2**: Slight deviation (95% of population)
- **Beyond +/-2**: Significant deviation

### Key Metrics

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

## Methodology

### Gyrification Index Calculation

This tool uses the **Zilles method** (pial/hull ratio) for gyrification calculation:

```
GI = Pial Surface Area / Outer Hull Surface Area
```

- **Pial surface**: The actual brain surface including all gyri and sulci
- **Outer hull**: A smooth convex envelope that wraps the brain
- **Normal human range**: 2.3 - 3.2 (decreases with age: GI = 3.4 - 0.17 x ln(age))

References:
- [How to Measure Cortical Folding from MR Images](https://pmc.ncbi.nlm.nih.gov/articles/PMC3369773/) - PMC Tutorial
- [Lifespan Gyrification Trajectories](https://www.nature.com/articles/s41598-017-00582-1) - Nature Scientific Reports

### CT vs MRI Considerations

| Aspect | CT | MRI |
|--------|-----|-----|
| Tissue contrast | Lower | Higher |
| Absolute volumes | Include extra-cranial tissue | More accurate segmentation |
| Gyrification | Approximation only | Gold standard (Zilles method) |
| Recommended metrics | **Ratios and indices** | Both volumes and ratios |
| Speed | Fast | Slower |
| Cost | Lower | Higher |

## Normative Data Sources

The comparison values are based on peer-reviewed neuroimaging studies:

- Allen, J. S., et al. (2002). *Normal neuroanatomical variation in the human brain*
- Sled, J. G., et al. (2010). *Regional variations in gray matter morphometry*
- [Lifespan Gyrification Trajectories](https://www.nature.com/articles/s41598-017-00582-1) - Nature Scientific Reports (PMC5428697)
- [Inter-scanner HU variability](https://pubmed.ncbi.nlm.nih.gov/30017694/) - AJEM 2018
- [CT-determined intracranial volume](https://pubmed.ncbi.nlm.nih.gov/11314299/) - Normal population values
- [Gray-to-white matter ratio](https://pubmed.ncbi.nlm.nih.gov/18843066/) - CT density studies

## Limitations

**This tool is for research and educational purposes only.**

- CT scans have lower tissue contrast than MRI
- Volume measurements from CT include some non-brain tissue
- **Ratios and proportions are more reliable than absolute volumes**
- Gyrification calculated from CT is an approximation (MRI preferred)
- Cannot detect pathologies - consult a medical professional for diagnosis
- Inter-scanner variability affects HU measurements

## Research Applications

Potential uses for cognitive signatures:

- **Longitudinal studies** - Track brain changes over time
- **Population studies** - Compare groups
- **Biometric research** - Brain-based identification
- **AI training** - Structural brain feature extraction
- **Cognitive modeling** - Correlate structure with function

## Future Development

- [ ] MRI T1 3D support with FreeSurfer integration
- [ ] Blood biomarker correlation module
- [ ] Age-adjusted normative comparisons
- [ ] Multi-subject batch processing

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

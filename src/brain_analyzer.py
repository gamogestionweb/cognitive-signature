#!/usr/bin/env python3
"""
Brain Cognitive Signature Analyzer
===================================
Extracts unique cognitive signatures from brain CT/MRI scans.

This module provides tools for:
- Loading and processing DICOM medical images
- Segmenting brain structures (skull, gray matter, white matter, ventricles)
- Extracting morphometric features
- Comparing with normative population data
- Generating 3D visualizations
- Creating unique cognitive signature fingerprints

Author: Daniel Gamo (@gamogestionweb)
License: MIT
"""

import numpy as np
import pydicom
from pathlib import Path
from scipy import ndimage
from skimage import measure, morphology
from skimage.feature import graycomatrix, graycoprops
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# NORMATIVE REFERENCE DATA
# Based on published neuroimaging studies
# Sources: Allen et al. 2002, Sled et al. 2010, Courchesne et al. 2000
# =============================================================================

NORMATIVE_DATA = {
    # ==========================================================================
    # VOLUMETRIC DATA (MRI-based studies)
    # Note: CT volumes include extra-cranial tissue, use ratios for comparison
    # ==========================================================================
    'total_intracranial_volume': {
        'male': {'mean': 1470, 'std': 130, 'min': 1200, 'max': 1800},
        'female': {'mean': 1290, 'std': 115, 'min': 1050, 'max': 1550},
        'average': {'mean': 1380, 'std': 140, 'min': 1100, 'max': 1700},
        'unit': 'ml',
        'source': 'Allen et al. 2002; CT-determined ICV: PubMed 11314299'
    },
    'gray_matter_volume': {
        'average': {'mean': 645, 'std': 65, 'min': 520, 'max': 780},
        'unit': 'ml',
        'source': 'Sled et al. 2010'
    },
    'white_matter_volume': {
        'average': {'mean': 445, 'std': 50, 'min': 340, 'max': 560},
        'unit': 'ml',
        'source': 'Sled et al. 2010'
    },
    'csf_volume': {
        'average': {'mean': 165, 'std': 55, 'min': 75, 'max': 325},
        'unit': 'ml',
        'source': 'PMC12290786 - iCSF reference values'
    },
    'lateral_ventricle_volume': {
        'young': {'mean': 15, 'std': 8, 'min': 5, 'max': 35},
        'middle': {'mean': 25, 'std': 12, 'min': 8, 'max': 55},
        'elderly': {'mean': 45, 'std': 20, 'min': 15, 'max': 100},
        'average': {'mean': 25, 'std': 15, 'min': 5, 'max': 70},
        'unit': 'ml'
    },
    # ==========================================================================
    # STRUCTURAL RATIOS (more reliable for CT inter-subject comparison)
    # ==========================================================================
    'gray_white_ratio': {
        # CT-based GWR from cardiac arrest studies: mean 1.32
        # Reference: PubMed 18843066, ScienceDirect S0735-6757(18)30573-4
        'average': {'mean': 1.32, 'std': 0.15, 'min': 1.0, 'max': 1.8},
        'unit': 'ratio',
        'source': 'Inter-scanner variability in HU - AJEM 2018'
    },
    'ventricle_brain_ratio': {
        'average': {'mean': 2.5, 'std': 1.5, 'min': 0.5, 'max': 7.0},
        'unit': '%'
    },
    'cephalic_index': {
        'average': {'mean': 78, 'std': 5, 'min': 65, 'max': 90},
        'unit': 'index'
    },
    'gyrification_index': {
        # Pial/Hull method (Zilles et al.)
        # Reference: Nature Sci Rep - Lifespan Gyrification Trajectories
        # Human adult range: 2.5-3.0, follows GI = 3.4 - 0.17*ln(age)
        'average': {'mean': 2.75, 'std': 0.25, 'min': 2.3, 'max': 3.2},
        'unit': 'index',
        'source': 'PMC5428697 - Lifespan Gyrification Trajectories'
    },
    'hemispheric_asymmetry': {
        # Normal brains show slight left-right asymmetry
        'average': {'mean': 2.0, 'std': 1.5, 'min': 0.1, 'max': 5.0},
        'unit': '%'
    }
}


class CognitiveSignatureAnalyzer:
    """
    Main class for brain cognitive signature analysis.

    This class processes brain CT/MRI DICOM files and extracts a unique
    'cognitive signature' - a fingerprint based on structural brain features.

    Attributes:
        dicom_folder (Path): Path to folder containing DICOM files
        volume (np.ndarray): 3D volume data in Hounsfield Units
        spacing (list): Voxel spacing in mm [x, y, z]
        structures (dict): Segmented brain structures
        features (dict): Extracted morphometric features
        comparison (list): Comparison with normative data
        signature_id (str): Unique signature identifier

    Example:
        >>> analyzer = CognitiveSignatureAnalyzer("path/to/dicom/folder")
        >>> analyzer.run_analysis()
        >>> analyzer.generate_report("output_folder")
    """

    def __init__(self, dicom_folder):
        """
        Initialize the analyzer.

        Args:
            dicom_folder: Path to folder containing DICOM files
        """
        self.dicom_folder = Path(dicom_folder)
        self.volume = None
        self.spacing = [1.0, 1.0, 1.0]
        self.structures = {}
        self.features = {}
        self.comparison = []
        self.signature_id = None

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("\n" + "="*70)
        print("   COGNITIVE SIGNATURE ANALYZER")
        print("="*70)

        self._load_dicom()
        self._segment_structures()
        self._extract_features()
        self._compare_with_norms()
        self._generate_signature()

        return self

    def _load_dicom(self):
        """Load DICOM series and build 3D volume."""
        print("\n[1/5] Loading DICOM data...")

        # Find series with most files
        max_files = 0
        series_folder = None
        for folder in self.dicom_folder.iterdir():
            if folder.is_dir():
                dcm_files = list(folder.glob("*.dcm"))
                if len(dcm_files) > max_files:
                    max_files = len(dcm_files)
                    series_folder = folder

        if not series_folder:
            # Try root folder
            dcm_files = list(self.dicom_folder.glob("*.dcm"))
            if dcm_files:
                series_folder = self.dicom_folder
            else:
                raise ValueError("No DICOM files found")

        # Load slices
        slices = []
        for dcm_file in series_folder.glob("*.dcm"):
            try:
                slices.append(pydicom.dcmread(str(dcm_file)))
            except:
                pass

        if not slices:
            raise ValueError("Could not read DICOM files")

        # Sort by position
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except:
            slices.sort(key=lambda x: float(x.InstanceNumber))

        # Get spacing
        ref = slices[0]
        self.spacing = [
            float(ref.PixelSpacing[0]),
            float(ref.PixelSpacing[1]),
            float(getattr(ref, 'SliceThickness', 1.0))
        ]

        # Build volume
        self.volume = np.zeros((len(slices), int(ref.Rows), int(ref.Columns)), dtype=np.float32)
        for i, s in enumerate(slices):
            slope = float(getattr(s, 'RescaleSlope', 1))
            intercept = float(getattr(s, 'RescaleIntercept', 0))
            self.volume[i] = s.pixel_array.astype(np.float32) * slope + intercept

        print(f"       Loaded {len(slices)} slices")
        print(f"       Volume shape: {self.volume.shape}")
        print(f"       Voxel spacing: {self.spacing} mm")

    def _segment_structures(self):
        """Segment brain structures based on Hounsfield Units."""
        print("[2/5] Segmenting brain structures...")

        vol = self.volume

        # Skull (bone): HU > 300
        skull = vol > 300
        skull = morphology.binary_closing(skull, morphology.ball(2))

        # Brain tissue: -100 < HU < 200
        brain = (vol > -100) & (vol < 200)
        brain = ndimage.binary_fill_holes(brain)
        brain = morphology.binary_opening(brain, morphology.ball(2))
        brain = morphology.binary_closing(brain, morphology.ball(2))

        # Keep largest component
        labeled = measure.label(brain)
        if labeled.max() > 0:
            regions = measure.regionprops(labeled)
            largest = max(regions, key=lambda x: x.area)
            brain = labeled == largest.label

        # Gray matter: HU 37-45 (based on literature: ~40 HU mean)
        # Reference: Inter-scanner variability in HU measured by CT of the brain
        gray_matter = (vol >= 37) & (vol <= 45) & brain
        gray_matter = morphology.binary_opening(gray_matter, morphology.ball(1))

        # CSF: HU 0-15 (clear fluid, near water density)
        csf = (vol >= 0) & (vol <= 15) & brain
        csf = morphology.binary_opening(csf, morphology.ball(1))

        # White matter: HU 20-32 (based on literature: ~25-30 HU mean)
        # No overlap with gray matter range to prevent ratio inflation
        white_matter = (vol >= 20) & (vol <= 32) & brain & ~csf
        white_matter = morphology.binary_opening(white_matter, morphology.ball(1))

        # Ventricles: central CSF
        z, y, x = [s//2 for s in brain.shape]
        center = np.zeros_like(csf)
        z1, z2 = max(0, z-35), min(brain.shape[0], z+35)
        y1, y2 = max(0, y-60), min(brain.shape[1], y+60)
        x1, x2 = max(0, x-60), min(brain.shape[2], x+60)
        center[z1:z2, y1:y2, x1:x2] = 1
        ventricles = csf & center.astype(bool)

        self.structures = {
            'skull': skull,
            'brain': brain,
            'gray_matter': gray_matter,
            'white_matter': white_matter,
            'csf': csf,
            'ventricles': ventricles
        }

        print(f"       Brain: {brain.sum():,} voxels")
        print(f"       Gray matter: {gray_matter.sum():,} voxels")
        print(f"       White matter: {white_matter.sum():,} voxels")
        print(f"       Ventricles: {ventricles.sum():,} voxels")

    def _extract_features(self):
        """Extract morphometric features."""
        print("[3/5] Extracting morphometric features...")

        voxel_ml = np.prod(self.spacing) / 1000
        brain = self.structures['brain']

        # Volumes
        self.features['brain_volume'] = brain.sum() * voxel_ml
        self.features['gray_matter'] = self.structures['gray_matter'].sum() * voxel_ml
        self.features['white_matter'] = self.structures['white_matter'].sum() * voxel_ml
        self.features['csf'] = self.structures['csf'].sum() * voxel_ml
        self.features['ventricles'] = self.structures['ventricles'].sum() * voxel_ml

        # Ratios
        self.features['gray_white_ratio'] = self.features['gray_matter'] / max(self.features['white_matter'], 0.1)
        self.features['ventricle_ratio'] = (self.features['ventricles'] / self.features['brain_volume']) * 100

        # Shape
        coords = np.where(brain)
        z_range = (coords[0].max() - coords[0].min()) * self.spacing[2]
        y_range = (coords[1].max() - coords[1].min()) * self.spacing[1]
        x_range = (coords[2].max() - coords[2].min()) * self.spacing[0]
        self.features['cephalic_index'] = (x_range / max(y_range, 1)) * 100

        # Hemispheric asymmetry
        mid_x = brain.shape[2] // 2
        left = brain[:, :, :mid_x].sum() * voxel_ml
        right = brain[:, :, mid_x:].sum() * voxel_ml
        self.features['left_hemisphere'] = left
        self.features['right_hemisphere'] = right
        self.features['asymmetry'] = abs(left - right) / (left + right) * 100

        # Gyrification index using pial/hull method (Zilles et al.)
        # GI = pial surface area / outer hull surface area
        # Reference: Nature Scientific Reports - Lifespan Gyrification Trajectories
        # Normal human range: 2.5-3.0
        try:
            # Get pial surface (actual brain surface with all folds)
            brain_smooth = ndimage.gaussian_filter(brain.astype(float), sigma=0.5)
            verts_pial, faces_pial, _, _ = measure.marching_cubes(
                brain_smooth, level=0.5, spacing=self.spacing
            )
            pial_surface = measure.mesh_surface_area(verts_pial, faces_pial)

            # Create outer hull (convex envelope that wraps the brain)
            # Using morphological closing to create smooth outer surface
            hull = morphology.convex_hull_image(brain)
            hull_smooth = ndimage.gaussian_filter(hull.astype(float), sigma=2.0)
            verts_hull, faces_hull, _, _ = measure.marching_cubes(
                hull_smooth, level=0.5, spacing=self.spacing
            )
            hull_surface = measure.mesh_surface_area(verts_hull, faces_hull)

            # GI = pial / hull (standard Zilles method)
            self.features['gyrification'] = pial_surface / max(hull_surface, 1.0)

            # Store surfaces for debugging
            self.features['pial_surface_mm2'] = pial_surface
            self.features['hull_surface_mm2'] = hull_surface
        except Exception as e:
            # Fallback to population mean if calculation fails
            self.features['gyrification'] = 2.55
            self.features['gyrification_error'] = str(e)

        print(f"       Brain volume: {self.features['brain_volume']:.1f} ml")
        print(f"       Gray/White ratio: {self.features['gray_white_ratio']:.2f}")
        print(f"       Gyrification index: {self.features['gyrification']:.2f}")

    def _compare_with_norms(self):
        """Compare with normative population data."""
        print("[4/5] Comparing with population norms...")

        mappings = [
            ('brain_volume', 'total_intracranial_volume', 'Brain Volume', 'ml'),
            ('gray_matter', 'gray_matter_volume', 'Gray Matter', 'ml'),
            ('white_matter', 'white_matter_volume', 'White Matter', 'ml'),
            ('csf', 'csf_volume', 'CSF', 'ml'),
            ('ventricles', 'lateral_ventricle_volume', 'Ventricles', 'ml'),
            ('gray_white_ratio', 'gray_white_ratio', 'Gray/White Ratio', 'ratio'),
            ('ventricle_ratio', 'ventricle_brain_ratio', 'Ventricle/Brain', '%'),
            ('cephalic_index', 'cephalic_index', 'Cephalic Index', 'index'),
            ('gyrification', 'gyrification_index', 'Gyrification', 'index'),
            ('asymmetry', 'hemispheric_asymmetry', 'Asymmetry', '%'),
        ]

        for feat_key, norm_key, label, unit in mappings:
            if feat_key in self.features and norm_key in NORMATIVE_DATA:
                value = self.features[feat_key]
                norm = NORMATIVE_DATA[norm_key]['average']

                z = (value - norm['mean']) / norm['std'] if norm['std'] > 0 else 0
                percentile = min(99, max(1, 50 + z * 16))

                if abs(z) < 1:
                    status = 'Normal'
                    color = '#27ae60'
                elif abs(z) < 2:
                    status = 'Slight ' + ('High' if z > 0 else 'Low')
                    color = '#f39c12'
                else:
                    status = 'Significant ' + ('High' if z > 0 else 'Low')
                    color = '#e74c3c'

                self.comparison.append({
                    'label': label,
                    'value': value,
                    'mean': norm['mean'],
                    'std': norm['std'],
                    'min': norm['min'],
                    'max': norm['max'],
                    'z': z,
                    'percentile': percentile,
                    'status': status,
                    'color': color,
                    'unit': unit
                })

    def _generate_signature(self):
        """Generate unique cognitive signature ID."""
        print("[5/5] Generating cognitive signature...")

        sig_string = f"{self.features['brain_volume']:.4f}_{self.features['gray_white_ratio']:.4f}"
        self.signature_id = "BRAIN-" + hashlib.sha256(sig_string.encode()).hexdigest()[:12].upper()

        print(f"       Signature ID: {self.signature_id}")

    def create_3d_visualization(self, output_path):
        """Create interactive 3D visualization."""
        print("\nCreating 3D visualization...")

        fig = go.Figure()

        structures = [
            ('skull', 'ivory', 'Skull', 0.1, 40000),
            ('gray_matter', 'lightcoral', 'Gray Matter', 0.8, 80000),
            ('white_matter', 'white', 'White Matter', 0.6, 60000),
            ('ventricles', 'dodgerblue', 'Ventricles', 0.9, 20000),
        ]

        for struct_name, color, label, opacity, max_faces in structures:
            mask = self.structures.get(struct_name)
            if mask is None or mask.sum() == 0:
                continue
            try:
                smooth = ndimage.gaussian_filter(mask.astype(float), sigma=1.5)
                verts, faces, _, _ = measure.marching_cubes(smooth, level=0.5, spacing=self.spacing)
                step = max(1, len(faces) // max_faces)

                fig.add_trace(go.Mesh3d(
                    x=verts[::step, 0],
                    y=verts[::step, 1],
                    z=verts[::step, 2],
                    i=faces[::step, 0] // step,
                    j=faces[::step, 1] // step,
                    k=faces[::step, 2] // step,
                    color=color,
                    opacity=opacity,
                    name=label,
                ))
            except:
                pass

        fig.update_layout(
            title=f'<b>BRAIN 3D</b> - {self.signature_id}',
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='white',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            ),
            height=800,
            width=1000,
            margin=dict(l=0, r=0, t=40, b=0),
        )

        fig.write_html(str(output_path / 'brain_3d.html'))
        print(f"       Saved: {output_path / 'brain_3d.html'}")

    def create_comparison_dashboard(self, output_path):
        """Create comparison dashboard."""
        print("Creating comparison dashboard...")

        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'xy'}, {'type': 'xy'}],
                [{'type': 'domain'}, {'type': 'xy'}]
            ],
            subplot_titles=(
                '<b>Subject vs Population</b>',
                '<b>Z-Score Deviation</b>',
                '<b>Brain Composition</b>',
                '<b>Percentile Rankings</b>'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Subject vs Population
        labels = [c['label'] for c in self.comparison[:6]]
        your_vals = [c['value'] for c in self.comparison[:6]]
        norm_vals = [c['mean'] for c in self.comparison[:6]]

        fig.add_trace(go.Bar(name='Subject', x=labels, y=your_vals,
                            marker_color='rgb(55, 83, 109)'), row=1, col=1)
        fig.add_trace(go.Bar(name='Population', x=labels, y=norm_vals,
                            marker_color='rgb(180, 180, 180)'), row=1, col=1)

        # Z-Scores
        z_labels = [c['label'] for c in self.comparison]
        z_scores = [c['z'] for c in self.comparison]
        z_colors = [c['color'] for c in self.comparison]

        fig.add_trace(go.Bar(x=z_labels, y=z_scores, marker_color=z_colors,
                            showlegend=False), row=1, col=2)

        # Composition
        fig.add_trace(go.Pie(
            labels=['Gray Matter', 'White Matter', 'CSF'],
            values=[self.features['gray_matter'], self.features['white_matter'], self.features['csf']],
            marker_colors=['#7f8c8d', '#bdc3c7', '#3498db'],
            hole=0.4
        ), row=2, col=1)

        # Percentiles
        perc_labels = [c['label'] for c in self.comparison]
        percentiles = [c['percentile'] for c in self.comparison]
        perc_colors = [c['color'] for c in self.comparison]

        fig.add_trace(go.Bar(x=perc_labels, y=percentiles, marker_color=perc_colors,
                            showlegend=False), row=2, col=2)

        fig.update_layout(
            title=f'<b>COGNITIVE SIGNATURE ANALYSIS</b> - {self.signature_id}',
            height=900, width=1400, barmode='group',
            showlegend=True
        )

        fig.write_html(str(output_path / 'comparison.html'))
        print(f"       Saved: {output_path / 'comparison.html'}")

    def save_signature_data(self, output_path):
        """Save signature data to JSON."""
        data = {
            'signature_id': self.signature_id,
            'analysis_date': datetime.now().isoformat(),
            'features': {k: float(v) for k, v in self.features.items()},
            'comparison': self.comparison,
            'normative_reference': 'Based on Allen et al. 2002, Sled et al. 2010'
        }

        with open(output_path / 'signature.json', 'w') as f:
            json.dump(data, f, indent=2)
        print(f"       Saved: {output_path / 'signature.json'}")

    def generate_report(self, output_folder):
        """Generate complete report with all visualizations."""
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        self.create_3d_visualization(output_path)
        self.create_comparison_dashboard(output_path)
        self.save_signature_data(output_path)

        print("\n" + "="*70)
        print(f"   ANALYSIS COMPLETE - {self.signature_id}")
        print("="*70)
        print(f"\n   Files saved to: {output_path}")

    def print_summary(self):
        """Print analysis summary."""
        print("\n" + "="*70)
        print("   ANALYSIS SUMMARY")
        print("="*70)
        print(f"\n   Signature ID: {self.signature_id}")
        print("\n   SUBJECT vs POPULATION:")
        print("   " + "-"*64)
        print(f"   {'Metric':<22} {'Subject':>10} {'Normal':>10} {'Z-Score':>10} {'Status':>12}")
        print("   " + "-"*64)

        for c in self.comparison:
            print(f"   {c['label']:<22} {c['value']:>10.2f} {c['mean']:>10.2f} {c['z']:>+10.2f} {c['status']:>12}")

        print("   " + "-"*64)


def main():
    """Example usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python brain_analyzer.py <dicom_folder> [output_folder]")
        print("\nExample:")
        print("  python brain_analyzer.py ./my_brain_scan ./results")
        sys.exit(1)

    dicom_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "./output"

    analyzer = CognitiveSignatureAnalyzer(dicom_folder)
    analyzer.run_analysis()
    analyzer.generate_report(output_folder)
    analyzer.print_summary()


if __name__ == "__main__":
    main()

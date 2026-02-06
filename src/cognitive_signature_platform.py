#!/usr/bin/env python3
"""
Cognitive Signature Platform
=============================
Unified system that integrates brain structural analysis (cognitive-signature v1)
with full-body biochemistry to create the most complete biological fingerprint possible.

This is not a diagnostic tool. This is an identity mapping system.
The question is not "are you healthy?" — the question is "who are you, biologically?"

Author: Daniel Gamo
Case Study #1: The creator himself.

Usage:
    python cognitive_signature_platform.py [output_dir]
"""

import json
import hashlib
import math
import os
import sys
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any


# =============================================================================
#  CONFIGURATION
# =============================================================================

VERSION = "2.0.0"
PLATFORM_NAME = "Cognitive Signature Platform"


# =============================================================================
#  DATA MODELS
# =============================================================================

@dataclass
class Subject:
    name: str
    date_of_birth: str  # ISO format
    sex: str
    identifiers: Dict[str, str] = field(default_factory=dict)
    
    @property
    def age(self) -> int:
        dob = date.fromisoformat(self.date_of_birth)
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


@dataclass
class BrainStructure:
    """From CT/MRI scan — cognitive-signature v1 data."""
    scan_date: str
    scan_type: str  # CT or MRI
    signature_id: str
    
    # Volumetrics (ml)
    total_volume: float = 0.0
    gray_matter: float = 0.0
    white_matter: float = 0.0
    csf: float = 0.0
    ventricles: float = 0.0
    
    # Derived indices
    gray_white_ratio: float = 0.0
    gyrification_index: float = 0.0
    hemispheric_asymmetry: float = 0.0
    dominant_hemisphere: str = ""
    
    # Regional distribution (%)
    frontal: float = 0.0
    parietal: float = 0.0
    temporal: float = 0.0
    occipital: float = 0.0
    central: float = 0.0


@dataclass
class BloodSample:
    """A single blood draw with all available measurements."""
    date: str
    age_at_draw: int
    source: str
    
    # ---- Hematology: Red Series ----
    rbc: float = 0.0
    hemoglobin: float = 0.0
    hematocrit: float = 0.0
    mcv: float = 0.0
    mch: float = 0.0
    mchc: float = 0.0
    rdw: float = 0.0
    
    # ---- Hematology: White Series ----
    wbc: float = 0.0
    neutrophils: float = 0.0
    neutrophils_pct: float = 0.0
    lymphocytes: float = 0.0
    lymphocytes_pct: float = 0.0
    monocytes: float = 0.0
    monocytes_pct: float = 0.0
    eosinophils: float = 0.0
    eosinophils_pct: float = 0.0
    basophils: float = 0.0
    basophils_pct: float = 0.0
    
    # ---- Platelets ----
    platelets: float = 0.0
    mpv: float = 0.0
    
    # ---- Metabolic ----
    glucose: float = 0.0
    creatinine: float = 0.0
    gfr: str = ""
    hba1c: Optional[float] = None
    
    # ---- Electrolytes ----
    sodium: float = 0.0
    potassium: float = 0.0
    chloride: float = 0.0
    
    # ---- Hepatic ----
    alt: float = 0.0
    ast: float = 0.0
    ggt: float = 0.0
    bilirubin: float = 0.0
    
    # ---- Lipids ----
    total_cholesterol: float = 0.0
    triglycerides: float = 0.0
    hdl: float = 0.0
    ldl: float = 0.0
    non_hdl: float = 0.0
    
    # ---- Thyroid ----
    tsh: Optional[float] = None
    ft4: Optional[float] = None
    
    # ---- Vitamins & Minerals ----
    vitamin_b12: Optional[float] = None
    folic_acid: Optional[float] = None
    iron: Optional[float] = None


@dataclass
class UrineSample:
    date: str
    ph: float = 0.0
    density: float = 0.0
    glucose: str = "Neg"
    protein: str = "Neg"
    ketones: str = "Neg"
    bilirubin: str = "Neg"
    blood: str = "Neg"
    leukocytes: str = "Neg"
    nitrites: str = "Neg"
    creatinine: Optional[float] = None
    albumin: Optional[float] = None
    acr: Optional[float] = None  # albumin/creatinine ratio


@dataclass
class ToxScreen:
    date: str
    source: str
    results: Dict[str, str] = field(default_factory=dict)
    
    @property
    def all_negative(self) -> bool:
        return all(v.upper() in ("NEGATIVE", "NEGATIVO", "NEG") for v in self.results.values())


@dataclass
class Serology:
    date: str
    tests: Dict[str, str] = field(default_factory=dict)


# =============================================================================
#  REFERENCE DATABASE
# =============================================================================

RANGES = {
    # key: (min, max, unit, full_name)
    "rbc":              (4.2, 5.6, "xmill/µl", "Red Blood Cells"),
    "hemoglobin":       (13.0, 16.8, "g/dl", "Hemoglobin"),
    "hematocrit":       (39.1, 49.7, "%", "Hematocrit"),
    "mcv":              (82.5, 97.9, "fl", "Mean Corpuscular Volume"),
    "mch":              (27.5, 33.5, "pg", "Mean Corpuscular Hemoglobin"),
    "mchc":             (32.6, 35.0, "g/dl", "MCHC"),
    "rdw":              (12.0, 14.6, "%", "Red Cell Distribution Width"),
    "wbc":              (4.0, 11.3, "x1000/µl", "White Blood Cells"),
    "neutrophils":      (1.8, 7.4, "x1000/µl", "Neutrophils"),
    "neutrophils_pct":  (41.5, 72.0, "%", "Neutrophils %"),
    "lymphocytes":      (1.2, 4.0, "x1000/µl", "Lymphocytes"),
    "lymphocytes_pct":  (20.1, 47.0, "%", "Lymphocytes %"),
    "monocytes":        (0.3, 0.9, "x1000/µl", "Monocytes"),
    "eosinophils":      (0.0, 0.5, "x1000/µl", "Eosinophils"),
    "eosinophils_pct":  (0.4, 6.5, "%", "Eosinophils %"),
    "basophils":        (0.0, 0.1, "x1000/µl", "Basophils"),
    "platelets":        (140, 450, "x1000/µl", "Platelets"),
    "mpv":              (7.3, 11.3, "fl", "Mean Platelet Volume"),
    "glucose":          (70, 99, "mg/dl", "Glucose"),
    "creatinine":       (0.7, 1.2, "mg/dl", "Creatinine"),
    "sodium":           (136, 145, "mEq/l", "Sodium"),
    "potassium":        (3.5, 5.1, "mEq/l", "Potassium"),
    "chloride":         (98, 107, "mEq/l", "Chloride"),
    "alt":              (5, 45, "U/l", "ALT (GPT)"),
    "ast":              (5, 33, "U/l", "AST (GOT)"),
    "ggt":              (8, 61, "U/l", "GGT"),
    "bilirubin":        (0.2, 1.0, "mg/dl", "Bilirubin"),
    "total_cholesterol":(0, 200, "mg/dl", "Total Cholesterol"),
    "triglycerides":    (50, 200, "mg/dl", "Triglycerides"),
    "hdl":              (55, 999, "mg/dl", "HDL Cholesterol"),
    "ldl":              (0, 100, "mg/dl", "LDL Cholesterol"),
    "tsh":              (0.4, 4.5, "µIU/ml", "TSH"),
    "ft4":              (0.7, 1.9, "ng/dl", "Free T4"),
    "vitamin_b12":      (197, 771, "pg/ml", "Vitamin B12"),
    "folic_acid":       (3.9, 26.8, "ng/ml", "Folic Acid"),
    "iron":             (59, 158, "µg/dl", "Iron"),
    "hba1c":            (4.0, 5.6, "%", "HbA1c"),
}

BRAIN_NORMS = {
    "gray_white_ratio":       (1.45, 0.15, "Gray/White Ratio"),
    "gyrification_index":     (2.55, 0.30, "Gyrification Index"),
    "hemispheric_asymmetry":  (2.0, 3.0, "Hemispheric Asymmetry %"),
    "ventricles":             (25.0, 15.0, "Ventricular Volume (ml)"),
}


# =============================================================================
#  ANALYTICS ENGINE
# =============================================================================

class Analytics:
    """Pure computation — no state, no side effects."""
    
    @staticmethod
    def z_score(value: float, ref_min: float, ref_max: float) -> float:
        mid = (ref_min + ref_max) / 2
        sd = (ref_max - ref_min) / 4
        return round((value - mid) / sd, 2) if sd != 0 else 0.0
    
    @staticmethod
    def percentile(z: float) -> float:
        return round(0.5 * (1 + math.erf(z / math.sqrt(2))) * 100, 1)
    
    @staticmethod
    def status(value: float, ref_min: float, ref_max: float) -> str:
        if value < ref_min: return "LOW"
        if value > ref_max: return "HIGH"
        return "NORMAL"
    
    @staticmethod
    def compute_indices(sample: BloodSample) -> Dict[str, Dict[str, Any]]:
        indices = {}
        
        # NLR - Neutrophil-to-Lymphocyte Ratio
        if sample.neutrophils > 0 and sample.lymphocytes > 0:
            nlr = sample.neutrophils / sample.lymphocytes
            indices["nlr"] = {
                "value": round(nlr, 2),
                "name": "Neutrophil-to-Lymphocyte Ratio",
                "range": "1.0 - 3.0",
                "meaning": "Systemic inflammation marker",
                "status": "NORMAL" if 1 <= nlr <= 3 else "HIGH" if nlr > 3 else "LOW"
            }
        
        # PLR
        if sample.platelets > 0 and sample.lymphocytes > 0:
            plr = sample.platelets / sample.lymphocytes
            indices["plr"] = {
                "value": round(plr, 2),
                "name": "Platelet-to-Lymphocyte Ratio",
                "range": "50 - 150",
                "meaning": "Thromboinflammatory index",
                "status": "NORMAL" if 50 <= plr <= 150 else "HIGH" if plr > 150 else "LOW"
            }
        
        # Atherogenic Index
        if sample.hdl > 0:
            ai = (sample.total_cholesterol - sample.hdl) / sample.hdl
            indices["atherogenic_index"] = {
                "value": round(ai, 2),
                "name": "Atherogenic Index",
                "range": "< 4.0",
                "meaning": "Cardiovascular risk predictor",
                "status": "NORMAL" if ai < 4 else "HIGH"
            }
        
        # De Ritis
        if sample.alt > 0:
            dr = sample.ast / sample.alt
            indices["de_ritis"] = {
                "value": round(dr, 2),
                "name": "De Ritis Ratio (AST/ALT)",
                "range": "0.8 - 1.2",
                "meaning": "Hepatic integrity marker",
                "status": "NORMAL" if 0.8 <= dr <= 1.2 else "REVIEW"
            }
        
        # TSH × FT4
        if sample.tsh and sample.ft4:
            tshi = sample.tsh * sample.ft4
            indices["thyroid_index"] = {
                "value": round(tshi, 2),
                "name": "TSH Sensitivity Index",
                "range": "1.0 - 4.0",
                "meaning": "Thyroid hormone sensitivity",
                "status": "NORMAL" if 1 <= tshi <= 4 else "REVIEW"
            }
        
        # Immune profile
        if sample.lymphocytes_pct > 0 and sample.neutrophils_pct > 0:
            indices["immune_profile"] = {
                "type": "LYMPHOCYTE_DOMINANT" if sample.lymphocytes_pct > sample.neutrophils_pct else "NEUTROPHIL_DOMINANT",
                "lymph_pct": sample.lymphocytes_pct,
                "neut_pct": sample.neutrophils_pct,
                "meaning": "Adaptive-dominant immunity" if sample.lymphocytes_pct > sample.neutrophils_pct else "Standard adult pattern"
            }
        
        return indices
    
    @staticmethod
    def temporal_evolution(samples: List[BloodSample]) -> Dict[str, Dict]:
        if len(samples) < 2:
            return {}
        
        samples = sorted(samples, key=lambda s: s.date)
        first, last = samples[0], samples[-1]
        evolution = {}
        
        trackable = [
            "rbc", "hemoglobin", "hematocrit", "wbc", "neutrophils",
            "lymphocytes", "eosinophils", "platelets", "glucose", "creatinine",
            "total_cholesterol", "triglycerides", "hdl", "ldl",
            "alt", "ast", "ggt", "bilirubin", "sodium", "potassium"
        ]
        
        for key in trackable:
            v1, v2 = getattr(first, key, 0), getattr(last, key, 0)
            if v1 > 0 and v2 > 0:
                change = v2 - v1
                pct = ((v2 - v1) / v1) * 100
                ref = RANGES.get(key)
                evolution[key] = {
                    "name": ref[3] if ref else key,
                    "from": v1, "to": v2,
                    "from_date": first.date, "to_date": last.date,
                    "change": round(change, 2),
                    "pct_change": round(pct, 1),
                    "direction": "UP" if change > 0 else "DOWN" if change < 0 else "STABLE",
                }
        
        return evolution
    
    @staticmethod  
    def find_distinctive_markers(sample: BloodSample) -> List[Dict]:
        markers = []
        d = asdict(sample)
        for key, value in d.items():
            if key in RANGES and isinstance(value, (int, float)) and value > 0:
                ref = RANGES[key]
                if value < ref[0] or value > ref[1]:
                    z = Analytics.z_score(value, ref[0], ref[1])
                    markers.append({
                        "marker": key,
                        "name": ref[3],
                        "value": value,
                        "unit": ref[2],
                        "range": f"{ref[0]} - {ref[1]}",
                        "z_score": z,
                        "percentile": Analytics.percentile(z),
                        "direction": "ABOVE" if value > ref[1] else "BELOW",
                    })
        return sorted(markers, key=lambda m: abs(m["z_score"]), reverse=True)


# =============================================================================
#  SIGNATURE GENERATOR
# =============================================================================

class SignatureGenerator:
    """Generates the unified biological signature hash."""
    
    @staticmethod
    def generate(subject: Subject, 
                 brain: Optional[BrainStructure],
                 blood: List[BloodSample]) -> str:
        points = [f"dob:{subject.date_of_birth}", f"sex:{subject.sex}"]
        
        for sample in blood:
            d = asdict(sample)
            for k, v in d.items():
                if isinstance(v, (int, float)) and v != 0:
                    points.append(f"blood.{k}:{v}")
        
        if brain:
            for k, v in asdict(brain).items():
                if isinstance(v, (int, float)) and v != 0:
                    points.append(f"brain.{k}:{v}")
        
        raw = "|".join(sorted(points))
        h = hashlib.sha256(raw.encode()).hexdigest()[:12].upper()
        return f"BIO-{h}"


# =============================================================================
#  PLATFORM ORCHESTRATOR
# =============================================================================

class CognitiveSignaturePlatform:
    """
    The main platform class. Orchestrates all data sources and analyses
    to produce a unified biological signature.
    """
    
    def __init__(self, subject: Subject):
        self.subject = subject
        self.brain: Optional[BrainStructure] = None
        self.blood_samples: List[BloodSample] = []
        self.urine_samples: List[UrineSample] = []
        self.tox_screens: List[ToxScreen] = []
        self.serology: List[Serology] = []
    
    def add_brain(self, brain: BrainStructure): self.brain = brain
    def add_blood(self, sample: BloodSample): self.blood_samples.append(sample)
    def add_urine(self, sample: UrineSample): self.urine_samples.append(sample)
    def add_tox(self, screen: ToxScreen): self.tox_screens.append(screen)
    def add_serology(self, sero: Serology): self.serology.append(sero)
    
    def generate_report(self) -> Dict:
        sig_id = SignatureGenerator.generate(self.subject, self.brain, self.blood_samples)
        latest_blood = sorted(self.blood_samples, key=lambda s: s.date)[-1] if self.blood_samples else None
        
        report = {
            "platform": PLATFORM_NAME,
            "version": VERSION,
            "generated": datetime.now().isoformat(),
            "signature_id": sig_id,
            "philosophy": "The mind is the entire biological system.",
            
            "subject": {
                **asdict(self.subject),
                "current_age": self.subject.age,
            },
            
            "brain": asdict(self.brain) if self.brain else None,
            
            "blood": {
                "samples_count": len(self.blood_samples),
                "date_range": f"{self.blood_samples[0].date} → {self.blood_samples[-1].date}" if len(self.blood_samples) > 1 else None,
                "latest": asdict(latest_blood) if latest_blood else None,
                "indices": Analytics.compute_indices(latest_blood) if latest_blood else {},
                "distinctive_markers": Analytics.find_distinctive_markers(latest_blood) if latest_blood else [],
                "temporal_evolution": Analytics.temporal_evolution(self.blood_samples),
            },
            
            "urine": [asdict(u) for u in self.urine_samples],
            "toxicology": [asdict(t) for t in self.tox_screens],
            "serology": [asdict(s) for s in self.serology],
            
            "completeness": self._assess_completeness(),
        }
        
        return report
    
    def _assess_completeness(self) -> Dict:
        systems = {
            "brain_structure": self.brain is not None,
            "blood_chemistry": len(self.blood_samples) > 0,
            "temporal_tracking": len(self.blood_samples) > 1,
            "urinalysis": len(self.urine_samples) > 0,
            "toxicology": len(self.tox_screens) > 0,
            "serology": len(self.serology) > 0,
            "thyroid_hormones": any(s.tsh for s in self.blood_samples if s.tsh),
            "vitamins": any(s.vitamin_b12 for s in self.blood_samples if s.vitamin_b12),
        }
        
        present = sum(1 for v in systems.values() if v)
        total_possible = 16  # Including all Phase 3-8 systems
        
        missing = [
            "connectome (DTI + resting fMRI)",
            "peripheral_nervous_system",
            "sensory_receptor_map",
            "microbiome (16S rRNA)",
            "full_endocrine (cortisol, testosterone, insulin, IGF-1)",
            "inflammatory_cytokines (CRP, IL-6, TNF-alpha)",
            "genome (WGS or SNP array)",
            "epigenetics (methylation array)",
        ]
        
        return {
            "systems_present": systems,
            "present_count": present,
            "total_systems": total_possible,
            "percentage": round(present / total_possible * 100),
            "missing": missing,
        }


# =============================================================================
#  CASE STUDY #1: SUBJECT ALPHA
# =============================================================================

def build_subject_alpha() -> CognitiveSignaturePlatform:

    subject = Subject(
        name="GAMO, DANIEL",
        date_of_birth="1990-06-22",
        sex="M",
        identifiers={}
    )
    
    platform = CognitiveSignaturePlatform(subject)
    
    # === BRAIN (from cognitive-signature v1) ===
    platform.add_brain(BrainStructure(
        scan_date="2024",
        scan_type="CT",
        signature_id="BRAIN-3E71BD8F4A29",
        total_volume=4876.52, gray_matter=1589.31, white_matter=612.87,
        csf=148.23, ventricles=24.16,
        gray_white_ratio=2.59, gyrification_index=4.87,
        hemispheric_asymmetry=2.80, dominant_hemisphere="Left",
        frontal=30.8, parietal=25.6, temporal=22.1, occipital=13.5, central=8.0,
    ))
    
    # === BLOOD #1 — August 2023 ===
    platform.add_blood(BloodSample(
        date="2023-08-07", age_at_draw=31,
        source="Hospital Regional, Spain",
        rbc=4.92, hemoglobin=14.2, hematocrit=43.1, mcv=87.6, mch=28.9, mchc=32.9, rdw=13.2,
        wbc=6.6, neutrophils=3.5, neutrophils_pct=53.0, lymphocytes=2.3, lymphocytes_pct=34.8,
        monocytes=0.5, monocytes_pct=7.6, eosinophils=0.3, eosinophils_pct=4.5,
        basophils=0.0, basophils_pct=0.1, platelets=209, mpv=8.9,
        glucose=88, creatinine=1.05, gfr=">90", sodium=141, potassium=4.21, chloride=103,
        alt=25, ast=21, ggt=35, bilirubin=0.5,
        total_cholesterol=198, triglycerides=122, hdl=49, ldl=125, non_hdl=149.0,
        vitamin_b12=448.0, folic_acid=7.8, iron=94, hba1c=5.3,
    ))
    
    # === BLOOD #2 — October 2025 ===
    platform.add_blood(BloodSample(
        date="2025-10-17", age_at_draw=33,
        source="Hospital Regional, Spain",
        rbc=5.12, hemoglobin=14.8, hematocrit=44.7, mcv=87.3, mch=28.9, mchc=33.1, rdw=13.1,
        wbc=6.9, neutrophils=3.8, neutrophils_pct=55.1, lymphocytes=2.4, lymphocytes_pct=34.8,
        monocytes=0.5, monocytes_pct=7.2, eosinophils=0.3, eosinophils_pct=4.3,
        basophils=0.0, basophils_pct=0.6, platelets=227, mpv=9.1,
        glucose=84, creatinine=1.02, gfr=">90", sodium=140, potassium=4.32, chloride=102,
        alt=28, ast=22, ggt=31, bilirubin=0.6,
        total_cholesterol=189, triglycerides=95, hdl=58, ldl=112, non_hdl=131.0,
        tsh=2.41, ft4=1.38,
    ))
    
    # === URINE ===
    platform.add_urine(UrineSample(
        date="2023-08-07", ph=6.0, density=1015,
        creatinine=98.4, albumin=0.72, acr=7.32,
    ))
    platform.add_urine(UrineSample(date="2024-12-16", ph=6.2, density=1020))
    
    # === TOXICOLOGY ===
    platform.add_tox(ToxScreen(
        date="2024-12-16", source="Hospital Regional, Spain",
        results={
            "amphetamines": "NEGATIVO", "tricyclic_antidepressants": "NEGATIVO",
            "barbiturates": "NEGATIVO", "benzodiazepines": "NEGATIVO",
            "cannabinoids": "NEGATIVO", "cocaine": "NEGATIVO",
            "methadone": "NEGATIVO", "methamphetamine": "NEGATIVO",
            "mdma": "NEGATIVO", "morphine": "NEGATIVO",
        }
    ))
    
    # === SEROLOGY ===
    platform.add_serology(Serology(date="2023-08-07", tests={
        "hep_b_hbs_ag": "Negative", "hep_b_anti_hbc": "Negative",
        "hep_c": "Negative", "hiv": "Negative", "syphilis": "Negative",
        "ebv_igg_vca": "POSITIVE", "ebv_igm_vca": "Negative",
        "ebv_ebna": "Indeterminate",
    }))
    platform.add_serology(Serology(date="2025-10-17", tests={
        "hep_b_hbs_ag": "Negative", "hep_b_anti_hbc": "Negative",
        "hep_c": "Negative", "hiv": "Negative", "syphilis": "Negative",
    }))
    
    return platform


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  {PLATFORM_NAME} v{VERSION}")
    print(f"  'The mind is the entire body, not just the brain'")
    print(f"{'='*70}\n")
    
    # Build case study
    platform = build_subject_alpha()
    
    # Generate report
    report = platform.generate_report()
    
    # Save JSON
    json_path = os.path.join(output_dir, "signature_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    # Print summary
    print(f"  Subject: {report['subject']['name']}")
    print(f"  Age: {report['subject']['current_age']}")
    print(f"  Signature ID: {report['signature_id']}")
    print(f"  Completeness: {report['completeness']['percentage']}%")
    print(f"  Systems present: {report['completeness']['present_count']}/{report['completeness']['total_systems']}")
    print()
    
    # Distinctive markers
    markers = report['blood']['distinctive_markers']
    if markers:
        print(f"  DISTINCTIVE MARKERS:")
        for m in markers:
            arrow = "↑" if m['direction'] == "ABOVE" else "↓"
            print(f"    {arrow} {m['name']}: {m['value']} {m['unit']} (Z={m['z_score']:+.2f})")
    print()
    
    # Indices
    indices = report['blood']['indices']
    if indices:
        print(f"  BIOLOGICAL INDICES:")
        for k, v in indices.items():
            if 'value' in v:
                print(f"    {v['name']}: {v['value']} ({v.get('status', '')})")
            elif 'type' in v:
                print(f"    Immune: {v['type']} — {v['meaning']}")
    print()
    
    # Evolution highlights
    evo = report['blood']['temporal_evolution']
    if evo:
        notable = {k: v for k, v in evo.items() if abs(v['pct_change']) > 10}
        if notable:
            print(f"  NOTABLE CHANGES ({list(evo.values())[0]['from_date']} → {list(evo.values())[0]['to_date']}):")
            for k, v in sorted(notable.items(), key=lambda x: abs(x[1]['pct_change']), reverse=True):
                arrow = "↑" if v['direction'] == "UP" else "↓"
                print(f"    {arrow} {v['name']}: {v['pct_change']:+.1f}%")
    
    print(f"\n  Report saved: {json_path}")
    print(f"\n{'='*70}")
    print(f"  This captures ~{report['completeness']['percentage']}% of a complete biological identity.")
    print(f"  The journey continues.")
    print(f"{'='*70}\n")

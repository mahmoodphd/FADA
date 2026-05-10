"""Constants for the FADA HuggingFace Space.

Prompts, class mappings, color palette, and inference settings.
Ported from expert_eval/constants.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

INTERPRET_PROMPT = (
    "Provide a comprehensive clinical interpretation of this fetal ultrasound "
    "image. Identify all visible anatomical structures, describe the fetal "
    "orientation and imaging plane, note any measurable biometric parameters, "
    "estimate the gestational age, assess image quality, evaluate normality, "
    "and provide clinical recommendations. "
    "Return your answer as a JSON object with keys: "
    "anatomical_structures, fetal_orientation, imaging_plane, "
    "biometric_measurements, gestational_age, image_quality, "
    "normality_assessment, clinical_recommendations."
)

CLASSIFY_PROMPT = (
    "Classify this fetal ultrasound image. "
    "What anatomical view or structure does it show? "
    "Return a JSON object with \"label\" as the classification."
)

# ---------------------------------------------------------------------------
# Classification -> Detection / Segmentation class mapping
# ---------------------------------------------------------------------------

CLASSIFY_TO_DETECT: Dict[str, str] = {
    "huvf": "abdomen, arm, head, legs",
    "huvb": "abdomen, arm, head, legs",
    "hdvf": "abdomen, arm, head, legs",
    "hdvb": "abdomen, arm, head, legs",
    "legs": "abdomen, legs",
    "arms": "abdomen, arm",
    "abdomen": "abdomen",
    "Abd_plane": "abdomen, arm",
    "head": "head",
    "BPD_PLANE": "Brain, CSP, LV",
    "FL_PLANE": "legs",
    "NO_PLANE": "abdomen, arm, head, legs",
    "Aorta": "cardiac, thorax",
    "Flows": "artery, liver, stomach, vein",
    "V sign": "cardiac, thorax",
    "X sign": "cardiac, thorax",
    "Other": "abdomen, arm, head, legs",
    "AC_PLANE": "abdomen, arm, head, legs",
    "FL": "legs",
    "liver": "artery, liver, stomach, vein",
}

CLASSIFY_TO_SEGMENT: Dict[str, Optional[str]] = {
    "BPD_PLANE": "Brain",
    "Aorta": "cardiac, thorax",
    "Flows": "artery, liver, stomach, vein",
    "V sign": "cardiac, thorax",
    "X sign": "cardiac, thorax",
}

SPECIFIC_CLS_LABELS = {
    "Aorta", "Flows", "BPD_PLANE", "FL_PLANE", "FL",
    "V sign", "X sign", "liver",
}

# ---------------------------------------------------------------------------
# Keyword-based mapping (ordered specific-first)
# ---------------------------------------------------------------------------

KEYWORD_FALLBACK: List[Tuple[List[str], str, Optional[str]]] = [
    (["cerebel", "cervellet", "ventric", "thalm", "bpd",
      "banana", "holopros", "cisterna", "fossa", "csp", "cavum",
      "septum pellucidum", "lateral ventricle"],
     "Brain, CSP, LV", "Brain"),
    (["brain"],
     "Brain, CSP, LV", "Brain"),
    (["cardiac", "thorax", "thorac", "thoric", "thorcic",
      "aort", "heart", "stenos", "chamber", "valve"],
     "cardiac, thorax", "cardiac, thorax"),
    (["fetal head", "fetal_head", "cervix", "cervic", "cervex", "pubic", "symphys",
      "pelvimetr", "angle of progression"],
     "fetal_head, pubic_symphysis", "fetal_head, pubic_symphysis"),
    (["femur", "femoral", "fetal.femur", "fetal-femur"],
     "legs", None),
    (["nasal", "nuchal", "translucen"],
     "NT, nasal_bone, nasal_skin, nasal_tip", "NT"),
    (["crl", "crown", "rump"],
     "B, CRL, H, NB", None),
    (["liver", "stomach", "vein", "artery"],
     "artery, liver, stomach, vein", "artery, liver, stomach, vein"),
    (["abdomen", "abdom"],
     "abdomen", None),
    (["arm"],
     "abdomen, arm", None),
    (["head"],
     "head", None),
    (["leg"],
     "abdomen, legs", None),
]

DEFAULT_DETECT_CLASSES = "abdomen, arm, head, legs"


def _keyword_lookup(text: str) -> Optional[Tuple[str, Optional[str], str]]:
    """Check text against KEYWORD_FALLBACK. Returns (det, seg, kw) or None."""
    text_lower = text.lower()
    for keywords, det_classes, seg_classes in KEYWORD_FALLBACK:
        for kw in keywords:
            if kw in text_lower:
                return det_classes, seg_classes, kw
    return None


# ---------------------------------------------------------------------------
# Per-class keyword scoring for interpretation-first mapping
# ---------------------------------------------------------------------------

FIELD_WEIGHTS: Dict[str, float] = {
    "biometric_measurements": 3.0,
    "imaging_plane": 2.5,
    "anatomical_structures": 1.0,
    "fetal_orientation": 0.5,
    "normality_assessment": 0.5,
    "gestational_age": 0.3,
    "image_quality": 0.0,
    "clinical_recommendations": 0.3,
}

CLASS_KEYWORDS: Dict[str, List[str]] = {
    "Brain": ["brain", "cerebr", "cerebellum", "cortex", "vermis",
              "falx", "interhemispheric", "cranial", "intracranial",
              "parieto-occipital", "frontal lobe", "occipital"],
    "CSP":   ["septum pellucidum", "cavum", "csp", "midline echo"],
    "LV":    ["lateral ventricle", "ventricl", "choroid plexus",
              "ventriculomegaly"],
    "cardiac": ["heart", "cardiac", "chamber", "four chamber",
                "4-chamber", "valve", "myocard", "atri",
                "foramen ovale", "interventricular", "aort", "aortic"],
    "thorax":  ["thorax", "thorac", "thoric", "thorcic", "chest",
                "lung", "pleural", "rib cage", "diaphragm", "mediastin"],
    "NT":         ["nuchal translucen", "nuchal fold", "nuchal",
                   " nt ", " nt,", "nt_", "nt "],
    "nasal_bone": ["nasal bone", "nasal bridge"],
    "nasal_skin": ["nasal skin"],
    "nasal_tip":  ["nasal tip"],
    "CRL": ["crown-rump", "crown rump", "crl"],
    "B":   [],
    "H":   [],
    "NB":  [],
    "artery":  ["artery", "arter", "umbilical arter"],
    "liver":   ["liver", "hepat"],
    "stomach": ["stomach", "gastric", "stomach bubble"],
    "vein":    ["vein", "venous", "vena cava", "umbilical vein"],
    "fetal_head":       ["cervix", "cervic", "cervical length"],
    "pubic_symphysis":  ["pubic", "symphys", "pelvimetr"],
    "abdomen": ["abdomen", "abdomin", "abdom", "ac_plane"],
    "arm":     ["arm", "upper limb", "humerus"],
    "head":    ["skull", "calvar", "biparietal", "fetal head", "head", "neck"],
    "legs":    ["femur", "femoral", "thigh", "lower limb",
                "lower extremit", "tibia", "legs", "leg "],
}

CO_OCCURRENCE_GROUPS: List[Tuple[str, List[str], Optional[List[str]], List[str]]] = [
    ("brain",      ["Brain", "CSP", "LV"],                           ["Brain"],                                ["nt_nasal"]),
    ("cardiac",    ["cardiac", "thorax"],                             ["cardiac", "thorax"],                    []),
    ("nt_nasal",   ["NT", "nasal_bone", "nasal_skin", "nasal_tip"],  ["NT"],                                   ["brain", "crl"]),
    ("crl",        ["B", "CRL", "H", "NB"],                          None,                                     ["nt_nasal"]),
    ("doppler",    ["artery", "liver", "stomach", "vein"],            ["artery", "liver", "stomach", "vein"],   []),
    ("pelvimetry", ["fetal_head", "pubic_symphysis"],                 ["fetal_head", "pubic_symphysis"],         []),
    ("body_full",  ["abdomen", "arm", "head", "legs"],                None,                                     []),
    ("femur",      ["legs"],                                          None,                                     []),
]

GROUP_ALIASES: Dict[str, List[str]] = {
    "brain":      ["bpd", "biparietal", " hc ", "head circumference",
                   "v sign", "x sign", "trans-thalamic", "trans-cerebellar",
                   "transventricular", "transcerebellar"],
    "crl":        ["first trimester", "dating scan", "embryo",
                   "sagittal profile"],
    "femur":      [" fl ", "femur length", "fl_plane", "fl plane", "fl "],
    "body_full":  ["abdominal circumference", " ac ", "ac_plane",
                   "ac plane", "fetal pose", "body habitus",
                   "no_plane", "no plane"],
    "doppler":    ["doppler", "flow", "waveform", "pulsatility",
                   "resistance index", "stomach bubble",
                   "hepatic", "ductus venosus", "umbilical arter",
                   "umbilical vein"],
    "nt_nasal":   ["first trimester screen", "aneuploidy", "trisomy",
                   "down syndrome"],
    "pelvimetry": ["pelvimetry", "pelvic", "birth canal",
                   "vaginal delivery"],
    "cardiac":    ["echocardiogra", "outflow tract", "aortic arch", "aorta"],
}

PLANE_TO_GROUP: Dict[str, str] = {
    "trans-thalamic": "brain", "transthalamic": "brain",
    "trans-cerebellar": "brain", "transcerebellar": "brain",
    "transventricular": "brain", "biparietal": "brain",
    "four chamber": "cardiac", "4-chamber": "cardiac",
    "4 chamber": "cardiac", "outflow tract": "cardiac",
    "three vessel": "cardiac", "aortic": "cardiac",
    "nuchal": "nt_nasal",
    "longitudinal view of femur": "femur", "femur length": "femur",
    "abdominal circumference": "body_full", "ac plane": "body_full",
    "crown-rump": "crl", "crown rump": "crl",
    "doppler": "doppler", "waveform": "doppler",
    "abdominal organ": "doppler", "stomach bubble": "doppler",
    "cervical length": "pelvimetry",
}

NEGATIVE_MEASUREMENT_PATTERNS: List[str] = [
    "not measurable", "not visible", "cannot be measured",
    "unable to measure", "not assessed", "not obtained",
    "not clearly", "inadequate",
]

# ---------------------------------------------------------------------------
# Routing group map: informal name -> (det_classes, seg_classes)
# Used by intent parser to map user requests to class strings
# ---------------------------------------------------------------------------

ROUTING_GROUP_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    "brain":      ("Brain, CSP, LV",                          "Brain"),
    "cardiac":    ("cardiac, thorax",                          "cardiac, thorax"),
    "nt_nasal":   ("NT, nasal_bone, nasal_skin, nasal_tip",   "NT"),
    "crl":        ("B, CRL, H, NB",                           None),
    "doppler":    ("artery, liver, stomach, vein",             "artery, liver, stomach, vein"),
    "pelvimetry": ("fetal_head, pubic_symphysis",              "fetal_head, pubic_symphysis"),
    "body_pose":  ("abdomen, arm, head, legs",                 None),
    "femur":      ("legs",                                     None),
    "crl_kp":     ("CRL_KP, ScaleBarKpoints", None),
    "nt_kp":      ("NTKpoints, ScaleBarKpoints", None),
}

# ---------------------------------------------------------------------------
# Color palette for annotation rendering
# ---------------------------------------------------------------------------

_RAW_PALETTE = [
    "#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231",
    "#911EB4", "#42D4F4", "#F032E6", "#BFEF45", "#FABED4",
    "#469990", "#DCBEFF", "#9A6324", "#FFFAC8", "#800000",
    "#AAFFC3", "#808000", "#FFD8B1", "#000075", "#A9A9A9",
]

ALL_CLASSES_SORTED = sorted({
    "AB", "B", "Brain", "C", "CRL", "CRL_KP", "CSP", "DP", "G", "H", "LV",
    "MDS", "MLS", "MX", "NB", "NT", "NTAPS", "NTKpoints", "RBP",
    "ScaleBar", "ScaleBarKpoints",
    "abdomen", "arm", "artery", "cardiac",
    "fetal_head", "head", "legs", "liver",
    "nasal_bone", "nasal_skin", "nasal_tip",
    "pubic_symphysis", "stomach", "thorax", "vein",
})

CLASS_COLORS: Dict[str, str] = {
    cls_name: _RAW_PALETTE[i % len(_RAW_PALETTE)]
    for i, cls_name in enumerate(ALL_CLASSES_SORTED)
}

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------

TEMPERATURE = 0.1
MAX_NEW_TOKENS = 1024

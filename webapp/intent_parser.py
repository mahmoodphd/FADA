"""Intent parser: maps natural language chat messages to task types.

Rule-based keyword matching for the narrow domain of fetal ultrasound
analysis. Detects: interpret, detect, segment, classify, greeting.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from constants import (
    DEFAULT_DETECT_CLASSES,
    KEYWORD_FALLBACK,
    ROUTING_GROUP_MAP,
)


@dataclass
class IntentResult:
    task: str  # "interpret", "detect", "segment", "classify", "greeting", "general"
    classes: Optional[str] = None
    seg_classes: Optional[str] = None


# Keyword banks for intent detection
_INTERPRET_KW = [
    "interpret", "analyze", "analyse", "describe", "assessment",
    "what do you see", "what is this", "clinical interpretation",
    "examine", "evaluate", "tell me about",
]

_DETECT_KW = [
    "detect", "find", "locate", "where is", "where are",
    "bounding box", "identify", "show me", "mark", "highlight",
]

_SEGMENT_KW = [
    "segment", "outline", "contour", "mask", "delineate",
    "boundary", "trace", "polygon",
]

_CLASSIFY_KW = [
    "classify", "what type", "what view", "which plane",
    "what kind", "categorize", "classification",
]

_GREETING_KW = [
    "hello", "hi ", "hey", "good morning", "good afternoon",
    "thanks", "thank you",
]

# Informal anatomy names -> ROUTING_GROUP_MAP keys
#
# ORDERING IS CRITICAL -- matching is substring-based and first match wins.
# Rules:
#   1. Compound specific terms FIRST (e.g. "nt keypoint" before "keypoint")
#   2. Multi-word before single-word (e.g. "fetal head" before "head")
#   3. Keypoint terms before "crl" (so "CRL keypoints" -> crl_kp, not crl)
#   4. Everything else in logical groups
#
# NOTE: standalone "nt" is NOT here because it matches inside "segment".
# NT detection is handled via "nuchal"/"translucency" + regex fallback.
_ANATOMY_TO_GROUP = {
    # Compound terms with "keypoint" + anatomy -- MUST be before generic "keypoint"
    "nt keypoint": "nt_kp",
    "nt kp": "nt_kp",
    "nuchal keypoint": "nt_kp",
    "ntkpoint": "nt_kp",
    "brain keypoint": "brain",
    "cardiac keypoint": "cardiac",
    # Pelvimetry -- specific multi-word terms BEFORE generic "head"
    "fetal head": "pelvimetry", "fetal_head": "pelvimetry",
    "angle of progression": "pelvimetry",
    # CRL + keypoints -- BEFORE standalone "crl"
    "keypoint": "crl_kp", "kpoint": "crl_kp",
    "scalebar": "crl_kp", "scale bar": "crl_kp",
    "crl_kp": "crl_kp", "scalebarpoint": "crl_kp",
    # Brain / BPD group
    "brain": "brain", "cerebral": "brain", "cerebellum": "brain",
    "ventricle": "brain", "thalami": "brain", "bpd": "brain",
    "head circumference": "brain",
    "csp": "brain", "cavum": "brain", "septum pellucidum": "brain",
    "lv": "brain", "lateral ventricle": "brain",
    "choroid plexus": "brain", "falx": "brain",
    # Cardiac group (+ common typos)
    "heart": "cardiac", "cardiac": "cardiac", "thorax": "cardiac",
    "thoracic": "cardiac", "aorta": "cardiac", "four chamber": "cardiac",
    "cardic": "cardiac", "cariac": "cardiac",
    # NT / nasal group (+ common typos)
    "nuchal": "nt_nasal", "nasal": "nt_nasal",
    "translucency": "nt_nasal", "nasal bone": "nt_nasal",
    "nasal tip": "nt_nasal", "nasal skin": "nt_nasal",
    "nuchel": "nt_nasal", "translucen": "nt_nasal",
    # CRL group (AFTER keypoint entries)
    "crown-rump": "crl", "crown rump": "crl", "crl": "crl",
    # Doppler group
    "doppler": "doppler", "flow": "doppler", "vessel": "doppler",
    "artery": "doppler", "vein": "doppler", "umbilical": "doppler",
    # Pelvimetry group (additional keywords + typos)
    "cervix": "pelvimetry", "pelvimetry": "pelvimetry",
    "symphysis": "pelvimetry", "pubic": "pelvimetry",
    "symphsis": "pelvimetry", "symphis": "pelvimetry",
    "pelvimtry": "pelvimetry",
    # Body / pose group
    "body": "body_pose", "abdomen": "body_pose", "arm": "body_pose",
    "legs": "body_pose", "pose": "body_pose",
    "abdomin": "body_pose",
    # Femur group
    "femur": "femur", "thigh": "femur", "femor": "femur",
}


def parse_intent(
    user_text: str,
    has_image: bool = False,
    has_interpretation: bool = False,
) -> IntentResult:
    """Parse user message into a task intent."""
    text = user_text.strip().lower()

    if not text and has_image:
        return IntentResult(task="interpret")

    # Check for greetings
    for kw in _GREETING_KW:
        if text.startswith(kw) or f" {kw}" in text:
            if has_image and not has_interpretation:
                return IntentResult(task="interpret")
            return IntentResult(task="greeting")

    # Check for segment intent (before detect, since "segment" is more specific)
    for kw in _SEGMENT_KW:
        if kw in text:
            det_cls, seg_cls = _extract_classes(text)
            return IntentResult(
                task="segment",
                classes=det_cls or seg_cls,
                seg_classes=seg_cls,
            )

    # Check for detect intent
    for kw in _DETECT_KW:
        if kw in text:
            det_cls, seg_cls = _extract_classes(text)
            return IntentResult(task="detect", classes=det_cls)

    # Check for classify intent
    for kw in _CLASSIFY_KW:
        if kw in text:
            return IntentResult(task="classify")

    # Check for interpret intent
    for kw in _INTERPRET_KW:
        if kw in text:
            return IntentResult(task="interpret")

    # Default: if image present and no interpretation yet -> interpret
    if has_image and not has_interpretation:
        return IntentResult(task="interpret")

    # If interpretation exists, try to detect anatomy references
    if has_interpretation:
        det_cls, seg_cls = _extract_classes(text)
        if det_cls:
            return IntentResult(task="detect", classes=det_cls)

    return IntentResult(task="general")


def _extract_classes(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract detection/segmentation class strings from user text."""
    text_lower = text.lower()

    # Check against anatomy-to-group mapping (first match wins)
    for anatomy_name, group_key in _ANATOMY_TO_GROUP.items():
        if anatomy_name in text_lower:
            if group_key in ROUTING_GROUP_MAP:
                det, seg = ROUTING_GROUP_MAP[group_key]
                return det, seg

    # Fallback to keyword lookup
    for keywords, det_classes, seg_classes in KEYWORD_FALLBACK:
        for kw in keywords:
            if kw in text_lower:
                return det_classes, seg_classes

    # Word-boundary check for "nt" (can't be in _ANATOMY_TO_GROUP
    # because "nt" matches inside "segment")
    if re.search(r'\bnt\b', text_lower):
        if "nt_nasal" in ROUTING_GROUP_MAP:
            det, seg = ROUTING_GROUP_MAP["nt_nasal"]
            return det, seg

    # Check for "all" or "everything"
    if re.search(r'\b(all|every|everything)\b', text_lower):
        return DEFAULT_DETECT_CLASSES, None

    return None, None

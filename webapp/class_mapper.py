"""Class mapper: interpretation-to-class cascade mapping.

Ported from expert_eval/run_interpret_first.py.
Implements the 5-priority cascade that maps clinical interpretation
+ classification label to detection/segmentation class strings.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from constants import (
    CLASS_KEYWORDS,
    CLASSIFY_TO_DETECT,
    CLASSIFY_TO_SEGMENT,
    CO_OCCURRENCE_GROUPS,
    DEFAULT_DETECT_CLASSES,
    FIELD_WEIGHTS,
    GROUP_ALIASES,
    NEGATIVE_MEASUREMENT_PATTERNS,
    PLANE_TO_GROUP,
    SPECIFIC_CLS_LABELS,
    _keyword_lookup,
)

logger = logging.getLogger(__name__)


def parse_interpretation_json(raw_text: str) -> Dict[str, Any]:
    """Parse interpretation JSON from model output.

    Returns dict with interpretation fields, or a fallback dict on failure.
    """
    text = raw_text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data["_parse_success"] = True
            return data
    except json.JSONDecodeError:
        pass

    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start:brace_end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                data["_parse_success"] = True
                return data
        except json.JSONDecodeError:
            pass

    return {"_parse_success": False, "_raw_text": text}


def _flatten_field_text(value: Any) -> str:
    """Flatten an interpretation field value to a searchable string."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def _score_classes_per_field(
    interp_parsed: Dict[str, Any],
    field_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Score each detection class against interpretation fields."""
    weights = field_weights if field_weights is not None else FIELD_WEIGHTS
    scores: Dict[str, float] = {cls: 0.0 for cls in CLASS_KEYWORDS}

    for field_name, weight in weights.items():
        if weight == 0.0:
            continue
        raw = interp_parsed.get(field_name)
        if not raw:
            continue
        text = _flatten_field_text(raw).lower()
        if not text:
            continue

        for cls_name, keywords in CLASS_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    scores[cls_name] += weight
                    break

    return scores


def _aggregate_group_scores(
    class_scores: Dict[str, float],
    interp_parsed: Dict[str, Any],
) -> List[Tuple[str, float, List[str], Optional[List[str]], List[str]]]:
    """Aggregate class scores into co-occurrence group scores."""
    full_text = " ".join(
        _flatten_field_text(v) for k, v in interp_parsed.items()
        if not k.startswith("_") and v
    ).lower()

    group_results = []
    for group_name, det_classes, seg_classes, compatible in CO_OCCURRENCE_GROUPS:
        gscore = sum(class_scores.get(c, 0.0) for c in det_classes)

        aliases = GROUP_ALIASES.get(group_name, [])
        for alias in aliases:
            if alias in full_text:
                gscore += 2.0
                break

        group_results.append(
            (group_name, gscore, det_classes, seg_classes, compatible))

    group_results.sort(key=lambda x: x[1], reverse=True)
    return group_results


def _extract_cls_label(cls_result: Dict[str, Any]) -> Optional[str]:
    """Extract the classification label string from a classify result dict."""
    if not cls_result:
        return None
    parsed = cls_result.get("parsed", cls_result)
    label = parsed.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    return None


def map_interpretation_to_classes(
    interp_parsed: Dict[str, Any],
    cls_label: Optional[str] = None,
) -> Tuple[str, Optional[str], str]:
    """Map interpretation + classification to detection/segmentation classes.

    5-priority cascade:
      P1: SPECIFIC cls_label exact match
      P2: imaging_plane / orientation -> PLANE_TO_GROUP
      P3: keyword scoring from interpretation
      P4: generic cls_label fallback
      P5: default fallback

    Returns (det_classes_str, seg_classes_or_None, mapping_tier).
    """
    # P1: SPECIFIC classify label exact match
    if cls_label and cls_label in SPECIFIC_CLS_LABELS and cls_label in CLASSIFY_TO_DETECT:
        det = CLASSIFY_TO_DETECT[cls_label]
        seg = CLASSIFY_TO_SEGMENT.get(cls_label)
        return det, seg, f"cls_specific_{cls_label}"

    # P2: imaging plane / orientation -> group
    plane_text = _flatten_field_text(
        interp_parsed.get("imaging_plane", "")).lower()
    orient_text = _flatten_field_text(
        interp_parsed.get("fetal_orientation", "")).lower()
    combined_plane = f"{plane_text} {orient_text}"

    for plane_key, group_name in PLANE_TO_GROUP.items():
        if plane_key in combined_plane:
            for gname, det_cls, seg_cls, _compat in CO_OCCURRENCE_GROUPS:
                if gname == group_name:
                    det_str = ", ".join(det_cls)
                    seg_str = ", ".join(seg_cls) if seg_cls else None
                    return det_str, seg_str, f"plane_{plane_key}"
            break

    # P3: keyword scoring from interpretation
    bio_text = _flatten_field_text(
        interp_parsed.get("biometric_measurements", "")).lower()
    has_negative = any(pat in bio_text for pat in NEGATIVE_MEASUREMENT_PATTERNS)

    if has_negative:
        dampened_weights = dict(FIELD_WEIGHTS)
        dampened_weights["biometric_measurements"] = 0.3
        class_scores = _score_classes_per_field(interp_parsed, dampened_weights)
    else:
        class_scores = _score_classes_per_field(interp_parsed)

    groups = _aggregate_group_scores(class_scores, interp_parsed)

    # --- Disambiguation: body_full vs doppler ---
    # "abdomen/abdominal" appears in many interpretation fields for organ views,
    # inflating body_full score. If specific organ keywords (stomach, liver,
    # artery, vein) are present, doppler should win over body_full.
    _ORGAN_KEYWORDS = ("stomach", "liver", "hepat", "artery", "arter",
                       "vein", "venous", "gastric", "ductus")
    if len(groups) >= 2:
        gnames = {g[0]: i for i, g in enumerate(groups)}
        if ("body_full" in gnames and "doppler" in gnames
                and groups[0][0] == "body_full"):
            full_text = " ".join(
                _flatten_field_text(v) for k, v in interp_parsed.items()
                if not k.startswith("_") and v
            ).lower()
            if any(kw in full_text for kw in _ORGAN_KEYWORDS):
                # Swap: promote doppler above body_full
                di = gnames["doppler"]
                groups[0], groups[di] = groups[di], groups[0]
                logger.info("Disambiguated: organ keywords found, "
                            "promoting doppler over body_full")

    if groups and groups[0][1] > 0.0:
        top_name, top_score, top_det, top_seg, top_compat = groups[0]

        det_classes = list(top_det)
        seg_classes = list(top_seg) if top_seg else None
        tier = f"interp_{top_name}"

        if len(groups) > 1:
            sec_name, sec_score, sec_det, sec_seg, sec_compat = groups[1]
            if (sec_score > 0 and sec_score >= top_score * 0.3
                    and sec_name in top_compat):
                det_classes.extend(sec_det)
                if sec_seg:
                    if seg_classes is None:
                        seg_classes = []
                    seg_classes.extend(sec_seg)
                tier = f"interp_{top_name}+{sec_name}"

        det_classes = det_classes[:5]
        if seg_classes:
            seg_classes = seg_classes[:5]

        det_str = ", ".join(det_classes)
        seg_str = ", ".join(seg_classes) if seg_classes else None
        return det_str, seg_str, tier

    # P4: generic cls_label fallback
    if cls_label and cls_label in CLASSIFY_TO_DETECT:
        det = CLASSIFY_TO_DETECT[cls_label]
        seg = CLASSIFY_TO_SEGMENT.get(cls_label)
        return det, seg, f"cls_generic_{cls_label}"

    if cls_label:
        hit = _keyword_lookup(cls_label)
        if hit:
            det, seg, kw = hit
            return det, seg, f"cls_keyword_{kw}"

    # P5: default fallback
    return DEFAULT_DETECT_CLASSES, None, "fallback_no_match"

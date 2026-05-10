"""Output parser: extracts structured predictions from model text output.

Parses JSON responses from the Qwen3.5-VL model into structured detection,
segmentation, and classification predictions.

Coordinate convention:
- Model outputs bbox_2d as [x_min, y_min, x_max, y_max] normalized to [0, 1000).
- All coordinates in ParsedOutput are in this normalized space.
- Use rescale_predictions_to_original() to convert back to pixel space.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

COORD_SCALE = 1000


@dataclass
class DetectionPrediction:
    label: str
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float = 1.0


@dataclass
class SegmentationPrediction:
    label: str
    bbox_xyxy: Tuple[int, int, int, int]
    mask_polygon: List[Tuple[int, int]]
    confidence: float = 1.0


@dataclass
class ClassificationPrediction:
    label: str
    confidence: float = 1.0


@dataclass
class KeypointPrediction:
    label: str
    bbox_xyxy: Tuple[int, int, int, int]
    keypoints: List[Tuple[int, int, int]]
    confidence: float = 1.0


@dataclass
class ParsedOutput:
    detections: List[DetectionPrediction] = field(default_factory=list)
    segmentations: List[SegmentationPrediction] = field(default_factory=list)
    classifications: List[ClassificationPrediction] = field(default_factory=list)
    keypoints: List[KeypointPrediction] = field(default_factory=list)
    raw_text: str = ""
    parse_errors: List[str] = field(default_factory=list)


def parse_model_output(text: str) -> ParsedOutput:
    """Parse model text output into structured predictions."""
    result = ParsedOutput(raw_text=text)
    json_data = _extract_json(text)
    if json_data is None:
        result.parse_errors.append(f"No valid JSON found in output: {text[:200]}")
        return result

    if isinstance(json_data, dict):
        items = [json_data]
    elif isinstance(json_data, list):
        items = json_data
    else:
        result.parse_errors.append(f"Unexpected JSON type: {type(json_data)}")
        return result

    for item in items:
        if not isinstance(item, dict):
            continue
        _parse_item(item, result)

    return result


def _extract_json(text: str) -> Optional[Any]:
    """Extract JSON from model output text."""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    array_match = re.search(r'\[.*\]', text, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group())
        except json.JSONDecodeError:
            pass

    obj_match = re.search(r'\{.*\}', text, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group())
        except json.JSONDecodeError:
            pass

    cleaned = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return None


def _parse_item(item: Dict[str, Any], result: ParsedOutput) -> None:
    """Parse a single JSON item into the appropriate prediction type."""
    label = item.get("label", "")
    bbox_2d = item.get("bbox_2d") or item.get("box_2d")
    mask = item.get("mask")
    keypoints_data = item.get("keypoints")

    if not label:
        result.parse_errors.append(f"Item missing label: {item}")
        return

    if bbox_2d is None:
        result.classifications.append(ClassificationPrediction(label=label))
        return

    if not isinstance(bbox_2d, list) or len(bbox_2d) != 4:
        result.parse_errors.append(f"Invalid bbox_2d: {bbox_2d}")
        return

    try:
        bbox = tuple(int(v) for v in bbox_2d)
    except (ValueError, TypeError):
        result.parse_errors.append(f"Non-numeric bbox_2d values: {bbox_2d}")
        return

    if keypoints_data is not None:
        try:
            kps = [(int(k[0]), int(k[1]), int(k[2])) for k in keypoints_data]
            result.keypoints.append(
                KeypointPrediction(label=label, bbox_xyxy=bbox, keypoints=kps))
        except (ValueError, TypeError, IndexError) as e:
            result.parse_errors.append(f"Invalid keypoints: {e}")
        return

    if mask is not None:
        try:
            polygon = [(int(pt[0]), int(pt[1])) for pt in mask]
            if len(polygon) >= 3:
                result.segmentations.append(
                    SegmentationPrediction(
                        label=label, bbox_xyxy=bbox, mask_polygon=polygon))
            else:
                result.parse_errors.append(
                    f"Mask polygon too few points: {len(polygon)}")
        except (ValueError, TypeError, IndexError) as e:
            result.parse_errors.append(f"Invalid mask polygon: {e}")
        return

    result.detections.append(DetectionPrediction(label=label, bbox_xyxy=bbox))


def rescale_predictions_to_original(
    parsed: ParsedOutput,
    original_wh: Tuple[int, int],
) -> ParsedOutput:
    """Rescale all coordinates from [0, 1000) normalized space to original pixels."""
    ow, oh = original_wh

    def scale_box(box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return (
            round(x1 * ow / COORD_SCALE),
            round(y1 * oh / COORD_SCALE),
            round(x2 * ow / COORD_SCALE),
            round(y2 * oh / COORD_SCALE),
        )

    def scale_point(x: int, y: int) -> Tuple[int, int]:
        return (round(x * ow / COORD_SCALE), round(y * oh / COORD_SCALE))

    result = ParsedOutput(raw_text=parsed.raw_text,
                          parse_errors=list(parsed.parse_errors))

    for d in parsed.detections:
        result.detections.append(
            DetectionPrediction(label=d.label, bbox_xyxy=scale_box(d.bbox_xyxy)))

    for s in parsed.segmentations:
        scaled_poly = [scale_point(x, y) for x, y in s.mask_polygon]
        result.segmentations.append(
            SegmentationPrediction(
                label=s.label, bbox_xyxy=scale_box(s.bbox_xyxy),
                mask_polygon=scaled_poly))

    result.classifications = list(parsed.classifications)

    for k in parsed.keypoints:
        scaled_kps = [
            (scale_point(x, y)[0], scale_point(x, y)[1], v)
            for x, y, v in k.keypoints
        ]
        result.keypoints.append(
            KeypointPrediction(
                label=k.label, bbox_xyxy=scale_box(k.bbox_xyxy),
                keypoints=scaled_kps))

    return result

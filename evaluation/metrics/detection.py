"""Detection metrics: mAP (mean Average Precision) computation.

Computes per-class AP and mAP at various IoU thresholds for bounding
box detection, following the COCO-style evaluation protocol.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def bbox_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> float:
    """Compute IoU between two boxes in (a_min, b_min, a_max, b_max) format.

    Works with any consistent axis ordering (xyxy, yxyx, etc.) as long as
    both boxes use the same convention.
    """
    a1_1, a2_1, a1_2, a2_2 = box1
    b1_1, b2_1, b1_2, b2_2 = box2

    inter_a1 = max(a1_1, b1_1)
    inter_a2 = max(a2_1, b2_1)
    inter_b1 = min(a1_2, b1_2)
    inter_b2 = min(a2_2, b2_2)

    inter_area = max(0, inter_b1 - inter_a1) * max(0, inter_b2 - inter_a2)
    area_a = max(0, a1_2 - a1_1) * max(0, a2_2 - a2_1)
    area_b = max(0, b1_2 - b1_1) * max(0, b2_2 - b2_1)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray,
) -> float:
    """Compute AP (Area Under Precision-Recall Curve).

    Uses the 101-point interpolation method (COCO style).
    """
    # Prepend sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # 101-point interpolation
    interp_points = np.linspace(0, 1, 101)
    ap = 0.0
    for t in interp_points:
        mask = mrec >= t
        if mask.any():
            ap += mpre[mask].max()
    ap /= 101.0
    return ap


def compute_detection_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute detection metrics for a batch of images.

    Args:
        predictions: List of dicts, each with:
            - 'image_id': str
            - 'detections': List of {'label': str, 'bbox': (y1,x1,y2,x2), 'confidence': float}
        ground_truths: List of dicts, each with:
            - 'image_id': str
            - 'detections': List of {'label': str, 'bbox': (y1,x1,y2,x2)}
        iou_thresholds: IoU thresholds for matching. Default: [0.5, 0.75].

    Returns:
        Dict with mAP, per-class AP, and per-threshold metrics.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]

    # Build GT index: {image_id: {class: [boxes]}}
    gt_index: Dict[str, Dict[str, List[Tuple]]] = defaultdict(lambda: defaultdict(list))
    for gt_entry in ground_truths:
        img_id = gt_entry["image_id"]
        for det in gt_entry.get("detections", []):
            gt_index[img_id][det["label"]].append(tuple(det["bbox"]))

    # Build predictions index: {class: [(image_id, bbox, confidence)]}
    pred_by_class: Dict[str, List[Tuple]] = defaultdict(list)
    for pred_entry in predictions:
        img_id = pred_entry["image_id"]
        for det in pred_entry.get("detections", []):
            pred_by_class[det["label"]].append(
                (img_id, tuple(det["bbox"]), det.get("confidence", 1.0))
            )

    # Get all classes
    all_classes = set()
    for gt_entry in ground_truths:
        for det in gt_entry.get("detections", []):
            all_classes.add(det["label"])
    for cls in pred_by_class:
        all_classes.add(cls)

    results: Dict[str, float] = {}

    for iou_thresh in iou_thresholds:
        class_aps = []

        for cls in sorted(all_classes):
            preds = pred_by_class.get(cls, [])
            # Sort by confidence descending
            preds = sorted(preds, key=lambda x: -x[2])

            # Count total GT instances for this class
            n_gt = 0
            matched: Dict[str, set] = defaultdict(set)
            for img_id in gt_index:
                n_gt += len(gt_index[img_id].get(cls, []))

            if n_gt == 0 and not preds:
                continue

            if n_gt == 0:
                class_aps.append(0.0)
                results[f"ap_{cls}@{iou_thresh:.2f}"] = 0.0
                continue

            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))

            for i, (img_id, bbox, conf) in enumerate(preds):
                gt_boxes = gt_index.get(img_id, {}).get(cls, [])
                best_iou = 0.0
                best_j = -1

                for j, gt_box in enumerate(gt_boxes):
                    iou = bbox_iou(bbox, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

                if best_iou >= iou_thresh and best_j not in matched[img_id]:
                    tp[i] = 1
                    matched[img_id].add(best_j)
                else:
                    fp[i] = 1

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recalls = tp_cum / n_gt
            precisions = tp_cum / (tp_cum + fp_cum)

            ap = compute_ap(recalls, precisions)
            class_aps.append(ap)
            results[f"ap_{cls}@{iou_thresh:.2f}"] = ap

        mean_ap = float(np.mean(class_aps)) if class_aps else 0.0
        results[f"mAP@{iou_thresh:.2f}"] = mean_ap

    # COCO-style mAP (average over IoU 0.5:0.95:0.05)
    if set(iou_thresholds) == {0.5, 0.75}:
        coco_thresholds = np.arange(0.5, 1.0, 0.05)
        coco_aps = []
        for t in coco_thresholds:
            key = f"mAP@{t:.2f}"
            if key in results:
                coco_aps.append(results[key])
        if coco_aps:
            results["mAP@0.5:0.95"] = float(np.mean(coco_aps))

    return results


class DetectionEvaluator:
    """Accumulates detection predictions and GTs for batch evaluation."""

    def __init__(self, iou_thresholds: Optional[List[float]] = None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.75]
        self._predictions: List[Dict] = []
        self._ground_truths: List[Dict] = []

    def add_image(
        self,
        image_id: str,
        predictions: List[Dict],
        ground_truths: List[Dict],
    ) -> None:
        """Add predictions and ground truths for one image.

        Args:
            image_id: Unique image identifier.
            predictions: List of {'label': str, 'bbox': (y1,x1,y2,x2), 'confidence': float}.
            ground_truths: List of {'label': str, 'bbox': (y1,x1,y2,x2)}.
        """
        self._predictions.append({
            "image_id": image_id,
            "detections": predictions,
        })
        self._ground_truths.append({
            "image_id": image_id,
            "detections": ground_truths,
        })

    def compute(self) -> Dict[str, float]:
        """Compute detection metrics across all accumulated images."""
        return compute_detection_metrics(
            self._predictions,
            self._ground_truths,
            self.iou_thresholds,
        )

    def reset(self) -> None:
        self._predictions.clear()
        self._ground_truths.clear()

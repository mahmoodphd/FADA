"""Segmentation metrics: Dice coefficient and IoU.

Computes per-class and mean segmentation metrics from predicted
polygon masks and ground truth masks.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def polygon_to_mask(
    polygon: List[Tuple[int, int]],
    height: int,
    width: int,
) -> np.ndarray:
    """Convert polygon vertices to a binary mask.

    Args:
        polygon: List of (x, y) vertices.
        height: Image height.
        width: Image width.

    Returns:
        Binary mask of shape (height, width).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygon) < 3:
        return mask
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks.

    Dice = 2 * |pred ∩ gt| / (|pred| + |gt|)
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0  # Both empty = perfect match
    return 2.0 * intersection / total


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU (Intersection over Union) between two binary masks.

    IoU = |pred ∩ gt| / |pred ∪ gt|
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def compute_segmentation_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    image_size: Tuple[int, int],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute segmentation metrics for a single image.

    Args:
        predictions: List of dicts with keys 'label', 'polygon' [(x,y),...].
        ground_truths: List of dicts with keys 'label', 'polygon' [(x,y),...].
        image_size: (height, width) of the image.
        iou_threshold: IoU threshold for matching predictions to ground truths.

    Returns:
        Dict with keys: 'mean_dice', 'mean_iou', per-class metrics.
    """
    h, w = image_size
    metrics: Dict[str, float] = {}
    per_class_dice: Dict[str, List[float]] = defaultdict(list)
    per_class_iou: Dict[str, List[float]] = defaultdict(list)

    # Group by class
    gt_by_class: Dict[str, List[np.ndarray]] = defaultdict(list)
    for gt in ground_truths:
        mask = polygon_to_mask(gt["polygon"], h, w)
        gt_by_class[gt["label"]].append(mask)

    pred_by_class: Dict[str, List[np.ndarray]] = defaultdict(list)
    for pred in predictions:
        mask = polygon_to_mask(pred["polygon"], h, w)
        pred_by_class[pred["label"]].append(mask)

    # Compute per-class metrics
    all_classes = set(list(gt_by_class.keys()) + list(pred_by_class.keys()))

    for cls_name in all_classes:
        gt_masks = gt_by_class.get(cls_name, [])
        pred_masks = pred_by_class.get(cls_name, [])

        if not gt_masks and not pred_masks:
            continue

        # Merge all masks of same class into single mask
        gt_merged = np.zeros((h, w), dtype=np.uint8)
        for m in gt_masks:
            gt_merged = np.maximum(gt_merged, m)

        pred_merged = np.zeros((h, w), dtype=np.uint8)
        for m in pred_masks:
            pred_merged = np.maximum(pred_merged, m)

        d = dice_coefficient(pred_merged, gt_merged)
        i = iou_score(pred_merged, gt_merged)

        per_class_dice[cls_name].append(d)
        per_class_iou[cls_name].append(i)
        metrics[f"dice_{cls_name}"] = d
        metrics[f"iou_{cls_name}"] = i

    # Compute mean metrics
    all_dice = [v for vals in per_class_dice.values() for v in vals]
    all_iou = [v for vals in per_class_iou.values() for v in vals]

    metrics["mean_dice"] = float(np.mean(all_dice)) if all_dice else 0.0
    metrics["mean_iou"] = float(np.mean(all_iou)) if all_iou else 0.0

    return metrics


class SegmentationEvaluator:
    """Accumulates segmentation metrics across a dataset."""

    def __init__(self):
        self._per_class_dice: Dict[str, List[float]] = defaultdict(list)
        self._per_class_iou: Dict[str, List[float]] = defaultdict(list)
        self._count = 0

    def add_image(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        image_size: Tuple[int, int],
    ) -> Dict[str, float]:
        """Add metrics for one image."""
        metrics = compute_segmentation_metrics(
            predictions, ground_truths, image_size,
        )
        for key, val in metrics.items():
            if key.startswith("dice_"):
                cls = key[5:]
                self._per_class_dice[cls].append(val)
            elif key.startswith("iou_"):
                cls = key[4:]
                self._per_class_iou[cls].append(val)
        self._count += 1
        return metrics

    def compute(self) -> Dict[str, float]:
        """Compute aggregated metrics across all images."""
        results: Dict[str, float] = {}

        for cls, vals in self._per_class_dice.items():
            results[f"dice_{cls}"] = float(np.mean(vals))
        for cls, vals in self._per_class_iou.items():
            results[f"iou_{cls}"] = float(np.mean(vals))

        all_dice = [v for vals in self._per_class_dice.values() for v in vals]
        all_iou = [v for vals in self._per_class_iou.values() for v in vals]

        results["mean_dice"] = float(np.mean(all_dice)) if all_dice else 0.0
        results["mean_iou"] = float(np.mean(all_iou)) if all_iou else 0.0
        results["num_images"] = self._count

        return results

    def reset(self) -> None:
        self._per_class_dice.clear()
        self._per_class_iou.clear()
        self._count = 0

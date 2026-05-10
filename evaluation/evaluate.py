#!/usr/bin/env python3
"""Evaluation script for the fetal anatomy VLM.

Runs the fine-tuned Qwen3.5-VL model on the held-out test set and computes
detection (mAP), segmentation (Dice/IoU), and classification (accuracy) metrics.

Usage:
    # Smoke test (10 samples):
    python evaluate.py --checkpoint outputs_qwen35/final/ --limit 10

    # Full evaluation:
    python evaluate.py --checkpoint outputs_qwen35/final/

    # Resume interrupted run:
    python evaluate.py --checkpoint outputs_qwen35/final/ --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from infer import load_model, run_inference  # noqa: E402
from fetal_vlm_kd.metrics.output_parser import (  # noqa: E402
    COORD_SCALE,
    ParsedOutput,
    parse_model_output,
)
from fetal_vlm_kd.metrics.detection import DetectionEvaluator  # noqa: E402
from fetal_vlm_kd.metrics.segmentation import SegmentationEvaluator  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task classification
# ---------------------------------------------------------------------------

def classify_task(gt_text: str) -> str:
    """Determine the task type from the ground-truth assistant response.

    Returns one of: 'detection', 'segmentation', 'classification', 'keypoint'.
    """
    try:
        data = json.loads(gt_text)
    except json.JSONDecodeError:
        return "classification"

    items = data if isinstance(data, list) else [data]
    for item in items:
        if not isinstance(item, dict):
            continue
        if "mask" in item:
            return "segmentation"
        if "keypoints" in item:
            return "keypoint"
        if "bbox_2d" in item or "box_2d" in item:
            return "detection"
    return "classification"


def extract_dataset_name(image_path: str) -> str:
    """Extract the source dataset name from the image path."""
    marker = "/Dataset/"
    idx = image_path.find(marker)
    if idx == -1:
        return "unknown"
    rest = image_path[idx + len(marker):]
    return rest.split("/")[0]


# ---------------------------------------------------------------------------
# Bridge: ParsedOutput -> evaluator dict formats
# ---------------------------------------------------------------------------

def detections_to_eval(parsed: ParsedOutput) -> List[Dict]:
    """Convert ParsedOutput detections to DetectionEvaluator format."""
    return [
        {"label": d.label, "bbox": d.bbox_xyxy, "confidence": d.confidence}
        for d in parsed.detections
    ]


def segmentations_to_det_eval(parsed: ParsedOutput) -> List[Dict]:
    """Extract bbox-level dicts from segmentation predictions."""
    return [
        {"label": s.label, "bbox": s.bbox_xyxy, "confidence": s.confidence}
        for s in parsed.segmentations
    ]


def segmentations_to_seg_eval(parsed: ParsedOutput) -> List[Dict]:
    """Convert ParsedOutput segmentations to SegmentationEvaluator format."""
    return [
        {"label": s.label, "polygon": list(s.mask_polygon)}
        for s in parsed.segmentations
    ]


def classifications_to_label(parsed: ParsedOutput) -> Optional[str]:
    """Extract the first classification label from ParsedOutput."""
    if parsed.classifications:
        return parsed.classifications[0].label
    return None


# ---------------------------------------------------------------------------
# ClassificationEvaluator
# ---------------------------------------------------------------------------

class ClassificationEvaluator:
    """Accumulates classification predictions and computes accuracy."""

    def __init__(self):
        self._records: List[Tuple[str, str, str]] = []  # (pred, gt, dataset)

    def add_sample(self, pred_label: str, gt_label: str, dataset: str = "") -> None:
        self._records.append((pred_label, gt_label, dataset))

    def compute(self) -> Dict[str, Any]:
        if not self._records:
            return {"accuracy": 0.0, "accuracy_case_insensitive": 0.0, "num_samples": 0}

        exact = sum(1 for p, g, _ in self._records if p.strip() == g.strip())
        ci = sum(
            1 for p, g, _ in self._records
            if p.strip().lower() == g.strip().lower()
        )
        n = len(self._records)

        results: Dict[str, Any] = {
            "accuracy": exact / n,
            "accuracy_case_insensitive": ci / n,
            "num_samples": n,
        }

        # Per-class metrics
        class_correct: Dict[str, int] = defaultdict(int)
        class_total: Dict[str, int] = defaultdict(int)
        for pred, gt, _ in self._records:
            gt_norm = gt.strip().lower()
            class_total[gt_norm] += 1
            if pred.strip().lower() == gt_norm:
                class_correct[gt_norm] += 1

        per_class = {}
        for cls in sorted(class_total):
            per_class[cls] = {
                "accuracy": class_correct[cls] / class_total[cls],
                "correct": class_correct[cls],
                "total": class_total[cls],
            }
        results["per_class"] = per_class

        # Per-dataset metrics
        ds_correct: Dict[str, int] = defaultdict(int)
        ds_total: Dict[str, int] = defaultdict(int)
        for pred, gt, ds in self._records:
            ds_total[ds] += 1
            if pred.strip().lower() == gt.strip().lower():
                ds_correct[ds] += 1
        per_dataset = {}
        for ds in sorted(ds_total):
            per_dataset[ds] = {
                "accuracy": ds_correct[ds] / ds_total[ds],
                "correct": ds_correct[ds],
                "total": ds_total[ds],
            }
        results["per_dataset"] = per_dataset

        return results

    def reset(self) -> None:
        self._records.clear()


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

class EvalPipeline:
    """Runs the full evaluation pipeline on the test set."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        test_jsonl_path: str,
        output_dir: str = "eval_results",
        limit: Optional[int] = None,
        resume: bool = False,
        seed: int = 42,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        self.model = model
        self.processor = processor
        self.test_jsonl_path = Path(test_jsonl_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.resume = resume
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Global evaluators
        self.det_eval = DetectionEvaluator()
        self.seg_eval = SegmentationEvaluator()
        self.cls_eval = ClassificationEvaluator()

        # Per-dataset evaluators
        self.per_ds_det: Dict[str, DetectionEvaluator] = {}
        self.per_ds_seg: Dict[str, SegmentationEvaluator] = {}
        self.per_ds_cls: Dict[str, ClassificationEvaluator] = {}

        # Counters
        self.task_counts = defaultdict(int)
        self.errors: List[Dict] = []
        self.predictions_path = self.output_dir / "eval_predictions.jsonl"

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _parse_entry(self, raw: Dict) -> Dict[str, str]:
        """Extract image_path, prompt, gt_text from a conversation entry."""
        image_path = ""
        prompt = ""
        gt_text = ""
        for msg in raw["messages"]:
            if msg["role"] == "user":
                for c in msg["content"]:
                    if c.get("type") == "image":
                        image_path = c["image"]
                    elif c.get("type") == "text":
                        prompt = c["text"]
            elif msg["role"] == "assistant":
                for c in msg["content"]:
                    if c.get("type") == "text":
                        gt_text = c["text"]
        return {"image_path": image_path, "prompt": prompt, "gt_text": gt_text}

    def load_test_data(self) -> List[Tuple[int, Dict]]:
        """Load test entries. Returns list of (original_index, parsed_entry)."""
        entries = []
        with open(self.test_jsonl_path) as f:
            for i, line in enumerate(f):
                raw = json.loads(line)
                parsed = self._parse_entry(raw)
                entries.append((i, parsed))

        # Subsample if limit is set
        if self.limit and self.limit < len(entries):
            rng = random.Random(self.seed)
            entries = rng.sample(entries, self.limit)
            entries.sort(key=lambda x: x[0])

        # Resume: filter out already-processed indices
        if self.resume and self.predictions_path.exists():
            processed = set()
            with open(self.predictions_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        processed.add(rec["idx"])
                    except (json.JSONDecodeError, KeyError):
                        continue
            before = len(entries)
            entries = [(i, e) for i, e in entries if i not in processed]
            logger.info("Resume: skipping %d already-processed, %d remaining",
                        before - len(entries), len(entries))
            # Replay cached results into evaluators
            self._replay_cached(processed)

        return entries

    def _replay_cached(self, processed_indices: set) -> None:
        """Replay cached predictions into evaluators for resume."""
        with open(self.predictions_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") != "success":
                    continue
                task = rec.get("task", "")
                dataset = rec.get("dataset", "unknown")
                self.task_counts[task] += 1

                if task == "detection":
                    self._replay_detection(rec, dataset)
                elif task == "segmentation":
                    self._replay_segmentation(rec, dataset)
                elif task == "classification":
                    self._replay_classification(rec, dataset)

    def _replay_detection(self, rec: Dict, dataset: str) -> None:
        pred_parsed = parse_model_output(rec.get("pred_text", ""))
        gt_parsed = parse_model_output(rec.get("gt_text", ""))
        pred_dicts = detections_to_eval(pred_parsed)
        gt_dicts = detections_to_eval(gt_parsed)
        gt_dicts_no_conf = [{"label": d["label"], "bbox": d["bbox"]} for d in gt_dicts]
        img_id = str(rec["idx"])
        self.det_eval.add_image(img_id, pred_dicts, gt_dicts_no_conf)
        if dataset not in self.per_ds_det:
            self.per_ds_det[dataset] = DetectionEvaluator()
        self.per_ds_det[dataset].add_image(img_id, pred_dicts, gt_dicts_no_conf)

    def _replay_segmentation(self, rec: Dict, dataset: str) -> None:
        pred_parsed = parse_model_output(rec.get("pred_text", ""))
        gt_parsed = parse_model_output(rec.get("gt_text", ""))
        image_size = tuple(rec.get("image_size", (0, 0)))
        if image_size[0] == 0:
            return
        pred_seg = segmentations_to_seg_eval(pred_parsed)
        gt_seg = segmentations_to_seg_eval(gt_parsed)
        self.seg_eval.add_image(pred_seg, gt_seg, image_size)
        if dataset not in self.per_ds_seg:
            self.per_ds_seg[dataset] = SegmentationEvaluator()
        self.per_ds_seg[dataset].add_image(pred_seg, gt_seg, image_size)

    def _replay_classification(self, rec: Dict, dataset: str) -> None:
        pred_label = rec.get("pred_label", "")
        gt_label = rec.get("gt_label", "")
        if pred_label and gt_label:
            self.cls_eval.add_sample(pred_label, gt_label, dataset)
            if dataset not in self.per_ds_cls:
                self.per_ds_cls[dataset] = ClassificationEvaluator()
            self.per_ds_cls[dataset].add_sample(pred_label, gt_label, dataset)

    # ------------------------------------------------------------------
    # Per-sample evaluation
    # ------------------------------------------------------------------

    def evaluate_single(self, idx: int, entry: Dict) -> Dict:
        """Evaluate a single test sample."""
        image_path = entry["image_path"]
        prompt = entry["prompt"]
        gt_text = entry["gt_text"]
        dataset = extract_dataset_name(image_path)
        task = classify_task(gt_text)

        result = {
            "idx": idx,
            "image_path": image_path,
            "dataset": dataset,
            "task": task,
            "gt_text": gt_text,
            "pred_text": "",
            "status": "error",
        }

        try:
            img = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            result["error"] = f"Image load error: {e}"
            self.errors.append(result)
            return result

        # Segmentation evaluator needs an image size to rasterize polygons.
        # With normalized [0, 1000) coordinates, we use COORD_SCALE as the
        # common canvas size so both pred and GT are in the same space.
        image_size = (COORD_SCALE, COORD_SCALE)  # (height, width)
        result["image_size"] = image_size

        # Run inference
        try:
            raw_output = run_inference(
                self.model, self.processor, img, prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            result["error"] = f"Inference error: {e}"
            self.errors.append(result)
            return result

        result["pred_text"] = raw_output

        # Parse both prediction and GT
        pred_parsed = parse_model_output(raw_output)
        gt_parsed = parse_model_output(gt_text)

        if pred_parsed.parse_errors:
            result["parse_errors"] = pred_parsed.parse_errors

        # Route to evaluator
        self.task_counts[task] += 1

        if task == "detection":
            self._eval_detection(idx, pred_parsed, gt_parsed, dataset)
            result["status"] = "success"

        elif task == "segmentation":
            self._eval_segmentation(idx, pred_parsed, gt_parsed, dataset, image_size)
            result["status"] = "success"

        elif task == "classification":
            pred_label = classifications_to_label(pred_parsed)
            gt_label = classifications_to_label(gt_parsed)
            if pred_label is None:
                # Model may have returned something else; try raw text
                pred_label = raw_output.strip().strip('"').strip("'")
            if gt_label is None:
                gt_label = gt_text.strip()
            result["pred_label"] = pred_label
            result["gt_label"] = gt_label
            self.cls_eval.add_sample(pred_label, gt_label, dataset)
            if dataset not in self.per_ds_cls:
                self.per_ds_cls[dataset] = ClassificationEvaluator()
            self.per_ds_cls[dataset].add_sample(pred_label, gt_label, dataset)
            result["status"] = "success"

        elif task == "keypoint":
            # Store predictions for manual review, no evaluator
            result["pred_keypoints"] = [
                {"label": k.label, "bbox": k.bbox_xyxy, "keypoints": k.keypoints}
                for k in pred_parsed.keypoints
            ]
            result["gt_keypoints"] = [
                {"label": k.label, "bbox": k.bbox_xyxy, "keypoints": k.keypoints}
                for k in gt_parsed.keypoints
            ]
            result["status"] = "success"

        return result

    def _eval_detection(
        self, idx: int,
        pred: ParsedOutput, gt: ParsedOutput,
        dataset: str,
    ) -> None:
        pred_dicts = detections_to_eval(pred)
        gt_dicts = [{"label": d.label, "bbox": d.bbox_xyxy} for d in gt.detections]
        img_id = str(idx)

        self.det_eval.add_image(img_id, pred_dicts, gt_dicts)
        if dataset not in self.per_ds_det:
            self.per_ds_det[dataset] = DetectionEvaluator()
        self.per_ds_det[dataset].add_image(img_id, pred_dicts, gt_dicts)

    def _eval_segmentation(
        self, idx: int,
        pred: ParsedOutput, gt: ParsedOutput,
        dataset: str,
        image_size: Tuple[int, int],
    ) -> None:
        pred_seg = segmentations_to_seg_eval(pred)
        gt_seg = segmentations_to_seg_eval(gt)

        self.seg_eval.add_image(pred_seg, gt_seg, image_size)
        if dataset not in self.per_ds_seg:
            self.per_ds_seg[dataset] = SegmentationEvaluator()
        self.per_ds_seg[dataset].add_image(pred_seg, gt_seg, image_size)

        # Also evaluate bbox detection from segmentation predictions
        pred_det = segmentations_to_det_eval(pred)
        gt_det = [{"label": s.label, "bbox": s.bbox_xyxy} for s in gt.segmentations]
        img_id = f"seg_{idx}"
        self.det_eval.add_image(img_id, pred_det, gt_det)
        if dataset not in self.per_ds_det:
            self.per_ds_det[dataset] = DetectionEvaluator()
        self.per_ds_det[dataset].add_image(img_id, pred_det, gt_det)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Run the full evaluation. Returns aggregated metrics dict."""
        entries = self.load_test_data()
        total = len(entries)
        logger.info("Evaluating %d test samples", total)

        # Open predictions file (append mode for resume support)
        mode = "a" if self.resume else "w"
        pred_file = open(self.predictions_path, mode)

        start_time = time.time()

        try:
            for i, (idx, entry) in enumerate(tqdm(entries, desc="Evaluating")):
                result = self.evaluate_single(idx, entry)

                # Write immediately for resume support
                pred_file.write(json.dumps(result, default=str) + "\n")
                pred_file.flush()

                # Periodic logging
                if (i + 1) % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (total - i - 1) / rate if rate > 0 else 0
                    logger.info(
                        "Progress: %d/%d (%.1f%%) | %.1f samples/min | ETA: %.0fm | Errors: %d",
                        i + 1, total, 100 * (i + 1) / total,
                        rate * 60, eta / 60, len(self.errors),
                    )
        finally:
            pred_file.close()

        elapsed = time.time() - start_time
        metrics = self.compute_all_metrics(elapsed)
        self.save_results(metrics)
        self.print_summary(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Metrics aggregation
    # ------------------------------------------------------------------

    def compute_all_metrics(self, elapsed: float = 0) -> Dict:
        """Aggregate all metrics into a single results dict."""
        results: Dict[str, Any] = {
            "summary": {
                "total_samples": sum(self.task_counts.values()),
                "detection_samples": self.task_counts.get("detection", 0),
                "segmentation_samples": self.task_counts.get("segmentation", 0),
                "classification_samples": self.task_counts.get("classification", 0),
                "keypoint_samples": self.task_counts.get("keypoint", 0),
                "errors": len(self.errors),
                "elapsed_seconds": round(elapsed, 1),
            },
        }

        # Detection
        det_metrics = self.det_eval.compute() if self.task_counts.get("detection", 0) or self.task_counts.get("segmentation", 0) else {}
        det_per_ds = {}
        for ds, ev in self.per_ds_det.items():
            det_per_ds[ds] = ev.compute()
        results["detection"] = {"global": det_metrics, "per_dataset": det_per_ds}

        # Segmentation
        seg_metrics = self.seg_eval.compute() if self.task_counts.get("segmentation", 0) else {}
        seg_per_ds = {}
        for ds, ev in self.per_ds_seg.items():
            seg_per_ds[ds] = ev.compute()
        results["segmentation"] = {"global": seg_metrics, "per_dataset": seg_per_ds}

        # Classification
        cls_metrics = self.cls_eval.compute() if self.task_counts.get("classification", 0) else {}
        cls_per_ds = {}
        for ds, ev in self.per_ds_cls.items():
            cls_per_ds[ds] = ev.compute()
        results["classification"] = {"global": cls_metrics, "per_dataset": cls_per_ds}

        return results

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_results(self, metrics: Dict) -> None:
        """Save metrics to JSON file."""
        results_path = self.output_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info("Saved metrics to %s", results_path)

        # Save summary text
        summary_path = self.output_dir / "eval_summary.txt"
        with open(summary_path, "w") as f:
            f.write(self._format_summary(metrics))
        logger.info("Saved summary to %s", summary_path)

    def print_summary(self, metrics: Dict) -> None:
        """Print the summary table to console."""
        print(self._format_summary(metrics))

    def _format_summary(self, metrics: Dict) -> str:
        """Build a human-readable summary string."""
        s = metrics.get("summary", {})
        lines = [
            "",
            "=" * 60,
            "  Fetal VLM Evaluation Results",
            "=" * 60,
            f"  Total samples:    {s.get('total_samples', 0)}",
            f"  Detection:        {s.get('detection_samples', 0)}",
            f"  Segmentation:     {s.get('segmentation_samples', 0)}",
            f"  Classification:   {s.get('classification_samples', 0)}",
            f"  Keypoint:         {s.get('keypoint_samples', 0)}",
            f"  Errors:           {s.get('errors', 0)}",
            f"  Time:             {s.get('elapsed_seconds', 0):.0f}s",
            "",
        ]

        # Detection
        det = metrics.get("detection", {}).get("global", {})
        if det:
            lines.append("  DETECTION")
            lines.append("  " + "-" * 40)
            lines.append(f"  mAP@0.50:         {det.get('mAP@0.50', 0):.4f}")
            lines.append(f"  mAP@0.75:         {det.get('mAP@0.75', 0):.4f}")
            # Per-class AP@0.50
            per_cls = {k: v for k, v in det.items() if k.startswith("ap_") and k.endswith("@0.50")}
            if per_cls:
                lines.append("  Per-class AP@0.50:")
                for k in sorted(per_cls):
                    cls_name = k.replace("ap_", "").replace("@0.50", "")
                    lines.append(f"    {cls_name:30s} {per_cls[k]:.4f}")
            lines.append("")

        # Segmentation
        seg = metrics.get("segmentation", {}).get("global", {})
        if seg:
            lines.append("  SEGMENTATION")
            lines.append("  " + "-" * 40)
            lines.append(f"  Mean Dice:        {seg.get('mean_dice', 0):.4f}")
            lines.append(f"  Mean IoU:         {seg.get('mean_iou', 0):.4f}")
            per_cls_dice = {k: v for k, v in seg.items() if k.startswith("dice_")}
            per_cls_iou = {k.replace("dice_", "iou_"): seg.get(k.replace("dice_", "iou_"), 0) for k in per_cls_dice}
            if per_cls_dice:
                lines.append("  Per-class:")
                for k in sorted(per_cls_dice):
                    cls_name = k.replace("dice_", "")
                    d_val = per_cls_dice[k]
                    i_val = seg.get(f"iou_{cls_name}", 0)
                    lines.append(f"    {cls_name:30s} Dice={d_val:.4f}  IoU={i_val:.4f}")
            lines.append("")

        # Classification
        cls_m = metrics.get("classification", {}).get("global", {})
        if cls_m:
            lines.append("  CLASSIFICATION")
            lines.append("  " + "-" * 40)
            lines.append(f"  Accuracy (exact):     {cls_m.get('accuracy', 0):.4f}")
            lines.append(f"  Accuracy (case-ins):  {cls_m.get('accuracy_case_insensitive', 0):.4f}")
            per_ds = cls_m.get("per_dataset", {})
            if per_ds:
                lines.append("  Per-dataset:")
                for ds in sorted(per_ds):
                    acc = per_ds[ds].get("accuracy", 0)
                    n = per_ds[ds].get("total", 0)
                    lines.append(f"    {ds:40s} {acc:.4f} ({n} samples)")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the fetal anatomy VLM on the test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(PROJECT_ROOT / "outputs_qwen35" / "final"),
        help="Path to LoRA adapter checkpoint directory",
    )
    parser.add_argument(
        "--model-name", type=str, default="unsloth/Qwen3.5-4B",
        help="Base model identifier",
    )
    parser.add_argument(
        "--test-data", type=str,
        default=str(PROJECT_ROOT / "processed_data" / "test.jsonl"),
        help="Path to test JSONL file",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "eval_results"),
        help="Directory for evaluation results",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only N samples")
    parser.add_argument("--resume", action="store_true", help="Resume from partial progress")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    parser.add_argument("--load-in-4bit", action="store_true", help="4-bit quantization")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Loading model from %s", args.checkpoint)
    model, processor = load_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
    )

    pipeline = EvalPipeline(
        model=model,
        processor=processor,
        test_jsonl_path=args.test_data,
        output_dir=args.output_dir,
        limit=args.limit,
        resume=args.resume,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    metrics = pipeline.run()

    # Exit with error code if too many failures
    error_rate = len(pipeline.errors) / max(1, sum(pipeline.task_counts.values()))
    if error_rate > 0.1:
        logger.warning("High error rate: %.1f%%", error_rate * 100)
        sys.exit(1)


if __name__ == "__main__":
    main()

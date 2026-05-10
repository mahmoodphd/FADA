#!/usr/bin/env python3
"""Standalone inference script for fetal anatomy detection.

Usage examples:

    # Detection with default prompt:
    python infer.py --image path/to/fetal.png

    # Detection for specific structures:
    python infer.py --image path/to/fetal.png --prompt "Detect the femur and skull"

    # Use a specific checkpoint:
    python infer.py --image path/to/fetal.png --checkpoint outputs_qwen35/checkpoint-1500

    # Visualize results (save annotated image):
    python infer.py --image path/to/fetal.png --visualize --output result.png

    # Batch inference on a directory:
    python infer.py --image-dir path/to/images/ --output-dir results/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from fetal_vlm_kd.metrics.output_parser import (
    ParsedOutput,
    parse_model_output,
    rescale_predictions_to_original,
)

logger = logging.getLogger(__name__)


# ---- Default prompts --------------------------------------------------------
# Coordinates use Qwen VL native convention: xyxy normalized to [0, 1000)

DEFAULT_DETECT_PROMPT = (
    "Detect all anatomical structures visible in this fetal ultrasound image. "
    "For each detection, return a JSON array of objects with \"bbox_2d\" as "
    "[x_min, y_min, x_max, y_max] in normalized 0-1000 coordinates "
    "and \"label\" as the structure name."
)


def build_detect_prompt(classes: str) -> str:
    """Build a detection prompt targeting specific anatomy classes."""
    return (
        f"Detect all instances of the following anatomical structures in this "
        f"fetal ultrasound image: {classes}. "
        f"For each detection, return a JSON object with \"bbox_2d\" as "
        f"[x_min, y_min, x_max, y_max] in normalized 0-1000 coordinates "
        f"and \"label\" as the structure name."
    )


def build_segment_prompt(classes: str) -> str:
    """Build a segmentation prompt."""
    return (
        f"Detect and segment all instances of the following anatomical structures "
        f"in this fetal ultrasound image: {classes}. "
        f"For each instance, return a JSON object with \"bbox_2d\" as "
        f"[x_min, y_min, x_max, y_max] in normalized 0-1000 coordinates, "
        f"\"label\" as the structure name, "
        f"and \"mask\" as a list of [x, y] polygon vertices in normalized "
        f"0-1000 coordinates."
    )


# ---- Model loading -----------------------------------------------------------

def load_model(
    checkpoint_path: Optional[str] = None,
    model_name: str = "unsloth/Qwen3.5-4B",
    max_seq_length: int = 4096,
    load_in_4bit: bool = False,
) -> Tuple[Any, Any]:
    """Load the fine-tuned Qwen VL model.

    Args:
        checkpoint_path: Path to a training checkpoint directory
                         (e.g. outputs_qwen35/checkpoint-500).
                         If None, loads the base model without fine-tuning.
        model_name: Base model identifier for Unsloth.
        max_seq_length: Maximum sequence length.
        load_in_4bit: Whether to use 4-bit quantization.

    Returns:
        (model, processor) tuple.
    """
    from unsloth import FastVisionModel

    if checkpoint_path and Path(checkpoint_path).is_dir():
        logger.info("Loading fine-tuned model from checkpoint: %s", checkpoint_path)
        model, processor = FastVisionModel.from_pretrained(
            checkpoint_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
    else:
        if checkpoint_path:
            logger.warning(
                "Checkpoint path '%s' not found, loading base model", checkpoint_path
            )
        logger.info("Loading base model: %s", model_name)
        model, processor = FastVisionModel.from_pretrained(
            model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )

    FastVisionModel.for_inference(model)
    logger.info("Model loaded and set to inference mode")
    return model, processor


# ---- Inference ---------------------------------------------------------------

def _clean_generation(text: str) -> str:
    """Extract the first valid response from potentially looping output.

    Some models trained on mixed data (with interpretation prompts) may
    generate repetitive multi-turn patterns like:
        answer\nassistant\n<think>\n\n</think>\n\nanswer\nassistant\n...
    This function extracts only the first response.
    """
    import re
    # Strip any leading <think>...</think> block (from chat template leak)
    text = re.sub(r'^<think>\s*</think>\s*', '', text)

    # Cut at the first repetition boundary: "\nassistant\n" signals a new turn
    idx = text.find('\nassistant\n')
    if idx != -1:
        text = text[:idx]

    # Also handle "assistant\n" at the very start (shouldn't happen, but safe)
    if text.startswith('assistant\n'):
        text = text[len('assistant\n'):]

    return text.strip()


def run_inference(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.95,
) -> str:
    """Run a single inference on one image.

    Args:
        model: The Qwen VL model.
        processor: The Qwen VL processor/tokenizer.
        image: PIL Image to analyze.
        prompt: Text prompt for the model.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (low = deterministic).
        top_p: Nucleus sampling threshold.

    Returns:
        Raw text output from the model.
    """
    # Build the chat message in Qwen VL format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply chat template to get input_ids
    input_text = processor.apply_chat_template(
        messages, add_generation_prompt=True
    )
    inputs = processor(
        text=input_text,
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    # Resolve EOS token ID(s) -- ensure <|im_end|> is always included
    eos_ids = set()
    if hasattr(processor, 'tokenizer'):
        tok = processor.tokenizer
    else:
        tok = processor
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    # Explicitly add <|im_end|> in case it differs from eos_token_id
    im_end_ids = tok.encode('<|im_end|>', add_special_tokens=False)
    if len(im_end_ids) == 1:
        eos_ids.add(im_end_ids[0])
    eos_token_id = list(eos_ids) if len(eos_ids) > 1 else list(eos_ids)[0]

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            use_cache=True,
            eos_token_id=eos_token_id,
            stop_strings=["\nassistant\n", "\nassistant"],
            tokenizer=tok,
        )

    # Decode only the generated tokens (skip input)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()

    # Clean up any residual repetition patterns
    text = _clean_generation(text)

    return text


def predict(
    model: Any,
    processor: Any,
    image_path: str,
    prompt: Optional[str] = None,
    classes: Optional[str] = None,
    task: str = "detect",
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """Full prediction pipeline: load image -> inference -> parse -> rescale.

    Args:
        model: Loaded Qwen VL model.
        processor: Loaded Qwen VL processor.
        image_path: Path to the fetal ultrasound image.
        prompt: Custom prompt. If None, auto-generated based on task/classes.
        classes: Comma-separated anatomy classes (e.g. "femur, skull, abdomen").
        task: Task type - "detect" or "segment".
        max_new_tokens: Max generation tokens.
        temperature: Sampling temperature.

    Returns:
        Dict with keys: predictions, raw_text, image_path, original_wh
    """
    img = Image.open(image_path).convert("RGB")
    original_wh = img.size  # (width, height)

    # Build prompt
    if prompt is None:
        if classes:
            if task == "segment":
                prompt = build_segment_prompt(classes)
            else:
                prompt = build_detect_prompt(classes)
        else:
            prompt = DEFAULT_DETECT_PROMPT

    # Run inference
    raw_text = run_inference(
        model, processor, img, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    # Parse model output (coordinates in [0, 1000) normalized space)
    parsed = parse_model_output(raw_text)

    # Rescale coordinates from [0, 1000) back to original pixel space
    rescaled = rescale_predictions_to_original(parsed, original_wh)

    return {
        "predictions": rescaled,
        "raw_text": raw_text,
        "image_path": str(image_path),
        "original_wh": original_wh,
        "prompt": prompt,
    }


# ---- Visualization ----------------------------------------------------------

def visualize(
    image_path: str,
    parsed: ParsedOutput,
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[Any]:
    """Draw predictions on the image and optionally save/show.

    Args:
        image_path: Path to the original image.
        parsed: ParsedOutput with coordinates in original pixel space (xyxy).
        output_path: If given, save annotated image here.
        show: If True, display with matplotlib.

    Returns:
        matplotlib Figure if show or output_path is specified, else None.
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np

    img = np.array(Image.open(image_path).convert("RGB"))
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(img)

    # Color palette
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
        "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
    ]
    label_colors: Dict[str, str] = {}
    color_idx = 0

    # Draw detections
    for det in parsed.detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        if det.label not in label_colors:
            label_colors[det.label] = colors[color_idx % len(colors)]
            color_idx += 1
        c = label_colors[det.label]

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=c, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5, det.label,
            fontsize=10, color="white", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=c, alpha=0.8),
        )

    # Draw segmentations
    for seg in parsed.segmentations:
        x1, y1, x2, y2 = seg.bbox_xyxy
        if seg.label not in label_colors:
            label_colors[seg.label] = colors[color_idx % len(colors)]
            color_idx += 1
        c = label_colors[seg.label]

        # Draw bbox
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=c, facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)

        # Draw polygon mask
        if seg.mask_polygon:
            poly_pts = np.array(seg.mask_polygon)
            polygon = patches.Polygon(
                poly_pts, closed=True,
                linewidth=2, edgecolor=c, facecolor=c, alpha=0.2,
            )
            ax.add_patch(polygon)

        ax.text(
            x1, y1 - 5, seg.label,
            fontsize=10, color="white", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=c, alpha=0.8),
        )

    # Classifications shown as text overlay
    for cls_pred in parsed.classifications:
        ax.text(
            10, 30, f"Class: {cls_pred.label}",
            fontsize=14, color="white", weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#333", alpha=0.8),
        )

    ax.axis("off")
    ax.set_title("Fetal Anatomy Detection", fontsize=14, weight="bold")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved visualization to %s", output_path)
    if show:
        plt.show()

    return fig


# ---- JSON output helpers -----------------------------------------------------

def predictions_to_json(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert prediction result to a JSON-serializable dict."""
    parsed: ParsedOutput = result["predictions"]
    output: Dict[str, Any] = {
        "image_path": result["image_path"],
        "original_size": {"width": result["original_wh"][0], "height": result["original_wh"][1]},
        "prompt": result["prompt"],
        "raw_model_output": result["raw_text"],
        "detections": [],
        "segmentations": [],
        "classifications": [],
    }

    for det in parsed.detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        output["detections"].append({
            "label": det.label,
            "bbox_xyxy": [x1, y1, x2, y2],
        })

    for seg in parsed.segmentations:
        x1, y1, x2, y2 = seg.bbox_xyxy
        output["segmentations"].append({
            "label": seg.label,
            "bbox_xyxy": [x1, y1, x2, y2],
            "mask_polygon": [[x, y] for x, y in seg.mask_polygon],
        })

    for cls_pred in parsed.classifications:
        output["classifications"].append({"label": cls_pred.label})

    if parsed.parse_errors:
        output["parse_errors"] = parsed.parse_errors

    return output


# ---- CLI entry point ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetal anatomy detection with fine-tuned Qwen VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", type=str,
        help="Path to a single fetal ultrasound image",
    )
    input_group.add_argument(
        "--image-dir", type=str,
        help="Directory of images for batch inference",
    )

    # Model
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to fine-tuned checkpoint dir (e.g. outputs_qwen35/checkpoint-1500)",
    )
    parser.add_argument(
        "--model-name", type=str, default="unsloth/Qwen3.5-4B",
        help="Base model identifier (default: unsloth/Qwen3.5-4B)",
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load model in 4-bit quantization (saves VRAM)",
    )

    # Prompt
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Custom prompt (overrides --classes and --task)",
    )
    parser.add_argument(
        "--classes", type=str, default=None,
        help="Comma-separated anatomy classes (e.g. 'femur, skull, abdomen')",
    )
    parser.add_argument(
        "--task", choices=["detect", "segment"], default="detect",
        help="Task type (default: detect)",
    )

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)

    # Output
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save annotated image (single image mode)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save results (batch mode)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate visualization with bounding boxes",
    )
    parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Print JSON results to stdout",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load model
    model, processor = load_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
    )

    # Collect image paths
    if args.image:
        image_paths = [args.image]
    else:
        img_dir = Path(args.image_dir)
        image_paths = sorted(
            str(p) for p in img_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        )
        if not image_paths:
            logger.error("No images found in %s", args.image_dir)
            sys.exit(1)
        logger.info("Found %d images in %s", len(image_paths), args.image_dir)

    all_results = []
    for img_path in image_paths:
        logger.info("Processing: %s", img_path)
        result = predict(
            model, processor,
            image_path=img_path,
            prompt=args.prompt,
            classes=args.classes,
            task=args.task,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        parsed: ParsedOutput = result["predictions"]
        n_det = len(parsed.detections)
        n_seg = len(parsed.segmentations)
        n_cls = len(parsed.classifications)
        logger.info(
            "  Results: %d detections, %d segmentations, %d classifications",
            n_det, n_seg, n_cls,
        )

        if parsed.parse_errors:
            for err in parsed.parse_errors:
                logger.warning("  Parse error: %s", err)

        # Print detections to console
        for det in parsed.detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            print(f"  [{det.label}] bbox(xyxy): ({x1}, {y1}, {x2}, {y2})")

        for seg in parsed.segmentations:
            x1, y1, x2, y2 = seg.bbox_xyxy
            n_pts = len(seg.mask_polygon)
            print(f"  [{seg.label}] bbox(xyxy): ({x1}, {y1}, {x2}, {y2}) mask: {n_pts} pts")

        for cls_pred in parsed.classifications:
            print(f"  [classification] {cls_pred.label}")

        json_result = predictions_to_json(result)
        all_results.append(json_result)

        # Visualization
        if args.visualize:
            if args.output:
                vis_path = args.output
            elif args.output_dir:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                vis_path = str(out_dir / (Path(img_path).stem + "_pred.png"))
            else:
                vis_path = str(Path(img_path).with_suffix(".pred.png"))

            visualize(img_path, parsed, output_path=vis_path)

    # JSON output
    if args.output_json:
        if len(all_results) == 1:
            print(json.dumps(all_results[0], indent=2))
        else:
            print(json.dumps(all_results, indent=2))

    # Save JSON for batch mode
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "predictions.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Saved predictions JSON to %s", json_path)


if __name__ == "__main__":
    main()

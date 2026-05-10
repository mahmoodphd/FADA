"""GPU-decorated inference functions for the FADA HuggingFace Space.

Contains all @spaces.GPU decorated functions that require GPU access.
On ZeroGPU, the model is loaded lazily on first call and cached.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image

try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

import model_loader
from class_mapper import (
    _extract_cls_label,
    map_interpretation_to_classes,
    parse_interpretation_json,
)
from constants import (
    CLASSIFY_PROMPT,
    DEFAULT_DETECT_CLASSES,
    INTERPRET_PROMPT,
    MAX_NEW_TOKENS,
    TEMPERATURE,
)
from intent_parser import IntentResult, parse_intent
from output_parser import parse_model_output, rescale_predictions_to_original
from visualizer import draw_annotations

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builders (from infer.py)
# ---------------------------------------------------------------------------

def build_detect_prompt(classes: str) -> str:
    return (
        f"Detect all instances of the following anatomical structures in this "
        f"fetal ultrasound image: {classes}. "
        f'For each detection, return a JSON object with "bbox_2d" as '
        f"[x_min, y_min, x_max, y_max] in normalized 0-1000 coordinates "
        f'and "label" as the structure name.'
    )


def build_segment_prompt(classes: str) -> str:
    return (
        f"Detect and segment all instances of the following anatomical structures "
        f"in this fetal ultrasound image: {classes}. "
        f'For each instance, return a JSON object with "bbox_2d" as '
        f"[x_min, y_min, x_max, y_max] in normalized 0-1000 coordinates, "
        f'"label" as the structure name, '
        f'and "mask" as a list of [x, y] polygon vertices in normalized '
        f"0-1000 coordinates."
    )


def build_keypoint_prompt(classes: str) -> str:
    return (
        f"Detect measurement keypoints for the following structures in this "
        f"fetal ultrasound image: {classes}. "
        f'For each structure, return a JSON object with "bbox_2d" as '
        f"[x_min, y_min, x_max, y_max] in normalized 0-1000 coordinates, "
        f'"label" as the structure name, '
        f'and "keypoints" as a list of [x, y, visibility] triplets with '
        f"x, y in normalized 0-1000 coordinates."
    )


# Keypoint class names -- when detected in classes string, use keypoint prompt
_KEYPOINT_CLASSES = {"CRL_KP", "NTKpoints", "ScaleBarKpoints"}


def _has_keypoint_classes(classes: str) -> bool:
    """Check if the classes string contains keypoint-specific class names."""
    return any(kc in classes for kc in _KEYPOINT_CLASSES)


# ---------------------------------------------------------------------------
# Text cleanup (from infer.py)
# ---------------------------------------------------------------------------

def _clean_generation(text: str) -> str:
    """Extract the first valid response from potentially looping output."""
    text = re.sub(r'^<think>\s*</think>\s*', '', text)
    idx = text.find('\nassistant\n')
    if idx != -1:
        text = text[:idx]
    if text.startswith('assistant\n'):
        text = text[len('assistant\n'):]
    return text.strip()


# ---------------------------------------------------------------------------
# Core inference (from infer.py, adapted for PIL images)
# ---------------------------------------------------------------------------

def _run_inference(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """Run a single inference on one image. Must be called with GPU access."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    input_text = processor.apply_chat_template(
        messages, add_generation_prompt=True)
    inputs = processor(
        text=input_text, images=[image], return_tensors="pt",
    ).to(model.device)

    # Resolve EOS token IDs
    eos_ids = set()
    tok = getattr(processor, 'tokenizer', processor)
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    im_end_ids = tok.encode('<|im_end|>', add_special_tokens=False)
    if len(im_end_ids) == 1:
        eos_ids.add(im_end_ids[0])
    eos_token_id = list(eos_ids) if len(eos_ids) > 1 else list(eos_ids)[0]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=temperature > 0,
            use_cache=True,
            eos_token_id=eos_token_id,
            stop_strings=["\nassistant\n", "\nassistant"],
            tokenizer=tok,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0].strip()

    return _clean_generation(text)


# ---------------------------------------------------------------------------
# Format interpretation for chat display
# ---------------------------------------------------------------------------

def _format_interpretation(interp: Dict[str, Any]) -> str:
    """Format 8-field interpretation dict as readable markdown."""
    if not interp.get("_parse_success", False):
        raw = interp.get("_raw_text", "")
        return f"**Interpretation (raw):**\n```\n{raw[:1000]}\n```"

    fields = [
        ("Anatomical Structures", "anatomical_structures"),
        ("Fetal Orientation", "fetal_orientation"),
        ("Imaging Plane", "imaging_plane"),
        ("Biometric Measurements", "biometric_measurements"),
        ("Gestational Age", "gestational_age"),
        ("Image Quality", "image_quality"),
        ("Normality Assessment", "normality_assessment"),
        ("Clinical Recommendations", "clinical_recommendations"),
    ]

    parts = ["**Clinical Interpretation:**\n"]
    for title, key in fields:
        value = interp.get(key, "N/A")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, indent=2)
        parts.append(f"**{title}:** {value}\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Safe annotation helper
# ---------------------------------------------------------------------------

def _safe_annotate(image: Image.Image, raw_text: str, task_name: str):
    """Parse model output and draw annotations, with error handling.

    Returns (response_text, annotated_image_or_None).
    """
    try:
        parsed = parse_model_output(raw_text)
        rescaled = rescale_predictions_to_original(parsed, image.size)
        annotated = draw_annotations(image, rescaled)
    except Exception as e:
        logger.error("Annotation error in %s: %s", task_name, e, exc_info=True)
        return (
            f"**{task_name} completed** but annotation rendering failed. "
            f"Raw model output:\n```\n{raw_text[:500]}\n```"
        ), None

    n_det = len(rescaled.detections)
    n_seg = len(rescaled.segmentations)
    n_kp = len(rescaled.keypoints)

    all_labels = set()
    for d in rescaled.detections:
        all_labels.add(d.label)
    for s in rescaled.segmentations:
        all_labels.add(s.label)
    for k in rescaled.keypoints:
        all_labels.add(k.label)
    labels_str = ", ".join(sorted(all_labels)) or "none"

    parts = []
    if n_det > 0:
        parts.append(f"{n_det} detections")
    if n_seg > 0:
        parts.append(f"{n_seg} masks")
    if n_kp > 0:
        parts.append(f"{n_kp} keypoints")

    if not parts:
        return (
            f"**{task_name}:** No structures found in this image for the "
            f"requested classes. The model may not detect these structures "
            f"on this particular view."
        ), None

    count_str = ", ".join(parts)
    response = f"**{task_name} results** ({count_str}: {labels_str})"
    if rescaled.parse_errors:
        response += f"\n_Parse warnings: {len(rescaled.parse_errors)}_"
    return response, annotated


# ---------------------------------------------------------------------------
# Chat turn handler
# ---------------------------------------------------------------------------

def _run_chat_turn_impl(
    image: Optional[Image.Image],
    user_text: str,
    state: Dict[str, Any],
    temperature: float = TEMPERATURE,
) -> Tuple[str, Optional[Image.Image], Dict[str, Any]]:
    """Process one chat turn. Called from GPU-decorated wrapper."""
    model, processor = model_loader.get_model_and_processor()

    # If new image provided, update state
    if image is not None:
        image = image.convert("RGB")
        # Resize if too large
        max_side = 1024
        w, h = image.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        state["image"] = image
        state["image_wh"] = image.size
        state["interpretation"] = None
        state["classification"] = None
        state["det_classes"] = None
        state["seg_classes"] = None

    current_image = state.get("image")
    if current_image is None:
        return "Please upload a fetal ultrasound image first.", None, state

    # Parse intent
    intent = parse_intent(
        user_text,
        has_image=(current_image is not None),
        has_interpretation=(state.get("interpretation") is not None),
    )

    if intent.task == "greeting":
        return (
            "Welcome to FADA! Upload a fetal ultrasound image and I'll provide "
            "clinical interpretation. You can then ask me to detect or segment "
            "specific anatomical structures.",
            None, state,
        )

    if intent.task == "general":
        return (
            "I can help with fetal ultrasound analysis. Try:\n"
            "- Upload an image and ask me to **interpret** it\n"
            "- Ask me to **detect** specific structures (e.g., 'detect brain')\n"
            "- Ask me to **segment** anatomy (e.g., 'segment cardiac')",
            None, state,
        )

    if intent.task == "interpret":
        raw = _run_inference(model, processor, current_image, INTERPRET_PROMPT,
                             temperature=temperature)
        interp = parse_interpretation_json(raw)
        state["interpretation"] = interp

        # Also run classification
        raw_cls = _run_inference(model, processor, current_image, CLASSIFY_PROMPT,
                                 max_new_tokens=256, temperature=temperature)
        cls_result = parse_interpretation_json(raw_cls)
        cls_label = _extract_cls_label(cls_result)
        state["classification"] = cls_label

        # Derive default classes from interpretation
        if interp.get("_parse_success"):
            det_cls, seg_cls, tier = map_interpretation_to_classes(interp, cls_label)
            state["det_classes"] = det_cls
            state["seg_classes"] = seg_cls

        response = _format_interpretation(interp)
        if cls_label:
            response += f"\n**Classification:** {cls_label}"
        response += (
            "\n\n---\nYou can now ask me to **detect** or **segment** specific "
            "structures. For example: 'detect brain structures' or 'segment cardiac'."
        )
        return response, None, state

    if intent.task == "classify":
        raw_cls = _run_inference(model, processor, current_image, CLASSIFY_PROMPT,
                                 max_new_tokens=256, temperature=temperature)
        cls_result = parse_interpretation_json(raw_cls)
        cls_label = _extract_cls_label(cls_result)
        state["classification"] = cls_label
        return f"**Classification:** {cls_label or 'Unknown'}", None, state

    if intent.task == "detect":
        classes = intent.classes or state.get("det_classes") or DEFAULT_DETECT_CLASSES
        if _has_keypoint_classes(classes):
            prompt = build_keypoint_prompt(classes)
        else:
            prompt = build_detect_prompt(classes)
        raw = _run_inference(model, processor, current_image, prompt,
                             temperature=temperature)
        response, annotated = _safe_annotate(current_image, raw, "Detection")
        return response, annotated, state

    if intent.task == "segment":
        classes = (intent.seg_classes or intent.classes
                   or state.get("seg_classes") or state.get("det_classes")
                   or DEFAULT_DETECT_CLASSES)
        prompt = build_segment_prompt(classes)
        raw = _run_inference(model, processor, current_image, prompt,
                             temperature=temperature)
        response, annotated = _safe_annotate(current_image, raw, "Segmentation")
        return response, annotated, state

    return "I didn't understand that request. Please try again.", None, state


# ---------------------------------------------------------------------------
# Autonomous pipeline
# ---------------------------------------------------------------------------

def _run_autonomous_impl(
    image: Image.Image,
    temperature: float = TEMPERATURE,
) -> Dict[str, Any]:
    """Run the full 5-phase autonomous pipeline. Called from GPU-decorated wrapper."""
    model, processor = model_loader.get_model_and_processor()

    image = image.convert("RGB")
    max_side = 1024
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    results: Dict[str, Any] = {"timings": {}}
    t0 = time.time()

    # Phase 1: Interpret
    raw_interp = _run_inference(model, processor, image, INTERPRET_PROMPT,
                                temperature=temperature)
    interp = parse_interpretation_json(raw_interp)
    results["interpretation"] = interp
    results["timings"]["interpret"] = round(time.time() - t0, 1)

    # Phase 2: Classify
    t1 = time.time()
    raw_cls = _run_inference(model, processor, image, CLASSIFY_PROMPT,
                             max_new_tokens=256, temperature=temperature)
    cls_result = parse_interpretation_json(raw_cls)
    cls_label = _extract_cls_label(cls_result)
    results["classification"] = cls_label or "Unknown"
    results["timings"]["classify"] = round(time.time() - t1, 1)

    # Phase 3: Map
    if interp.get("_parse_success"):
        det_classes, seg_classes, tier = map_interpretation_to_classes(interp, cls_label)
    else:
        det_classes = DEFAULT_DETECT_CLASSES
        seg_classes = None
        tier = "fallback_parse_fail"
    results["mapping"] = {
        "det_classes": det_classes,
        "seg_classes": seg_classes,
        "tier": tier,
    }

    # Phase 4: Detect
    t2 = time.time()
    try:
        raw_det = _run_inference(model, processor, image,
                                 build_detect_prompt(det_classes),
                                 temperature=temperature)
        parsed_det = parse_model_output(raw_det)
        rescaled_det = rescale_predictions_to_original(parsed_det, image.size)
        det_image = draw_annotations(image, rescaled_det)
        results["detection_image"] = det_image
        results["detection_count"] = len(rescaled_det.detections)
    except Exception as e:
        logger.error("Detection phase error: %s", e, exc_info=True)
        results["detection_image"] = None
        results["detection_count"] = 0
    results["timings"]["detect"] = round(time.time() - t2, 1)

    # Phase 5: Segment (conditional)
    seg_image = None
    if seg_classes:
        t3 = time.time()
        try:
            raw_seg = _run_inference(model, processor, image,
                                     build_segment_prompt(seg_classes),
                                     temperature=temperature)
            parsed_seg = parse_model_output(raw_seg)
            rescaled_seg = rescale_predictions_to_original(parsed_seg, image.size)
            seg_image = draw_annotations(image, rescaled_seg)
            results["segmentation_count"] = len(rescaled_seg.segmentations)
        except Exception as e:
            logger.error("Segmentation phase error: %s", e, exc_info=True)
            results["segmentation_count"] = 0
        results["timings"]["segment"] = round(time.time() - t3, 1)
    else:
        results["segmentation_count"] = 0
        results["seg_skipped"] = True

    results["segmentation_image"] = seg_image
    results["total_time"] = round(time.time() - t0, 1)

    return results


# ---------------------------------------------------------------------------
# GPU wrappers -- adapts to ZeroGPU vs persistent GPU automatically
# ---------------------------------------------------------------------------
# ZeroGPU (zero-a10g): @spaces.GPU allocates GPU per-call with quota limits
# Persistent GPU (a10g-small, t4-small): GPU always available, no decorator needed

import os as _os
_HARDWARE = _os.environ.get('SPACE_HARDWARE', '')
_IS_ZEROGPU = _HARDWARE.startswith('zero-')

if HAS_SPACES and _IS_ZEROGPU:
    logger.info('ZeroGPU detected (%s), using @spaces.GPU decorators', _HARDWARE)

    @spaces.GPU(duration=90)
    def run_chat_turn(
        image: Optional[Image.Image],
        user_text: str,
        state: Dict[str, Any],
        temperature: float = TEMPERATURE,
    ) -> Tuple[str, Optional[Image.Image], Dict[str, Any]]:
        """Process one chat turn (ZeroGPU)."""
        return _run_chat_turn_impl(image, user_text, state, temperature)

    @spaces.GPU(duration=120)
    def run_autonomous_pipeline(
        image: Image.Image,
        temperature: float = TEMPERATURE,
    ) -> Dict[str, Any]:
        """Run full 5-phase pipeline (ZeroGPU)."""
        return _run_autonomous_impl(image, temperature)
else:
    logger.info('Persistent GPU or local (%s), no @spaces.GPU needed', _HARDWARE or 'local')

    def run_chat_turn(
        image: Optional[Image.Image],
        user_text: str,
        state: Dict[str, Any],
        temperature: float = TEMPERATURE,
    ) -> Tuple[str, Optional[Image.Image], Dict[str, Any]]:
        """Process one chat turn (persistent GPU / local)."""
        return _run_chat_turn_impl(image, user_text, state, temperature)

    def run_autonomous_pipeline(
        image: Image.Image,
        temperature: float = TEMPERATURE,
    ) -> Dict[str, Any]:
        """Run full 5-phase pipeline (persistent GPU / local)."""
        return _run_autonomous_impl(image, temperature)

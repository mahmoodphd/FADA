"""FADA: Fetal Anatomy Delineation and Analysis - HuggingFace Space.

Interactive demo with four tabs:
  Tab 1: Chat (Human-in-the-Loop) -- upload, interpret, then detect/segment
  Tab 2: Autonomous Analysis -- one-click full 5-phase pipeline
  Tab 3: Anatomy Reference -- what the model can do, abbreviation guide
  Tab 4: About -- project information, framework, and acknowledgments
"""
from __future__ import annotations

import glob
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

from inference import run_autonomous_pipeline, run_chat_turn

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def _make_empty_state() -> Dict[str, Any]:
    return {
        "image": None,
        "image_path": None,
        "image_wh": None,
        "interpretation": None,
        "classification": None,
        "det_classes": None,
        "seg_classes": None,
    }


def chat_respond(
    message: Dict[str, Any],
    history: List[Dict[str, Any]],
    state: Dict[str, Any],
    temperature: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Handle a chat message (text + optional image)."""
    user_text = message.get("text", "").strip()
    files = message.get("files", [])

    uploaded_image = None
    uploaded_path = None
    if files:
        try:
            fpath = files[0] if isinstance(files[0], str) else files[0].get("path", files[0])
            uploaded_image = Image.open(fpath).convert("RGB")
            uploaded_path = fpath
        except Exception as e:
            logger.warning("Failed to open uploaded file: %s", e)

    # Build user message text
    user_msg = user_text or "Interpret this image"
    # Show filename so sonographer can identify the image in validation
    if uploaded_path:
        fname = os.path.basename(uploaded_path)
        user_msg = f"**[{fname}]** {user_msg}"
    history.append({"role": "user", "content": user_msg})

    # If user uploaded an image, show it in the conversation
    if uploaded_path:
        history.append({"role": "user", "content": gr.Image(uploaded_path)})
        state["image_path"] = uploaded_path

    # Run inference
    try:
        response_text, annotated_image, state = run_chat_turn(
            uploaded_image, user_text, state, temperature)
    except Exception as e:
        logger.error("Inference error: %s", e, exc_info=True)
        response_text = f"An error occurred: {e}"
        annotated_image = None

    # Build assistant response
    if annotated_image is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        annotated_image.save(tmp.name)
        # Send annotated image
        history.append({"role": "assistant", "content": gr.Image(tmp.name)})
        # Then send text description
        if response_text:
            history.append({"role": "assistant", "content": response_text})
    else:
        # For interpretation: show the original image back alongside the text
        # so the user can reference it for follow-up detect/segment requests
        if state.get("image_path") and state.get("interpretation") is not None:
            history.append({"role": "assistant", "content": gr.Image(state["image_path"])})
        history.append({"role": "assistant", "content": response_text or "No response generated."})

    return history, state


# ---------------------------------------------------------------------------
# Autonomous pipeline handler
# ---------------------------------------------------------------------------

def autonomous_analyze(
    image: Optional[Image.Image],
    temperature: float,
) -> Tuple[str, str, Optional[Image.Image], Optional[Image.Image], str]:
    """Run the full 5-phase autonomous pipeline."""
    if image is None:
        return "Please upload an image first.", "", None, None, ""

    try:
        results = run_autonomous_pipeline(image, temperature)
    except Exception as e:
        logger.error("Pipeline error: %s", e, exc_info=True)
        return f"Error: {e}", "", None, None, ""

    # Format interpretation as readable markdown
    interp = results.get("interpretation", {})
    if interp.get("_parse_success", False):
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
        interp_parts = []
        for title, key in fields:
            val = interp.get(key, "N/A")
            if isinstance(val, (dict, list)):
                import json
                val = json.dumps(val, indent=2)
            interp_parts.append(f"**{title}:** {val}")
        interp_text = "\n\n".join(interp_parts)
    else:
        raw = interp.get("_raw_text", str(interp))
        interp_text = f"```\n{raw[:2000]}\n```"

    # Classification + mapping info
    cls_label = results.get("classification", "Unknown")
    mapping = results.get("mapping", {})
    mapping_text = (
        f"**Classification:** {cls_label}\n\n"
        f"**Detection classes:** {mapping.get('det_classes', 'N/A')}\n\n"
        f"**Segmentation classes:** {mapping.get('seg_classes', 'N/A')}\n\n"
        f"**Mapping tier:** {mapping.get('tier', 'N/A')}"
    )

    det_image = results.get("detection_image")
    seg_image = results.get("segmentation_image")

    # Timing info
    timings = results.get("timings", {})
    timing_parts = [f"{k}: {v}s" for k, v in timings.items()]
    total = results.get("total_time", 0)
    debug_text = (
        f"Phase timings: {', '.join(timing_parts)}\n"
        f"Total: {total}s\n"
        f"Detections: {results.get('detection_count', 0)}\n"
        f"Segmentations: {results.get('segmentation_count', 0)}"
    )
    if results.get("seg_skipped"):
        debug_text += "\n(Segmentation skipped: no applicable classes for this view)"

    return interp_text, mapping_text, det_image, seg_image, debug_text


# ---------------------------------------------------------------------------
# Content strings
# ---------------------------------------------------------------------------

PROJECT_BRIEF_MD = """\
**FADA** is a unified vision-language model (VLM) built on **Qwen3.5-VL 4B** with **LoRA** fine-tuning
and **offline selective knowledge distillation** from four domain-specific ultrasound foundation models.
Unlike existing single-task approaches, FADA performs clinical interpretation, anatomical detection,
and segmentation within a single conversational interface.

**Key capabilities:**
- **Clinical Interpretation** -- structured 8-field report (anatomical structures, imaging plane, biometrics, gestational age, normality, recommendations)
- **Anatomical Detection** -- bounding-box localization of 32+ fetal anatomy classes
- **Segmentation** -- polygon-mask delineation of target structures
- **Keypoint Localization** -- landmark points for CRL measurement and scale bar calibration
- **Interpret-First Pipeline** -- 5-phase cascade: Interpret, Classify, Map, Detect, Segment
"""

TEACHER_MD = """\
### Offline Knowledge Distillation

FADA-SKD (Selective KD) distills complementary features from four pretrained ultrasound foundation models.
All distillation is offline -- teacher features are precomputed to HDF5, requiring no teacher inference at deployment.

| Teacher | Modality | Weight | Reference |
|---------|----------|--------|-----------|
| **FetalCLIP** | CLIP embeddings (1024-d) | 0.40 | Maani et al., *Medical Image Analysis*, 2025 |
| **UltraSAM** | SAM encoder features (256-d) | 0.25 | Meyer et al., *IJCARS*, 2025 |
| **USF-MAE** | MAE latent features (768-d) | 0.20 | Megahed et al., *arXiv:2510.22990*, 2025 |
| **UltraFedFM** | Federated features (768-d) | 0.15 | Jiang et al., *npj Digital Medicine*, 2025 |
"""

DATASET_MD = """\
### Training & Evaluation Data

- **Annotation Dataset:** 7 publicly available sources, 14 anatomy categories, **4,478 held-out test samples**
- **Interpretation Dataset:** 56,805 multi-turn conversations (37,870 full 8-field + 18,935 alternative format)
- The example images below are randomly sampled from the actual test set (one per category)
"""

FUNDING_MD = """\
### Acknowledgments

This work has been funded by the **Canadian International Development Research Centre (IDRC)**
under Grant Agreement 110060-001, managed by the Global Health Institute at the American University
of Beirut through the **Global Health and Artificial Intelligence Network in MENA (GHAIN MENA)**.
It is part of a larger program of research on responsible AI for development, supported by IDRC
and the UK government's Foreign, Commonwealth and Development Office.

This publication was also funded by the **PPM 7th Cycle grant (PPM 07-0409-240041, AMAL-For-Qatar)**
from the **Qatar Research Development and Innovation Council (QRDI)**, a member of the Qatar Foundation.

The views expressed herein do not necessarily reflect those of the UK government's Foreign,
Commonwealth and Development Office, IDRC, IDRC's Board of Governors, or QRDI.
"""

DISCLAIMER_MD = """\
<div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 12px; margin-top: 10px;">
<strong>Research Prototype -- Not for Clinical Use</strong><br>
FADA is a research tool for fetal ultrasound analysis. It is not FDA-approved
and should not be used for clinical decision-making. Always consult a qualified
healthcare professional for medical advice.
</div>
"""

# ---------------------------------------------------------------------------
# Anatomy reference guide (from annotation_directory.md)
# ---------------------------------------------------------------------------

ANATOMY_REF_ABBREV_MD = """\
## Abbreviation Quick Reference

| Abbrev. | Full Name | Annotation Type | View Context |
|---------|-----------|-----------------|--------------|
| AB | Abdomen | Detection | CRL, NT |
| B | Body | Detection | CRL |
| Brain | Brain (whole structure) | Detection + Segment | BPD / Trans-thalamic |
| C | Chest (Thorax) | Detection | CRL, NT |
| CRL | Crown-Rump Length | Detection | CRL |
| CSP | Cavum Septum Pellucidum | Detection | BPD / Trans-thalamic |
| DP | Diencephalon | Detection | CRL, NT |
| G | Genital Tubercle | Detection | CRL |
| H | Head | Detection | CRL, NT |
| LV | Lateral Ventricle | Detection | CRL, NT, BPD |
| MDS | Mandible Symphysis | Detection | CRL, NT |
| MLS | Mandible Lower Surface | Detection | CRL, NT |
| MX | Maxilla | Detection | CRL, NT |
| NB | Nasal Bone | Detection | CRL, NT |
| NT | Nuchal Translucency | Detection + Segment | NT |
| NTAPS | Nasal Tip and Pre-nasal Skin | Detection | CRL, NT |
| RBP | Rhombencephalon (Hindbrain) | Detection | CRL, NT |
| cardiac | Heart | Detection + Segment | Echocardiography |
| thorax | Thorax / Chest cavity | Detection + Segment | Echocardiography |
| artery | Umbilical Artery | Detection + Segment | Doppler |
| vein | Umbilical Vein | Detection + Segment | Doppler |
| liver | Liver | Detection + Segment | Doppler |
| stomach | Stomach | Detection + Segment | Doppler |
| abdomen | Abdomen (full-word) | Detection | Body / Pose |
| arm | Arm / upper limb | Detection | Body / Pose |
| head | Head (full-word) | Detection | Body / Pose |
| legs | Legs / lower limbs | Detection | Body / Pose |
| fetal_head | Fetal Head | Detection + Segment | Pelvimetry |
| pubic_symphysis | Pubic Symphysis | Detection + Segment | Pelvimetry |
| CRL_KP | CRL Keypoints | Keypoints | CRL measurement |
| NTKpoints | NT Keypoints | Keypoints | NT measurement |
| ScaleBar | Scale Bar | Detection | Calibration |
| ScaleBarKpoints | Scale Bar Keypoints | Keypoints | Calibration |
"""

ANATOMY_REF_GROUPS_MD = """\
## Co-occurrence Groups

Classes within each group are always detected/segmented together.
**Do not mix classes across groups in a single request.**

| Group | Detection Classes | Segmentation | Example Prompt |
|-------|------------------|--------------|----------------|
| **Brain** | Brain, CSP, LV | Brain only (CSP/LV = detect only) | *"detect brain"*, *"detect CSP"* |
| **Cardiac** | cardiac, thorax | cardiac, thorax | *"detect cardiac"*, *"segment cardiac"* |
| **NT / Nasal** | NT, nasal_bone, nasal_skin, nasal_tip | NT | *"detect nuchal translucency"* |
| **CRL** | B, CRL, H, NB | -- | *"detect CRL"*, *"detect crown-rump"* |
| **CRL + Keypoints** | CRL_KP, ScaleBarKpoints | -- | *"detect CRL keypoints"*, *"detect scalebar"* |
| **NT + Keypoints** | NTKpoints, ScaleBarKpoints | -- | *"detect NT keypoints"*, *"detect NT kp"* |
| **Doppler** | artery, liver, stomach, vein | artery, liver, stomach, vein | *"detect vessels"*, *"detect doppler"* |
| **Pelvimetry** | fetal_head, pubic_symphysis | fetal_head, pubic_symphysis | *"detect fetal head"*, *"detect symphysis"* |
| **Body / Pose** | abdomen, arm, head, legs | -- | *"detect body"*, *"detect abdomen"* |
| **Femur** | legs | -- | *"detect femur"* |
"""

ANATOMY_REF_VIEWS_MD = """\
## Anatomical Views & What to Request

### First Trimester (CRL/NT)
- **CRL view** (mid-sagittal): *"detect CRL"* -- finds Body, CRL, Head, Nasal Bone
- **NT view** (magnified head/neck): *"detect nuchal translucency"* -- finds NT, nasal structures
- **CRL Keypoints**: *"detect CRL keypoints"* -- CRL measurement endpoints + scale bar calibration
- **NT Keypoints**: *"detect NT keypoints"* -- NT measurement landmark points + scale bar calibration
- **Important:** Use the correct keypoint type for the view. If the image shows an NT measurement, use *"detect NT keypoints"*, not *"detect CRL keypoints"*. Using the wrong keypoint type may return mislabeled points (e.g., CRL_KP placed where NTKpoints should be).

### Second Trimester (Brain / BPD)
- **Trans-thalamic**: *"detect brain"* or *"detect CSP"* -- Brain, CSP, LV; segment Brain contour
- **Trans-cerebellar**: *"detect brain"* -- same brain group (cerebellum visible in posterior fossa)
- **Trans-ventricular**: *"detect brain"* -- same brain group (LV level)
- **Note:** The model may describe trans-cerebellar views as "trans-thalamic" or trans-ventricular views as "sagittal" in interpretation text. Brain **annotation** (detect/segment) uses the same group (Brain, CSP, LV) for all axial brain planes. Only CSP and LV have detection (bounding box) -- they do NOT have segmentation. Only "Brain" has segmentation contours.

### Cardiac / Echocardiography
- **Four-chamber**, **Aortic arch**, **V sign / X sign** views
- *"detect cardiac"* or *"segment cardiac"* -- finds cardiac, thorax

### Abdominal Structures / Doppler / Flow
- **Abdominal organs** (stomach, liver, vessels): *"detect stomach"* or *"detect doppler"* -- artery, vein, liver, stomach
- **Important:** Do NOT use *"detect abdomen"* for abdominal organ views. The "abdomen" label refers to the fetal body region from the pose dataset (returns abdomen, arm, head, legs as body parts). For abdominal organs like stomach or liver, use *"detect stomach"* or *"detect doppler"*.

### Body / Pose (FPUS23)
- *"detect body"* or *"detect abdomen"* -- abdomen, arm, head, legs (fetal body parts, NOT internal organs)

### Pelvimetry / Cervix
- *"detect fetal head"* or *"detect symphysis"* -- fetal_head, pubic_symphysis

### Not Yet Supported
The following structures are **not available** for detection/segmentation in the current model:
- **Femur bone** (individual): *"detect femur"* maps to "legs" (full leg bounding box, not the femur bone)
- **Placenta**: not trained
- **Lungs** (individual): thoracic views use cardiac + thorax labels, not individual lung outlines
- **Aorta** (standalone organ): "Aorta" is a view classification mapping to cardiac + thorax, not a standalone label
- **Cervical length**: cervical views map to pelvimetry group (fetal_head + pubic_symphysis)
- **Femoral length**: FL views map to "legs" bounding box only
"""

ANATOMY_REF_TASKS_MD = """\
## Task Types

| Task | What It Does | How to Request |
|------|-------------|----------------|
| **Interpret** | Structured 8-field clinical report | *"interpret this image"* |
| **Detect** | Bounding boxes around structures | *"detect brain"*, *"find CSP"*, *"detect fetal head"* |
| **Segment** | Polygon mask contours | *"segment cardiac"*, *"outline brain"* |
| **Classify** | Image-level view classification | *"classify this view"* |
| **Keypoints** | Landmark coordinate points | *"detect CRL keypoints"*, *"detect NT keypoints"*, *"detect scalebar"* |
"""

ANATOMY_REF_TIPS_MD = """\
## Tips & Input Tolerance

**Typo Tolerance:** FADA handles common misspellings automatically:

| What You Type | What FADA Understands |
|---------------|----------------------|
| "cardic", "cariac" | cardiac |
| "symphsis", "symphis" | symphysis |
| "nuchel", "translucen" | nuchal translucency |
| "thoric", "thorcic" | thoracic |
| "abdomin" | abdomen (body region) |
| "femor" | femur -> maps to legs |
| "pelvimtry" | pelvimetry |
| "stomach", "gastric" | stomach (abdominal organ) |

**Keypoint Detection:** When you request keypoints (e.g., *"detect CRL keypoints"* or
*"detect NT keypoints"*), the system automatically uses a specialized measurement
keypoint prompt format -- no special syntax needed. **Match the keypoint type to the
view:** use CRL keypoints for CRL views and NT keypoints for NT views.

**Common Prompt Mistakes:**
- *"detect abdomen"* on an abdominal organ view -- use *"detect stomach"* or *"detect doppler"* instead
- *"detect CRL keypoints"* on an NT view -- use *"detect NT keypoints"* instead
- *"segment LV"* or *"segment CSP"* -- only Brain has segmentation; LV and CSP are detection-only
- *"detect femur"* -- returns "legs" bounding box (whole leg), not the femur bone

**Recommended Workflow:**
1. Start with *"interpret this image"* to get a structured clinical report
2. Based on the interpretation, ask to *"detect"* or *"segment"* specific structures
3. The system remembers the uploaded image, so you can make multiple requests

**Co-occurrence:** Some structures are always detected together (e.g., Brain + CSP + LV).
This is by design -- they were trained as a group. See the Co-occurrence Groups table above.
"""

# ---------------------------------------------------------------------------
# Example image lists -- SAME images in both Chat and Autonomous tabs
# (from actual 4,478 test set, one per category + keypoint examples)
# ---------------------------------------------------------------------------

ALL_EXAMPLES = [
    "examples/head_transthalamic.png",
    "examples/head_transcerebellum.png",
    "examples/head_transventricular.png",
    "examples/head_diverse.png",
    "examples/cardiac_aorta.jpg",
    "examples/cardiac_flows.jpg",
    "examples/cardiac_vsign.jpg",
    "examples/cardiac_xsign.jpg",
    "examples/cardiac_other.jpg",
    "examples/biometric_planes.png",
    "examples/abdominal_structures.png",
    "examples/pubic_symphysis.png",
    "examples/crl_nt_screening.jpg",
    "examples/crl_nt_scalebar_kp.jpg",
    "examples/crl_nt_crl_keypoints.jpg",
    "examples/crl_nt_detection.png",
    "examples/nt_keypoints.png",
    "examples/nt_scalebar_keypoints.png",
    "examples/first_trimester_h1.jpg",
    "examples/first_trimester_h2.jpg",
    "examples/first_trimester_h3.jpg",
    "examples/general_ultrasound.png",
]

# Chat examples: same image set, wrapped for MultimodalTextbox format
CHAT_EXAMPLES = [
    [{"text": "Interpret this fetal ultrasound image", "files": [img]}]
    for img in ALL_EXAMPLES
]

# Autonomous examples: just image paths
AUTO_EXAMPLES = [[img] for img in ALL_EXAMPLES]

GALLERY_ITEMS = [
    ("examples/head_transthalamic.png", "Head Trans-thalamic (150)"),
    ("examples/head_transcerebellum.png", "Head Trans-cerebellum (58)"),
    ("examples/head_transventricular.png", "Head Trans-ventricular (60)"),
    ("examples/head_diverse.png", "Head Diverse (110)"),
    ("examples/cardiac_aorta.jpg", "Cardiac - Aorta (133)"),
    ("examples/cardiac_flows.jpg", "Cardiac - Flows (219)"),
    ("examples/cardiac_vsign.jpg", "Cardiac - V sign (259)"),
    ("examples/cardiac_xsign.jpg", "Cardiac - X sign (79)"),
    ("examples/cardiac_other.jpg", "Cardiac - Other (689)"),
    ("examples/biometric_planes.png", "Biometric Planes / FPUS23 (1481)"),
    ("examples/abdominal_structures.png", "Abdominal Structures (172)"),
    ("examples/pubic_symphysis.png", "Pubic Symphysis (404)"),
    ("examples/crl_nt_screening.jpg", "CRL/NT Screening (360)"),
    ("examples/crl_nt_scalebar_kp.jpg", "CRL/NT - ScaleBar Keypoints"),
    ("examples/crl_nt_crl_keypoints.jpg", "CRL/NT - CRL Keypoints"),
    ("examples/crl_nt_detection.png", "CRL/NT - NT Detection"),
    ("examples/nt_keypoints.png", "NT Keypoints (NTKpoints)"),
    ("examples/nt_scalebar_keypoints.png", "NT + ScaleBar Keypoints"),
    ("examples/first_trimester_h1.jpg", "1st Trimester H1 (76)"),
    ("examples/first_trimester_h2.jpg", "1st Trimester H2 (81)"),
    ("examples/first_trimester_h3.jpg", "1st Trimester H3 (47)"),
    ("examples/general_ultrasound.png", "General US - FOCUS (100)"),
]

# ---------------------------------------------------------------------------
# External test images (real-world ultrasound samples)
# ---------------------------------------------------------------------------

EXTERNAL_EXAMPLES = sorted(glob.glob("examples/external/*.JPG"))

EXTERNAL_CHAT_EXAMPLES = [
    [{"text": "Interpret this fetal ultrasound image", "files": [img]}]
    for img in EXTERNAL_EXAMPLES
]

EXTERNAL_AUTO_EXAMPLES = [[img] for img in EXTERNAL_EXAMPLES]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """\
.header-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px; padding: 24px; margin-bottom: 16px;
}
.header-banner h1 { color: #fff !important; text-align: center; margin-bottom: 4px; }
.header-banner h3 { color: #ccc !important; text-align: center; font-weight: normal; margin-top: 0; }
.header-banner p  { color: #aaa !important; text-align: center; font-size: 0.9em; }
.about-section { max-width: 920px; margin: 0 auto; }
.ref-section { max-width: 960px; margin: 0 auto; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="FADA - Fetal Anatomy Delineation and Analysis",
    theme=gr.themes.Soft(),
    css=CSS,
) as demo:

    # ---- Header banner ----
    gr.HTML(
        '<div class="header-banner">'
        '<h1>FADA: Fetal Anatomy Delineation and Analysis</h1>'
        '<h3>A Unified Vision-Language System for Fetal Ultrasound '
        'Interpretation, Detection, and Segmentation</h3>'
        '<p><em>Submitted to NPJ Digital Medicine</em> &nbsp;|&nbsp; '
        'Qwen3.5-VL 4B + LoRA + Offline Selective Knowledge Distillation</p>'
        '</div>'
    )

    with gr.Tabs():
        # ==============================================================
        # Tab 1: Interactive Chat
        # ==============================================================
        with gr.TabItem("Interactive Chat"):
            gr.Markdown(
                "**How to use:** Upload a fetal ultrasound image and ask me to "
                "**interpret** it. After interpretation, you will see both the "
                "interpretation and the image. You can then ask to **detect** "
                "or **segment** specific anatomy.\n\n"
                "**Example queries:** *'interpret this image'*, *'detect brain'*, "
                "*'detect CSP'*, *'segment cardiac'*, *'detect fetal head'*, "
                "*'detect CRL keypoints'*, *'detect NT keypoints'*, *'detect scalebar'*\n\n"
                "See the **Anatomy Reference** tab for all supported structures.\n\n"
                "*Each request takes ~20-60s (ZeroGPU allocates GPU per request). "
                "Interpretation takes longest (~30-40s) as it generates a detailed report.*"
            )

            state = gr.State(_make_empty_state)

            chatbot = gr.Chatbot(
                type="messages",
                height=550,
                label="FADA Chat",
                show_copy_button=True,
            )

            chat_input = gr.MultimodalTextbox(
                placeholder="Upload an image and type 'interpret this', or ask to 'detect brain'...",
                file_types=["image"],
                show_label=False,
            )

            with gr.Accordion("Settings", open=False):
                chat_temp = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.1, step=0.05,
                    label="Temperature",
                    info="Lower = more deterministic. Default 0.1.",
                )

            chat_input.submit(
                fn=chat_respond,
                inputs=[chat_input, chatbot, state, chat_temp],
                outputs=[chatbot, state],
            ).then(lambda: None, outputs=chat_input)

            gr.Examples(
                examples=CHAT_EXAMPLES,
                inputs=chat_input,
                label="Test Set Examples -- 22 images from 4,478 test samples (click to load)",
            )

            gr.Examples(
                examples=EXTERNAL_CHAT_EXAMPLES,
                inputs=chat_input,
                label="External Test Images -- 27 real-world ultrasound samples (click to load)",
                examples_per_page=6,
            )

        # ==============================================================
        # Tab 2: Autonomous Analysis
        # ==============================================================
        with gr.TabItem("Autonomous Analysis"):
            gr.Markdown(
                "**One-click analysis:** Upload an image and run the full "
                "5-phase pipeline: **Interpret** -> **Classify** -> **Map** "
                "-> **Detect** -> **Segment**.\n\n"
                "*First run may take ~30-40s for model loading. Full pipeline takes ~40-90s.*"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    auto_image = gr.Image(
                        type="pil", label="Upload Fetal Ultrasound",
                        height=300,
                    )
                    with gr.Row():
                        auto_temp = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.1, step=0.05,
                            label="Temperature",
                        )
                    auto_btn = gr.Button(
                        "Run Full Analysis",
                        variant="primary", size="lg",
                    )

                with gr.Column(scale=2):
                    auto_interp = gr.Markdown(
                        label="Clinical Interpretation",
                        value="*Upload an image and click 'Run Full Analysis'*",
                    )
                    auto_mapping = gr.Markdown(
                        label="Classification & Mapping",
                        value="",
                    )

            with gr.Row():
                auto_det_img = gr.Image(
                    type="pil", label="Detection Results", height=400)
                auto_seg_img = gr.Image(
                    type="pil", label="Segmentation Results", height=400)

            auto_debug = gr.Textbox(label="Pipeline Info", lines=3)

            auto_btn.click(
                fn=autonomous_analyze,
                inputs=[auto_image, auto_temp],
                outputs=[auto_interp, auto_mapping,
                         auto_det_img, auto_seg_img, auto_debug],
            )

            gr.Examples(
                examples=AUTO_EXAMPLES,
                inputs=auto_image,
                label="Test Set Images -- 22 samples from 4,478 test images across 17 categories",
            )

            gr.Examples(
                examples=EXTERNAL_AUTO_EXAMPLES,
                inputs=auto_image,
                label="External Test Images -- 27 real-world ultrasound samples (click to load)",
                examples_per_page=6,
            )

        # ==============================================================
        # Tab 3: Anatomy Reference
        # ==============================================================
        with gr.TabItem("Anatomy Reference"):
            with gr.Column(elem_classes="ref-section"):
                gr.Markdown(
                    "# What FADA Can Do\n\n"
                    "This reference guide describes all anatomical structures, "
                    "abbreviations, and task types supported by the FADA model. "
                    "Use it to understand what to ask the model and what the "
                    "output labels mean."
                )

                gr.Markdown(ANATOMY_REF_TASKS_MD)

                with gr.Accordion("Abbreviation Quick Reference (all labels)", open=True):
                    gr.Markdown(ANATOMY_REF_ABBREV_MD)

                with gr.Accordion("Co-occurrence Groups (which classes go together)", open=True):
                    gr.Markdown(ANATOMY_REF_GROUPS_MD)

                with gr.Accordion("Anatomical Views & What to Request", open=False):
                    gr.Markdown(ANATOMY_REF_VIEWS_MD)

                with gr.Accordion("Tips & Input Tolerance", open=False):
                    gr.Markdown(ANATOMY_REF_TIPS_MD)

        # ==============================================================
        # Tab 4: About
        # ==============================================================
        with gr.TabItem("About"):
            with gr.Column(elem_classes="about-section"):
                gr.Markdown(PROJECT_BRIEF_MD)

                gr.Markdown("### System Architecture")
                gr.Image(
                    value="examples/fada_framework.png",
                    label="FADA Framework",
                    show_label=False,
                    show_download_button=False,
                    interactive=False,
                    height=420,
                )

                gr.Markdown(TEACHER_MD)
                gr.Markdown(DATASET_MD)

                gr.Markdown("### Test Set Gallery (22 samples from 17 categories, 4,478 images)")
                gr.Gallery(
                    value=GALLERY_ITEMS,
                    label="Random samples from the held-out test set",
                    columns=5,
                    height=360,
                    object_fit="contain",
                )

                gr.Markdown(FUNDING_MD)

    gr.Markdown(DISCLAIMER_MD)

demo.queue()

if __name__ == "__main__":
    demo.launch()

<div align="center">

# FADA: Accessible fetal ultrasound interpretation and annotation with a selectively distilled unified vision-language model.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-npj_Digital_Medicine-green.svg)](#citation)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace_Spaces-yellow.svg)](https://huggingface.co/spaces/mshz88/fada-ultrasound-vlm)
[![Model](https://img.shields.io/badge/Model-HuggingFace-orange.svg)](https://huggingface.co/mshz88/FADA-SKD-4B)
[![Mobile](https://img.shields.io/badge/Mobile-GGUF_Model-purple.svg)](https://huggingface.co/mshz88/FADA-Mobile-GGUF)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blue.svg)](https://doi.org/10.5281/zenodo.20104811)

</div>

---

## Demo Videos

### Web Application Demo

<div align="center">

[![Watch the FADA Web Demo](https://img.youtube.com/vi/CbXcz74fn6k/maxresdefault.jpg)](https://www.youtube.com/watch?v=CbXcz74fn6k)

**Click the image above to watch the web app demo on YouTube**

</div>

The video demonstrates FADA's three main interaction modes:
- **Interactive Chat** -- Natural-language queries for interpretation, detection, keypoint localization, and segmentation
- **Autonomous Mode** -- One-click full 5-phase pipeline (Interpret, Classify, Map, Detect, Segment)
- **Anatomy Reference** -- Built-in visual atlas of 14 anatomical planes and 33 detectable structures

### Mobile App Demo (Offline Edge Deployment)

<div align="center">

[![Watch the FADA Mobile Demo](https://img.youtube.com/vi/RoogJqPNZ4w/maxresdefault.jpg)](https://www.youtube.com/watch?v=RoogJqPNZ4w)

**Click the image above to watch the mobile app demo on YouTube**

</div>

The mobile demo showcases fully offline AI-assisted fetal ultrasound analysis on a commodity smartphone:
- **Model Download** -- One-time 712 MB download (516 MB text + 195 MB vision encoder)
- **Chat Mode** -- Interactive interpretation and detection with ~40s per task
- **Detection Overlay** -- On-device bounding-box rendering for CRL, head, body, nasal bone
- **Autonomous Pipeline** -- Full 5-phase analysis in ~59 seconds without cloud connectivity
- **Test Device** -- Honor 90 (Snapdragon 7 Gen 1, 12 GB RAM, Android 15)

---

## Overview

**FADA** (Fetal Anatomy Delineation and Analysis) is a unified vision-language model (VLM) built on Qwen3.5-VL that performs clinical interpretation, anatomical classification, bounding-box detection, and polygon segmentation of fetal ultrasound images within a single end-to-end pipeline. FADA employs **Selective Knowledge Distillation (SKD)** to transfer task-specific expertise from four domain-specific ultrasound foundation models into a compact student while preserving clinical reasoning capabilities.

A key finding is that applying feature-level distillation *only* to annotation data (detection, segmentation, classification) while training interpretation with supervised fine-tuning alone outperforms full distillation across all tasks. Expert sonographer validation across 237 images and 49 clinical cases confirms clinically acceptable performance. The system is designed for deployment in resource-constrained obstetric settings, aligned with UN Sustainable Development Goals 3 and 10.

**Edge Deployment:** The compressed 0.8B model variant is quantized to GGUF format and deployed via llama.cpp on a commodity Android smartphone (Honor 90, Snapdragon 7 Gen 1, 12 GB RAM), completing the full 5-phase pipeline in ~59 seconds entirely offline. This demonstrates that the model can be integrated with portable fetal ultrasound devices in a stand-alone fashion.

---

## System Architecture

<div align="center">

![FADA-SKD System Workflow](figures/WRKFLOW_Final.png)

</div>

**Figure 1.** Complete FADA-SKD system lifecycle. (A) Data collection and curation: 56,805 interpretation conversations across 14 categories plus annotation data from 7 repositories covering 33 structures. (B) Teacher ensemble with offline HDF5 feature caching (453K vectors). (C) Selective Knowledge Distillation training with Qwen3.5-VL (4B) student. (D) 5-phase inference pipeline with autonomous and human-in-the-loop modes. (E) Expert sonographer validation. (F) Open deployment across cloud and mobile platforms. (G) Explainability analysis via attention heatmaps and token attribution.

---

## Key Results

### Quantitative Evaluation (4,478 test samples)

| Model | mAP@0.50 | mAP@0.75 | Dice | IoU | Cls Acc |
|:------|:--------:|:--------:|:----:|:---:|:-------:|
| **FADA-Base (4B)** | 0.7798 | 0.4211 | 0.8813 | 0.8133 | 0.8225 |
| **FADA-SKD (4B)** | 0.7671 | 0.4402 | **0.8820** | **0.8149** | **0.8379** |
| **FADA-FKD (4B)** | 0.7695 | **0.4576** | 0.8790 | 0.8114 | 0.8296 |
| **FADA-Base (0.8B)** | 0.6885 | 0.3817 | 0.8625 | 0.7899 | 0.8375 |
| **FADA-SKD (0.8B)** | 0.6744 | 0.3756 | 0.8662 | 0.7921 | 0.8433 |

### Expert Sonographer Validation

| Evaluation Mode | Cases | Interpretation | Annotation | Overall |
|:----------------|:-----:|:--------------:|:----------:|:-------:|
| **Autonomous** | 237 images | 1.924 | 2.025 | 1.975 |
| **Human-in-the-Loop** | 49 cases | 1.286 (73.5% perfect) | 1.449 (63.3% perfect) | 1.368 |

> Scoring: 1 = clinically acceptable (no correction needed), 2 = minor issues, 3 = major errors. Lower is better.

<details>
<summary><strong>Validation Methodology (click to expand)</strong></summary>

**Evaluator:** An expert sonographer with >10 years of clinical experience in obstetric ultrasound imaging.

**Autonomous Evaluation (237 images):**
- 237 images spanning 18 anatomical categories (175 from the test set + 62 external clinical images from a different hospital)
- All three 4B model variants (FADA-Base, FADA-SKD, FADA-FKD) were evaluated
- **Blinded protocol:** For each image, all three model outputs were presented in randomized order without model identification
- The sonographer independently scored each output on two dimensions:
  - **Annotation quality** (1-3): Accuracy of bounding boxes and segmentation masks
  - **Interpretation quality** (1-3): Clinical correctness and completeness of the 8-field JSON output
- Score definitions: 1 = clinically acceptable (no correction needed), 2 = minor issues (acceptable with caveats), 3 = major errors (clinically unacceptable)

**Human-in-the-Loop Evaluation (49 cases):**
- Conducted using the deployed web application ([huggingface.co/spaces/mshz88/fada-ultrasound-vlm](https://huggingface.co/spaces/mshz88/fada-ultrasound-vlm))
- The sonographer processed 49 clinical cases (22 from test set + 27 external) in realistic clinical workflow
- In HiL mode, the clinician reviews the model's interpretation first, then selects which subsequent analysis phases to execute (detect, segment)
- The same 1-3 scoring scale was applied independently to interpretation and annotation outputs
- Raw scoring data: [`docs/external_validation_scoring_v2.csv`](docs/external_validation_scoring_v2.csv) in this repository

**Verification:** All reported metrics are computed directly from the raw CSV scoring data. The 49-case HiL results yield: Interpretation mean=1.286 (73.5% score 1, 24.5% score 2, 2.0% score 3), Annotation mean=1.449 (63.3% score 1, 28.6% score 2, 8.2% score 3). These values match Paper Section 4.2 (Table 3) exactly.

**Paper reference:** Section 3.6 "Evaluation Protocol" and Section 4.1-4.2 "Expert Sonographer Validation" / "Human-in-the-Loop Evaluation".
</details>

### Explainability (Token Attribution)

| Metric | FADA-SKD | FADA-FKD | FADA-Base |
|:-------|:--------:|:--------:|:---------:|
| Field Accuracy | **0.753** | 0.744 | 0.738 |
| Clinical Terms/Output | **17.27** | 17.13 | 17.04 |
| Unique Clinical Terms | **13.13** | 13.10 | 12.92 |
| Anatomical Structures | **3.96** | 3.83 | 3.72 |
| BLEU-1 | **0.766** | 0.752 | **0.766** |
| ROUGE-L | **0.790** | 0.774 | **0.790** |
| JSON Completeness | 100% | 100% | 100% |

<details>
<summary><strong>Token Attribution Methodology (click to expand)</strong></summary>

**Purpose:** Quantify the quality of structured clinical outputs generated by each model variant, explaining *why* FADA-SKD produces superior clinical interpretations despite receiving no feature-level supervision on interpretation data.

**Evaluation Protocol:**
- All three 4B model variants were evaluated on the same held-out interpretation test set
- Each model generated structured 8-field JSON outputs for fetal ultrasound images
- Outputs were compared against ground-truth expert sonographer annotations

**Metrics Explained:**

| Metric | How Computed |
|:-------|:------------|
| **Field Accuracy** | Mean semantic accuracy across all 8 JSON fields (anatomical_structures, fetal_orientation, imaging_plane, biometric_measurements, gestational_age, image_quality, normality_assessment, clinical_recommendations). Each field is scored for semantic correctness against ground truth. |
| **Clinical Terms/Output** | Count of recognized clinical/medical terminology tokens per model output (using a curated medical vocabulary). Higher = more clinically informative text. |
| **Unique Clinical Terms** | Number of distinct clinical terms per output (measures vocabulary diversity, not just repetition). |
| **Anatomical Structures** | Average number of anatomical structures correctly identified per image. |
| **BLEU-1** | Unigram overlap between generated text and reference (measures word-level precision). |
| **ROUGE-L** | Longest common subsequence overlap (measures structural similarity with reference). |
| **JSON Completeness** | Percentage of outputs that contain all required JSON fields with valid values. |

**Key Findings:**
- FADA-SKD achieves the highest per-field semantic accuracy (0.753 vs 0.738 Base, 0.744 FKD)
- FADA-FKD degrades on BLEU-1 (0.752 vs 0.766) and ROUGE-L (0.774 vs 0.790), confirming that full distillation introduces noise into language generation
- All variants achieve 100% JSON completeness, demonstrating reliable structured output generation

**Source data:** [`paper/analysis/xai_token_attribution.csv`](paper/analysis/xai_token_attribution.csv)

**Paper reference:** Section 4.5 "Interpretability Analysis", paragraph "Structured Output Quality" (corresponds to Supplementary Table S7).
</details>

---

## 5-Phase Inference Pipeline

FADA operates through a structured clinical workflow:

1. **INTERPRET** -- Generate structured 8-field JSON clinical interpretation (anatomical structures, orientation, plane, biometrics, gestational age, quality, normality, recommendations)
2. **CLASSIFY** -- Identify the anatomical view category from the interpretation
3. **MAP** -- Determine which anatomical structures to analyze based on a 5-tier priority system
4. **DETECT** -- Localize mapped structures with bounding boxes
5. **SEGMENT** -- Produce pixel-level polygon segmentation masks for detected structures

The pipeline supports two deployment modes:
- **Autonomous Mode** -- Fully automated end-to-end analysis
- **Human-in-the-Loop Mode** -- Clinician reviews interpretation, then selects subsequent analysis tasks

---

## Selective Knowledge Distillation

The core innovation of FADA-SKD is **selective** application of knowledge distillation:

```
L_SKD = L_task + lambda * 1[type in {det, seg, cls}] * L_feat
```

- **Interpretation data** (56,805 conversations): Trained with SFT loss only (L_task). Teachers are OFF.
- **Annotation data** (12,000 images): Trained with SFT + feature alignment from 4 teachers. Teachers are ON.

### Teacher Ensemble

| Teacher | Weight | Specialization | Architecture |
|:--------|:------:|:---------------|:-------------|
| FetalCLIP | 0.40 | Contrastive alignment | ViT-L, 1024-dim |
| UltraSAM | 0.25 | Segmentation | SAM ViT-B, 256-dim |
| USF-MAE | 0.20 | Self-supervised reconstruction | ViT-B, 768-dim |
| UltraFedFM | 0.15 | Federated foundation model | ViT-B, 768-dim |

Teacher features are pre-computed offline and cached in HDF5 format (37,799 images x 4 teachers x 3 layers = 453K vectors), eliminating concurrent teacher inference during training.

---

## Quick Start

### Installation

```bash
pip install torch transformers peft accelerate pillow
```

### Inference

```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image

# Load base model + FADA-SKD adapter
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3.5-VL-4B",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "mshz88/FADA-SKD-4B")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-VL-4B")

# Prepare input
image = Image.open("path/to/ultrasound.png")
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Interpret this fetal ultrasound image. Return a JSON object."}
    ]}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

# Generate
output = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

---

## Training

### Configuration

| Parameter | Value |
|:----------|:------|
| Base Model | Qwen3.5-VL-4B |
| Adaptation | LoRA (r=16, alpha=16) on all attention layers |
| Vision Encoder | 24-layer ViT with feature hooks at layers 7, 15, 23 |
| Epochs | 3 |
| Batch Size | 2 (gradient accumulation: 4, effective batch: 8) |
| Learning Rate | 2e-4 (cosine schedule) |
| Feature Loss Weight | lambda = 0.5 |
| Hardware | Single NVIDIA RTX 4090 (24GB VRAM) |

### Commands

```bash
# Train FADA-SKD (Selective Knowledge Distillation)
python training/train_skd.py --config training/configs/qwen35_vl_distill.yaml

# Train FADA-FKD (Full Knowledge Distillation)
python training/train_fkd.py --config training/configs/qwen35_vl_distill.yaml

# Train FADA-Base (no distillation)
python training/train_base.py --config training/configs/default.yaml
```

---

## Evaluation

```bash
python evaluation/evaluate.py \
    --model_path mshz88/FADA-SKD-4B \
    --data_dir data/ \
    --output_dir eval_results/
```

---

## Dataset

### Annotation Data (7 Repositories, 33 Structures)

| Source | Samples | Tasks | Structures |
|:-------|:-------:|:-----:|:-----------|
| FPUS23 | 2,400 | Classification | 6 standard fetal planes |
| FUSEP | 1,463 | Detection | 14 brain structures |
| Fetal Head | 544 | Segmentation | Brain, CSP, LV |
| CRL_NT | 5,481 | Detection/Segmentation/Keypoints | NT, CRL, scale bars |
| FOCUS | 1,500 | Detection | First-trimester cardiac |
| Fetal Abdominal | 700 | Segmentation | Artery, vein, liver, stomach |
| PS-FH | 1,358 | Segmentation | Pubic symphysis, fetal head |

**Test set:** 4,478 samples (1,463 detection, 544 segmentation, 2,400 classification, 71 keypoints)

### Interpretation Data

56,805 structured clinical conversations across 14 anatomical categories (18,935 unique images). An expert sonographer with >10 years of clinical experience provided structured clinical descriptions for each image, answering 8 standardized questions encoded as JSON fields:

| JSON Field | Description | Example |
|:-----------|:------------|:--------|
| `anatomical_structures` | Visible anatomical structures | "aorta, inferior vena cava, liver, stomach" |
| `fetal_orientation` | Spatial orientation of the fetus | "Axial upper abdomen, vertebral column to right" |
| `imaging_plane` | Ultrasound imaging plane | "Transverse trans-abdominal plane" |
| `biometric_measurements` | Obtainable biometric measurements | "AC" (abdominal circumference) |
| `gestational_age` | Estimated gestational age | "20-24 weeks" |
| `image_quality` | Assessment of image quality | "Good quality, adequate visualization" |
| `normality_assessment` | Normal vs. abnormal findings | "Normal anatomy, no abnormalities detected" |
| `clinical_recommendations` | Clinical follow-up recommendations | "Standard growth monitoring" |

**Dataset composition:** 37,870 entries with all 8 fields + 18,935 entries with 4-field subsets (anatomical_structures, fetal_orientation, imaging_plane, biometric_measurements).

**Our contribution:** The interpretation dataset is our primary data contribution. While the source ultrasound images come from publicly available repositories (see Data Licensing below), the structured clinical descriptions were created by our expert sonographer specifically for this project. This is what makes FADA's clinical interpretation capability possible.

Available on [Zenodo (DOI: 10.5281/zenodo.20104811)](https://doi.org/10.5281/zenodo.20104811).

---

## Data Licensing

### Source Images

Ultrasound images used in both the **annotation dataset** and the **interpretation dataset** are drawn from publicly available repositories released under [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/) or equivalent open licenses, except where noted below. No patient-identifiable data were used.

| Source Dataset | License | Original Repository |
|:---------------|:--------|:--------------------|
| FPUS23 (Prabakaran et al., 2023) | CC-BY-4.0 | [IEEE Access](https://ieeexplore.ieee.org/document/10146252) |
| Fetal Head / HC18 (van den Heuvel et al., 2018) | CC-BY-4.0 | [Zenodo](https://zenodo.org/records/1322001) |
| FOCUS (Wu et al., 2025) | CC-BY-4.0 | [Zenodo](https://zenodo.org/records/14597550) |
| Fetal Abdominal Structures (Da Correggio et al., 2023) | CC-BY-4.0 | [Mendeley Data](https://data.mendeley.com/datasets/4gcpm9dsc3/1) |
| PS-FH / PSFHS (Bai et al., 2024) | CC-BY-4.0 | [Zenodo](https://zenodo.org/records/10969427) |
| FETAL_PLANES_DB (Burgos-Artizzu et al., 2020) | CC-BY-4.0 | [Zenodo](https://zenodo.org/records/3904280) |
| Dataset for Fetus Framework (Cui & Dong, 2022) | CC-BY-4.0 | [Mendeley Data](https://data.mendeley.com/datasets/n2rbrb9t4f/1) |
| Fast-U-Net Dataset (Ashkani Chenarlogh et al., 2022) | CC-BY-4.0 | [GitHub](https://github.com/vahidashkani/Fast-U-Net) |
| First Trimester Fetal Echo (Stoean et al., 2022) | CC-BY-4.0 | [Figshare](https://figshare.com/articles/figure/First_Trimester_Fetal_Echocardiography_Data_Set_for_Classification/21215492) |
| FUSEP | Private | Not publicly available |
| CRL_NT | Private | Not publicly available |

### Our Contribution (Interpretation Dataset)

The structured clinical interpretation annotations (56,805 JSON conversations) are our original contribution, created by an expert sonographer. This dataset is released under **CC-BY-4.0** on [Zenodo](https://doi.org/10.5281/zenodo.20104811).

### Code and Model Weights

- **Code:** Apache License 2.0
- **Model weights (FADA-SKD-4B, FADA-Mobile-GGUF):** Apache License 2.0
- **Base model (Qwen3.5-VL):** Apache License 2.0 ([Qwen License](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/LICENSE))

**Ethics:** This study uses publicly available de-identified ultrasound datasets. Expert sonographer evaluation constitutes professional consultation and does not require separate IRB approval. No patient-identifiable data were collected or used. See Paper Section "Ethics approval" for details.

---

## Deployment

FADA supports multiple deployment targets:

| Platform | Model | Format | Latency | Connectivity |
|:---------|:------|:-------|:--------|:-------------|
| Cloud (high accuracy) | FADA-SKD (4B) | PyTorch + LoRA | ~20-25s/image | Required |
| Cloud (cost-efficient) | FADA-SKD (4B) | PyTorch + LoRA | ~35-45s/image | Required |
| **Mobile (Android)** | **FADA-SKD (0.8B)** | **GGUF Q4_K_M** | **~59s (full pipeline)** | **None** |

### Mobile Deployment

The 0.8B model is deployed on Android via [llama.cpp](https://github.com/ggerganov/llama.cpp) with GGUF quantization:

- **Model:** FADA-SKD 0.8B (Q4_K_M quantization)
- **Total download:** 712 MB (516 MB text model + 195 MB FP16 vision encoder)
- **Runtime:** llama.cpp with multimodal (MTMD) vision support
- **Test device:** Honor 90 (Snapdragon 7 Gen 1, 12 GB RAM, Android 15)
- **Latency:** ~40s per individual task (chat mode), ~59s full 5-phase pipeline (autonomous mode)
- **Connectivity:** None required (fully offline after model download)

**Deployment pipeline:** PyTorch -> GGUF Q4_K_M quantization -> llama.cpp (native C++ via JNI) -> Android app (Kotlin/Jetpack Compose) -> Clinical Point-of-Care

**Download the APK:** See [Releases](https://github.com/mahmoodphd/FADA/releases) for the latest Android APK.

**Mobile model weights:** [huggingface.co/mshz88/FADA-Mobile-GGUF](https://huggingface.co/mshz88/FADA-Mobile-GGUF)

---

## Project Structure

```
FADA/
├── README.md
├── LICENSE
├── requirements.txt
├── training/              # Training scripts
│   ├── train_skd.py       # Selective Knowledge Distillation
│   ├── train_fkd.py       # Full Knowledge Distillation
│   ├── train_base.py      # Baseline (no KD)
│   └── configs/           # YAML training configurations
├── evaluation/            # Evaluation pipeline
│   ├── evaluate.py
│   └── metrics/           # Detection, segmentation, parsing metrics
├── inference/             # Inference scripts
│   ├── infer.py
│   └── test_e2e_inference.py
├── data/                  # Data preparation utilities
│   ├── prepare_data.py
│   └── prepare_interpret_data.py
├── webapp/                # Gradio web application (HuggingFace Spaces)
├── losses/                # Knowledge distillation loss functions
├── models/                # Model architecture, feature hooks, projectors
└── figures/               # Architecture diagrams
```

---

## Links

| Resource | Link |
|:---------|:-----|
| **Web App Demo Video** | [YouTube](https://youtu.be/CbXcz74fn6k) |
| **Mobile App Demo Video** | [YouTube](https://youtu.be/RoogJqPNZ4w) |
| **Web Application** | [mshz88-fada-ultrasound-vlm.hf.space](https://mshz88-fada-ultrasound-vlm.hf.space) |
| **Model Weights (4B)** | [huggingface.co/mshz88/FADA-SKD-4B](https://huggingface.co/mshz88/FADA-SKD-4B) |
| **Model Weights (0.8B GGUF)** | [huggingface.co/mshz88/FADA-Mobile-GGUF](https://huggingface.co/mshz88/FADA-Mobile-GGUF) |
| **Mobile App (APK)** | [GitHub Releases](https://github.com/mahmoodphd/FADA/releases) |
| **Interpretation Dataset** | [Zenodo (DOI: 10.5281/zenodo.20104811)](https://doi.org/10.5281/zenodo.20104811) |
| **Source Code** | [github.com/mahmoodphd/FADA](https://github.com/mahmoodphd/FADA) |
| **Paper** | Submitted to *npj Digital Medicine* (2026) |

---

## Citation

If you use FADA in your research, please cite:

```bibtex
@article{fada2026,
  title={FADA: Knowledge-Distilled Vision-Language Models for Accessible Fetal
         Ultrasound Interpretation in Low-Resource Obstetric Settings},
  author={Alzubaidi, Mahmood and Agus, Marco},
  journal={Arxiv},
  year={2026},
  note={Submitted to the "Digital Health in Low-Resource Settings" Collection}
}
```

---

## Acknowledgments

- **IDRC** (International Development Research Centre) for funding support
- **QRDI** (Qatar Research, Development and Innovation Council) for research infrastructure
- The expert sonographers who contributed to clinical validation
- The open-source fetal ultrasound dataset communities

### Foundation Models Used

| Model | Paper | Code | Description |
|:------|:------|:-----|:------------|
| **Qwen3.5-VL** | [Bai et al., 2025](https://arxiv.org/abs/2502.13923) | [github.com/QwenLM/Qwen3.6](https://github.com/QwenLM/Qwen3.6) | Base vision-language model (4B parameters) |
| **FetalCLIP** | [Maani et al., Medical Image Analysis 2025](https://doi.org/10.1016/j.media.2025.103357) | [arxiv.org/abs/2502.14807](https://arxiv.org/abs/2502.14807) | CLIP-based model pre-trained on fetal ultrasound image-text pairs; ViT-L encoder, 1024-dim features |
| **UltraSAM** | [Meyer et al., IJCARS 2025](https://doi.org/10.1007/s11548-025-03517-8) | [arxiv.org/abs/2411.16222](https://arxiv.org/abs/2411.16222) | Segment Anything Model adapted for ultrasound; ViT-B encoder, 768-dim spatial features |
| **USF-MAE** | [Megahed et al., 2025](https://arxiv.org/abs/2510.22990) | [arxiv.org/abs/2510.22990](https://arxiv.org/abs/2510.22990) | Masked autoencoder pre-trained on 43 ultrasound datasets (500 epochs); ViT-B, 768-dim features |
| **UltraFedFM** | [Jiang et al., npj Digital Medicine 2025](https://doi.org/10.1038/s41746-025-02085-0) | [arxiv.org/abs/2411.16380](https://arxiv.org/abs/2411.16380) | Federated foundation model trained across multiple ultrasound domains; ViT-B, 768-dim features |

---

## License

This project is licensed under the [Apache License 2.0](LICENSE). The interpretation dataset is released under CC-BY-4.0. Most source images are from CC-BY-4.0 public repositories; two datasets (FUSEP, CRL_NT) remain private. See [Data Licensing](#data-licensing) for full details.

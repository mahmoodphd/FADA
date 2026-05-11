<div align="center">

# FADA: Knowledge-Distilled Vision-Language Models for Accessible Fetal Ultrasound Interpretation in Low-Resource Obstetric Settings

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-npj_Digital_Medicine-green.svg)](#citation)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace_Spaces-yellow.svg)](https://huggingface.co/spaces/mshz88/fada-ultrasound-vlm)
[![Model](https://img.shields.io/badge/Model-HuggingFace-orange.svg)](https://huggingface.co/mshz88/FADA-SKD-4B)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blue.svg)](https://doi.org/10.5281/zenodo.20104811)

</div>

---

## Overview

**FADA** (Fetal Anatomy Delineation and Analysis) is a unified vision-language model (VLM) built on Qwen3.5-VL that performs clinical interpretation, anatomical classification, bounding-box detection, and polygon segmentation of fetal ultrasound images within a single end-to-end pipeline. FADA employs **Selective Knowledge Distillation (SKD)** to transfer task-specific expertise from four domain-specific ultrasound foundation models into a compact student while preserving clinical reasoning capabilities.

A key finding is that applying feature-level distillation *only* to annotation data (detection, segmentation, classification) while training interpretation with supervised fine-tuning alone outperforms full distillation across all tasks. Expert sonographer validation across 237 images and 49 clinical cases confirms clinically acceptable performance. The system is designed for deployment in resource-constrained obstetric settings, aligned with UN Sustainable Development Goals 3 and 10.

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
| **FADA-Base (4B)** | 0.7798 | 0.4211 | 0.8813 | 0.8124 | 0.8225 |
| **FADA-SKD (4B)** | 0.7671 | 0.4402 | **0.8820** | **0.8149** | 0.8379 |
| **FADA-FKD (4B)** | 0.7413 | **0.4576** | 0.8798 | 0.8116 | **0.8383** |
| **FADA-Base (0.8B)** | 0.5876 | 0.2818 | 0.7956 | 0.7082 | 0.7564 |
| **FADA-SKD (0.8B)** | 0.6073 | 0.3111 | 0.8244 | 0.7388 | 0.7752 |

### Expert Sonographer Validation

| Evaluation Mode | Cases | Interpretation | Annotation | Overall |
|:----------------|:-----:|:--------------:|:----------:|:-------:|
| **Autonomous** | 237 images | 1.924 | 2.025 | 1.975 |
| **Human-in-the-Loop** | 49 cases | 1.286 (73.5% perfect) | 1.449 (63.3% perfect) | 1.368 |

> Scoring: 1 = clinically acceptable (no correction needed), 2 = minor issues, 3 = major errors. Lower is better.

### Explainability (Token Attribution)

| Metric | FADA-SKD | FADA-FKD | FADA-Base |
|:-------|:--------:|:--------:|:---------:|
| Field Accuracy | **0.753** | 0.744 | 0.738 |
| Clinical Terms/Output | **17.27** | 17.1 | 17.0 |
| Unique Terms | **13.13** | -- | -- |
| Anatomical Structures | **3.96** | -- | 3.72 |

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

56,805 structured clinical conversations across 14 anatomical categories. Each annotation follows an 8-field JSON schema covering anatomical structures, fetal orientation, imaging plane, biometric measurements, gestational age, image quality, normality assessment, and clinical recommendations.

Available on [Zenodo](https://doi.org/10.5281/zenodo.20104811).

---

## Deployment

FADA supports multiple deployment targets:

| Platform | Model | Format | Latency |
|:---------|:------|:-------|:--------|
| Cloud (Web App) | FADA-SKD (4B) | PyTorch + LoRA | ~3s/image |
| Mobile (Android) | FADA-SKD (0.8B) | ONNX INT8 | <2s/image |
| Browser (WebGPU) | FADA-SKD (0.8B) | ONNX | ~5s/image |

**Deployment pipeline:** PyTorch -> ONNX Export (INT8 quantization) -> ONNX Runtime 1.25 -> Android/Web App -> Clinical Point-of-Care

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
| **Web Application** | [huggingface.co/spaces/mshz88/fada-ultrasound-vlm](https://huggingface.co/spaces/mshz88/fada-ultrasound-vlm) |
| **Model Weights (4B)** | [huggingface.co/mshz88/FADA-SKD-4B](https://huggingface.co/mshz88/FADA-SKD-4B) |
| **Model Weights (0.8B ONNX)** | [huggingface.co/mshz88/FADA-Mobile-ONNX](https://huggingface.co/mshz88/FADA-Mobile-ONNX) |
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
  author={Al-Zubaidi, Mahmood Shehata and Al Maadeed, Somaya and Bouridane, Ahmed},
  journal={npj Digital Medicine},
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
- The developers of [Qwen-VL](https://github.com/QwenLM/Qwen2.5-VL), [FetalCLIP](https://github.com/BiomedCLIP), [UltraSAM](https://github.com/), [USF-MAE](https://github.com/), and [UltraFedFM](https://github.com/) for their foundational models

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

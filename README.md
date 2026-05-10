<div align="center">

# FADA: Knowledge-Distilled Vision-Language Models for Unified Fetal Ultrasound Interpretation and Annotation

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-npj_Digital_Medicine-green.svg)](#citation)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace_Spaces-yellow.svg)](https://huggingface.co/spaces/mshz88/fada-ultrasound-vlm)
[![Model](https://img.shields.io/badge/Model-HuggingFace-orange.svg)](https://huggingface.co/mshz88/FADA-SKD-4B)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blue.svg)](https://doi.org/10.5281/zenodo.15366091)

</div>

---

## Abstract

**FADA** is a unified Vision-Language Model (VLM) that performs five fetal ultrasound tasks — interpretation, classification, anatomical mapping, object detection, and segmentation — within a single end-to-end pipeline. We introduce **Selective Knowledge Distillation (SKD)**, which transfers task-specific expertise from four specialized teacher models (FetalCLIP, UltraSAM, USF-MAE, UltraFedFM) into a compact student VLM while preserving critical clinical reasoning capabilities. Expert validation with certified sonographers confirms that FADA achieves clinically acceptable performance. The framework is fully open-source and deployable in resource-constrained clinical settings via its lightweight 0.8B variant.

---

## Key Results

| Model | mAP@0.50 | mAP@0.75 | Dice | IoU | Cls Acc | Sonographer Score |
|:------|:--------:|:--------:|:----:|:---:|:-------:|:-----------------:|
| **FADA-Base (4B)** | 0.7798 | 0.4211 | 0.8813 | 0.8124 | 0.8225 | 1.925 |
| **FADA-SKD (4B)** | 0.7671 | 0.4402 | 0.8820 | 0.8149 | 0.8379 | 1.975 |
| **FADA-FKD (4B)** | 0.7413 | 0.4410 | 0.8798 | 0.8116 | 0.8383 | 1.904 |
| **FADA-Base (0.8B)** | 0.5876 | 0.2818 | 0.7956 | 0.7082 | 0.7564 | — |
| **FADA-SKD (0.8B)** | 0.6073 | 0.3111 | 0.8244 | 0.7388 | 0.7752 | — |

> **Sonographer Score**: Mean expert rating on a 3-point Likert scale (1=Acceptable, 2=Good, 3=Excellent) across 237 images and 49 clinical cases evaluated by certified sonographers.

---

## Architecture

![FADA Architecture](figures/workflow_diagram.png)

FADA operates through a **5-phase inference pipeline**:

1. **Interpret** — Generate a natural-language clinical interpretation of the ultrasound image
2. **Classify** — Identify the anatomical plane/view category
3. **Map** — Map anatomical structures present in the image
4. **Detect** — Localize structures with bounding boxes
5. **Segment** — Produce pixel-level segmentation masks

The Selective Knowledge Distillation framework transfers specialized knowledge from domain-expert teachers while preserving the student's language generation capabilities for clinical interpretation.

---

## Features

- **Unified multi-task analysis** — Single model handles interpretation, classification, mapping, detection, and segmentation
- **5-phase inference pipeline** — Structured clinical workflow from interpretation to segmentation
- **Selective Knowledge Distillation** — Preserves clinical reasoning while enhancing visual understanding
- **4-Teacher ensemble** — FetalCLIP, UltraSAM, USF-MAE, UltraFedFM
- **Expert validation** — 237 images + 49 clinical cases evaluated by certified sonographers
- **Dual deployment modes** — Autonomous and Human-in-the-Loop
- **Edge deployment** — Lightweight 0.8B variant for resource-constrained settings
- **14 anatomical views** — Comprehensive coverage of fetal ultrasound planes

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

# Load base model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3.5-VL-4B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load FADA-SKD LoRA adapter
model = PeftModel.from_pretrained(model, "mshz88/FADA-SKD-4B")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-VL-4B")

# Prepare input
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": "path/to/ultrasound.png"},
        {"type": "text", "text": "Interpret this fetal ultrasound image."}
    ]}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

# Generate
output = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

---

## Dataset

| Source | Images | Tasks | Categories |
|:-------|:------:|:-----:|:-----------|
| Custom Interpretation | 56,805 | Interpretation | 14 anatomical views |
| FOCUS Dataset | 1,500 | Detection | Biometry structures |
| CRL_NT Dataset | 5,481 | Detection / Segmentation | NT, CRL |
| Fetal Head Dataset | 1,334 | Segmentation | Head structures |
| FPUS23 | 11,398 | Classification | 4 standard planes |
| PS-FH Dataset | 1,358 | Segmentation | Pubic symphysis |
| Fast-U-Net | 700 | Segmentation | HC, AC |

> The combined dataset is available on [Zenodo](https://doi.org/10.5281/zenodo.15366091) (access upon request).

---

## Training

FADA uses **Selective Knowledge Distillation (SKD)** to train the student model:

| Parameter | Value |
|:----------|:------|
| Base Model | Qwen3.5-VL-4B |
| Adaptation | LoRA (r=16, α=32) |
| Epochs | 3 |
| Batch Size | 4 (gradient accumulation: 4) |
| Learning Rate | 2e-4 |
| Hardware | Single NVIDIA RTX 4090 (24GB VRAM) |
| Training Time | ~40 hours |

```bash
# Train FADA-SKD (Selective Knowledge Distillation)
python training/train_skd.py

# Train FADA-FKD (Full Knowledge Distillation)
python training/train_fkd.py

# Train FADA-Base (no distillation)
python training/train_base.py
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

## Project Structure

```
FADA/
├── README.md
├── LICENSE
├── requirements.txt
├── training/          # Training scripts
│   ├── train_skd.py   # Selective Knowledge Distillation
│   ├── train_fkd.py   # Full Knowledge Distillation
│   ├── train_base.py  # Baseline (no KD)
│   └── configs/       # Training configurations
├── evaluation/        # Evaluation pipeline
│   ├── evaluate.py
│   └── metrics/       # Detection, segmentation metrics
├── inference/         # Inference scripts
│   ├── infer.py
│   └── test_e2e_inference.py
├── data/              # Data preparation
│   ├── prepare_data.py
│   └── prepare_interpret_data.py
├── webapp/            # HuggingFace Spaces demo app
├── losses/            # KD loss functions
├── models/            # Model architecture & hooks
└── figures/           # Architecture diagrams
```

---

## Links

| Resource | Link |
|:---------|:-----|
| **Web Demo** | [HuggingFace Spaces](https://huggingface.co/spaces/mshz88/fada-ultrasound-vlm) |
| **Model Weights** | [HuggingFace](https://huggingface.co/mshz88/FADA-SKD-4B) *(available upon request)* |
| **Dataset** | [Zenodo](https://doi.org/10.5281/zenodo.15366091) *(available upon request)* |
| **Paper** | Submitted to *npj Digital Medicine* |

---

## Citation

```bibtex
@article{fada2026,
  title={FADA: Knowledge-Distilled Vision-Language Models for Unified Fetal Ultrasound Interpretation and Annotation},
  author={Mahmood, Marco Shehata and Al Maadeed, Somaya and Bouridane, Ahmed},
  journal={npj Digital Medicine},
  year={2026},
  note={Submitted}
}
```

---

## Acknowledgments

We gratefully acknowledge:

- **IDRC** (International Development Research Centre) for funding support
- **QRDI** (Qatar Research, Development and Innovation Council) for research infrastructure
- The **expert sonographers** who contributed their time and expertise to clinical validation
- The open-source **fetal ultrasound dataset communities** whose shared resources made this work possible
- The developers of [Qwen-VL](https://github.com/QwenLM/Qwen2.5-VL), [FetalCLIP](https://github.com/), [UltraSAM](https://github.com/), [USF-MAE](https://github.com/), and [UltraFedFM](https://github.com/) for their foundational models

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

```
Copyright 2026 FADA Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

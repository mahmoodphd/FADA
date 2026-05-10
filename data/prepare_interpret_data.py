"""
prepare_interpret_data.py
=========================
Parse the sonographer-annotated Excel file and generate interpretation
training data in the same conversation JSONL format used by the existing
fetal VLM pipeline.

Produces 3 prompt variants per image (~57K conversations from ~19K images).

Usage:
    python prepare_interpret_data.py \
        --excel  "/media/mshz88/OS/Qwen/Fetal Ultrasound Annotations Normalized.xlsx" \
        --images "/media/mshz88/OS/Qwen/fetal_ultrasound_interpret/images" \
        --output "/media/mshz88/OS/Qwen/fetal_vlm_kd/processed_data/interpret_train.jsonl"
"""

import argparse
import json
import os
import random
from pathlib import Path

import openpyxl

# ---------------------------------------------------------------------------
# Column name mapping (exact names from Excel)
# ---------------------------------------------------------------------------
COL_FOLDER = "Folder Name"
COL_IMAGE  = "Image Name"
COL_Q1     = "Q1: Anatomical Structures"
COL_Q2     = "Q2: Fetal Orientation"
COL_Q3     = "Q3: Imaging Plane"
COL_Q4     = "Q4: Biometric Measurements"
COL_Q5     = "Q5: Gestational Age"
COL_Q6     = "Q6: Image Quality"
COL_Q7     = "Q7: Normality Assessment"
COL_Q8     = "Q8: Clinical Recommendations"

Q_COLS = [COL_Q1, COL_Q2, COL_Q3, COL_Q4, COL_Q5, COL_Q6, COL_Q7, COL_Q8]

# ---------------------------------------------------------------------------
# Prompt templates  (3 variants)
# ---------------------------------------------------------------------------

# Variant 1: Full interpretation -- all 8 fields
PROMPT_FULL = (
    "Provide a comprehensive clinical interpretation of this fetal ultrasound "
    "image. Identify all visible anatomical structures, describe the fetal "
    "orientation and imaging plane, note any measurable biometric parameters, "
    "estimate the gestational age, assess image quality, evaluate normality, "
    "and provide clinical recommendations. "
    "Return your answer as a JSON object with keys: "
    "anatomical_structures, fetal_orientation, imaging_plane, "
    "biometric_measurements, gestational_age, image_quality, "
    "normality_assessment, clinical_recommendations."
)

# Variant 2: Anatomical focus -- Q1, Q2, Q3, Q4
PROMPT_ANATOMY = (
    "Analyze this fetal ultrasound image. Identify all visible anatomical "
    "structures, describe the fetal orientation, determine the imaging plane, "
    "and list the biometric measurements obtainable from this view. "
    "Return your answer as a JSON object with keys: "
    "anatomical_structures, fetal_orientation, imaging_plane, "
    "biometric_measurements."
)

# Variant 3: Clinical assessment -- Q5, Q6, Q7, Q8
PROMPT_CLINICAL = (
    "Evaluate this fetal ultrasound image for clinical assessment. "
    "Estimate the gestational age, assess the image quality, evaluate "
    "whether the findings are normal or abnormal, and provide clinical "
    "recommendations. "
    "Return your answer as a JSON object with keys: "
    "gestational_age, image_quality, normality_assessment, "
    "clinical_recommendations."
)


def _val(cell_value):
    """Return stripped string or empty string for None."""
    if cell_value is None:
        return ""
    return str(cell_value).strip()


def build_response_full(row: dict) -> str:
    """Build JSON response string for the full interpretation variant."""
    obj = {
        "anatomical_structures": _val(row[COL_Q1]),
        "fetal_orientation":     _val(row[COL_Q2]),
        "imaging_plane":         _val(row[COL_Q3]),
        "biometric_measurements": _val(row[COL_Q4]),
        "gestational_age":       _val(row[COL_Q5]),
        "image_quality":         _val(row[COL_Q6]),
        "normality_assessment":  _val(row[COL_Q7]),
        "clinical_recommendations": _val(row[COL_Q8]),
    }
    return json.dumps(obj, ensure_ascii=False)


def build_response_anatomy(row: dict) -> str:
    """Build JSON response for anatomy variant (Q1-Q4)."""
    obj = {
        "anatomical_structures": _val(row[COL_Q1]),
        "fetal_orientation":     _val(row[COL_Q2]),
        "imaging_plane":         _val(row[COL_Q3]),
        "biometric_measurements": _val(row[COL_Q4]),
    }
    return json.dumps(obj, ensure_ascii=False)


def build_response_clinical(row: dict) -> str:
    """Build JSON response for clinical assessment variant (Q5-Q8)."""
    obj = {
        "gestational_age":       _val(row[COL_Q5]),
        "image_quality":         _val(row[COL_Q6]),
        "normality_assessment":  _val(row[COL_Q7]),
        "clinical_recommendations": _val(row[COL_Q8]),
    }
    return json.dumps(obj, ensure_ascii=False)


def make_conversation(image_path: str, prompt: str, response: str) -> dict:
    """Create a single conversation entry matching the existing JSONL format."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text",  "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response},
                ],
            },
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare interpretation JSONL")
    parser.add_argument(
        "--excel",
        default="/media/mshz88/OS/Qwen/Fetal Ultrasound Annotations Normalized.xlsx",
    )
    parser.add_argument(
        "--images",
        default="/media/mshz88/OS/Qwen/fetal_ultrasound_interpret/images",
    )
    parser.add_argument(
        "--output",
        default="/media/mshz88/OS/Qwen/fetal_vlm_kd/processed_data/interpret_train.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    images_root = Path(args.images)

    # ---- Read Excel --------------------------------------------------------
    print(f"Reading Excel: {args.excel}")
    wb = openpyxl.load_workbook(args.excel, read_only=True, data_only=True)
    ws = wb.active

    rows_iter = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h else "" for h in next(rows_iter)]

    # Map column indices
    col_idx = {}
    for name in [COL_FOLDER, COL_IMAGE] + Q_COLS:
        try:
            col_idx[name] = header.index(name)
        except ValueError:
            raise ValueError(f"Column '{name}' not found in header: {header}")

    # ---- Process rows ------------------------------------------------------
    conversations = []
    skipped_missing = 0
    skipped_empty = 0
    seen = set()  # deduplicate

    for raw_row in rows_iter:
        row = {name: raw_row[col_idx[name]] for name in col_idx}
        folder = _val(row[COL_FOLDER])
        image  = _val(row[COL_IMAGE])

        if not folder or not image:
            skipped_empty += 1
            continue

        # Deduplicate by (folder, image)
        key = (folder, image)
        if key in seen:
            continue
        seen.add(key)

        # Build absolute image path
        img_path = str(images_root / folder / image)
        if not os.path.isfile(img_path):
            skipped_missing += 1
            continue

        # Check that at least Q1 has content
        if not _val(row[COL_Q1]):
            skipped_empty += 1
            continue

        # Variant 1: Full interpretation
        conversations.append(
            make_conversation(img_path, PROMPT_FULL, build_response_full(row))
        )
        # Variant 2: Anatomical analysis
        conversations.append(
            make_conversation(img_path, PROMPT_ANATOMY, build_response_anatomy(row))
        )
        # Variant 3: Clinical assessment
        conversations.append(
            make_conversation(img_path, PROMPT_CLINICAL, build_response_clinical(row))
        )

    wb.close()

    # Shuffle for training
    random.shuffle(conversations)

    # ---- Write JSONL -------------------------------------------------------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    # ---- Summary -----------------------------------------------------------
    n_images = len(seen)
    n_convs  = len(conversations)
    print(f"\nDone!")
    print(f"  Images processed : {n_images}")
    print(f"  Conversations    : {n_convs}  (3 variants x {n_images} images)")
    print(f"  Skipped (missing): {skipped_missing}")
    print(f"  Skipped (empty)  : {skipped_empty}")
    print(f"  Output           : {args.output}")


if __name__ == "__main__":
    main()

"""
merge_datasets.py
=================
Merge the existing detection/segmentation/classification training JSONL
with the new interpretation training JSONL into a single mixed dataset.

Usage:
    python merge_datasets.py \
        --existing  processed_data/train.jsonl \
        --interpret processed_data/interpret_train.jsonl \
        --output    processed_data/mixed_train.jsonl
"""

import argparse
import json
import os
import random


def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    parser = argparse.ArgumentParser(description="Merge training datasets")
    parser.add_argument(
        "--existing",
        default="/media/mshz88/OS/Qwen/fetal_vlm_kd/processed_data/train.jsonl",
    )
    parser.add_argument(
        "--interpret",
        default="/media/mshz88/OS/Qwen/fetal_vlm_kd/processed_data/interpret_train.jsonl",
    )
    parser.add_argument(
        "--output",
        default="/media/mshz88/OS/Qwen/fetal_vlm_kd/processed_data/mixed_train.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ---- Load both datasets ------------------------------------------------
    all_lines = []

    print(f"Loading existing data: {args.existing}")
    n_existing = 0
    with open(args.existing, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_lines.append(line)
                n_existing += 1

    print(f"Loading interpretation data: {args.interpret}")
    n_interpret = 0
    with open(args.interpret, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_lines.append(line)
                n_interpret += 1

    # ---- Shuffle -----------------------------------------------------------
    random.shuffle(all_lines)

    # ---- Write output ------------------------------------------------------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")

    # ---- Summary -----------------------------------------------------------
    n_total = len(all_lines)
    pct_existing  = n_existing / n_total * 100
    pct_interpret = n_interpret / n_total * 100
    print(f"\nDone!")
    print(f"  Existing (det/seg/cls) : {n_existing:>6d}  ({pct_existing:.1f}%)")
    print(f"  Interpretation        : {n_interpret:>6d}  ({pct_interpret:.1f}%)")
    print(f"  Total mixed           : {n_total:>6d}")
    print(f"  Output                : {args.output}")


if __name__ == "__main__":
    main()

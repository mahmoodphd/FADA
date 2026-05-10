#!/usr/bin/env python3
"""Train Fetal VLM from scratch (no KD) on the mixed dataset.

Fresh LoRA on Qwen3.5-VL 4B, trained on mixed_train.jsonl
(det/seg/cls/keypoint + interpretation). Same hyperparameters as v2
(lr=2e-4, 3 epochs) but on the larger mixed dataset.

Usage:
    python train_v2_mixed.py
    python train_v2_mixed.py --output-dir outputs_mixed/v2_mixed
    python train_v2_mixed.py --resume
"""
from __future__ import annotations

# Disable torch.compile/dynamo BEFORE other imports
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.reset()
import torch._inductor.config as inductor_config
inductor_config.compile_threads = 1

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

# Same hyperparameters as v2 (train.py)
DEFAULTS = {
    "model_name": "unsloth/Qwen3.5-4B",
    "max_seq_length": 4096,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.001,
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 500,
    "save_total_limit": 3,
    "eval_strategy": "steps",
    "eval_steps": 500,
}


def load_dataset_from_jsonl(jsonl_path: str):
    """Load a pre-processed JSONL dataset."""
    conversations = []
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            conversations.append(entry)
    logger.info("Loaded %d conversations from %s", len(conversations), jsonl_path)
    return conversations


def main():
    parser = argparse.ArgumentParser(description="Train Fetal VLM from scratch on mixed data (no KD)")
    parser.add_argument("--model-name", default=DEFAULTS["model_name"])
    parser.add_argument(
        "--train-data",
        default=str(PROJECT_ROOT / "processed_data" / "mixed_train.jsonl"),
    )
    parser.add_argument(
        "--val-data",
        default=str(PROJECT_ROOT / "processed_data" / "val.jsonl"),
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs_mixed" / "v2_mixed"))
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["num_train_epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["per_device_train_batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["gradient_accumulation_steps"])
    parser.add_argument("--save-steps", type=int, default=DEFAULTS["save_steps"])
    parser.add_argument("--eval-steps", type=int, default=DEFAULTS["eval_steps"])
    parser.add_argument("--max-seq-length", type=int, default=DEFAULTS["max_seq_length"])
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # ---- Environment ----
    logger.info("=" * 60)
    logger.info("  Fetal VLM v2_mixed (from scratch, no KD, mixed data)")
    logger.info("=" * 60)
    logger.info("Training data:  %s", args.train_data)
    logger.info("Output dir:     %s", args.output_dir)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s (%.1f GB total)", gpu.name, gpu.total_memory / 1e9)
    else:
        logger.error("No GPU available!")
        sys.exit(1)

    # ---- Load model ----
    logger.info("Loading base model: %s", args.model_name)
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
    )

    # ---- Apply fresh LoRA (no adapter loading) ----
    logger.info("Applying fresh LoRA (r=%d, alpha=%d)", DEFAULTS["lora_r"], DEFAULTS["lora_alpha"])
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=DEFAULTS["lora_r"],
        lora_alpha=DEFAULTS["lora_alpha"],
        lora_dropout=DEFAULTS["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ---- Load data ----
    logger.info("Loading training data...")
    train_data = load_dataset_from_jsonl(args.train_data)
    logger.info("Loading validation data...")
    val_data = load_dataset_from_jsonl(args.val_data)

    train_dataset = train_data
    val_dataset = val_data

    logger.info("Train: %d conversations", len(train_dataset))
    logger.info("Val:   %d conversations", len(val_dataset))

    # ---- Estimate steps ----
    steps_per_epoch = len(train_dataset) // (args.batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    logger.info("Steps/epoch: %d | Total steps: %d", steps_per_epoch, total_steps)

    # ---- Set up trainer ----
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    FastVisionModel.for_training(model)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=DEFAULTS["weight_decay"],
        warmup_ratio=DEFAULTS["warmup_ratio"],
        lr_scheduler_type=DEFAULTS["lr_scheduler_type"],
        logging_steps=DEFAULTS["logging_steps"],
        logging_strategy="steps",
        save_strategy=DEFAULTS["save_strategy"],
        save_steps=args.save_steps,
        save_total_limit=DEFAULTS["save_total_limit"],
        eval_strategy=DEFAULTS["eval_strategy"],
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=args.max_seq_length,
        per_device_eval_batch_size=8,
        dataloader_num_workers=8,
        seed=42,
        report_to="none",
        disable_tqdm=True,
        log_level="info",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
    )

    # ---- Memory info ----
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    max_memory = round(gpu_stats.total_memory / 1024**3, 3)
    logger.info("GPU memory reserved: %.1f / %.1f GB", start_gpu_memory, max_memory)

    # ---- Train ----
    logger.info("Starting training...")
    start_time = time.time()

    resume_from = None
    if args.resume:
        output_path = Path(args.output_dir)
        checkpoints = sorted(
            output_path.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if checkpoints:
            resume_from = str(checkpoints[-1])
            logger.info("Resuming from checkpoint: %s", resume_from)
        else:
            logger.warning("No checkpoints found in %s, starting fresh", args.output_dir)

    trainer_stats = trainer.train(resume_from_checkpoint=resume_from)

    elapsed = time.time() - start_time

    # ---- Final stats ----
    used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    used_for_training = round(used_memory - start_gpu_memory, 3)

    logger.info("=" * 60)
    logger.info("  Training Complete!")
    logger.info("=" * 60)
    logger.info("  Duration: %.0f seconds (%.1f hours)", elapsed, elapsed / 3600)
    logger.info("  Final train loss: %.4f", trainer_stats.metrics.get("train_loss", -1))
    logger.info("  Peak GPU memory: %.1f GB (%.1f GB for training)", used_memory, used_for_training)
    logger.info("  Output dir: %s", args.output_dir)

    # ---- Save final model ----
    final_dir = str(Path(args.output_dir) / "final")
    logger.info("Saving final model to %s", final_dir)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train Fetal VLM from scratch with KD on the mixed dataset.

Fresh LoRA on Qwen3.5-VL 4B, trained on mixed_train.jsonl with offline KD
using 4 teacher models (fusion_mse strategy, same as exp2_teacher_fusion).

For interpretation images not in the teacher feature cache, KD is automatically
skipped and only SFT loss is used. For original det/seg/cls/keypoint images,
SFT + KD loss is applied.

Usage:
    python train_exp2_mixed.py
    python train_exp2_mixed.py --output-dir outputs_mixed/exp2_mixed
    python train_exp2_mixed.py --resume
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
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
# Add cloud code directory to path for fetal_vlm_kd package imports
_cloud_code = PROJECT_ROOT.parent / "fetal_vlm_kd_cloud" / "code"
if _cloud_code.is_dir() and str(_cloud_code) not in sys.path:
    sys.path.insert(0, str(_cloud_code))

# Hyperparameters: lr=2e-4, 3 epochs (same as v2, since training from scratch)
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

# Student layer indices (24 vision blocks, proportional depth)
STUDENT_LAYER_INDICES = [7, 15, 23]

# Teacher configurations (same as exp2_teacher_fusion)
TEACHER_LAYER_MAP = {
    "fetal_clip":    {"layer_indices": [7, 15, 23],   "feature_dim": 1024},
    "ultra_fed_fm":  {"layer_indices": [3, 7, 11],    "feature_dim": 768},
    "ultra_sam":     {"layer_indices": [3, 7, 11],    "feature_dim": 768},
    "usf_mae":       {"layer_indices": [3, 7, 11],    "feature_dim": 768},
}

STUDENT_DIM = 1024

# Teacher importance weights (same as exp2)
TEACHER_WEIGHTS = {
    "fetal_clip": 0.4,
    "ultra_sam": 0.25,
    "usf_mae": 0.2,
    "ultra_fed_fm": 0.15,
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


def build_feature_pairs(
    teachers: Optional[List[str]] = None,
    student_layers: Optional[List[int]] = None,
) -> List[Tuple[str, str]]:
    """Build (student_hook_name, teacher_cache_key) pairs."""
    if student_layers is None:
        student_layers = STUDENT_LAYER_INDICES
    if teachers is None:
        teachers = list(TEACHER_LAYER_MAP.keys())

    pairs = []
    for teacher_name in teachers:
        if teacher_name not in TEACHER_LAYER_MAP:
            logger.warning("Unknown teacher '%s', skipping", teacher_name)
            continue
        config = TEACHER_LAYER_MAP[teacher_name]
        teacher_indices = config["layer_indices"]

        for s_idx in student_layers:
            try:
                pos = STUDENT_LAYER_INDICES.index(s_idx)
            except ValueError:
                continue
            if pos < len(teacher_indices):
                t_idx = teacher_indices[pos]
                s_name = f"student_layer{s_idx}"
                t_key = f"{teacher_name}_layer{t_idx}"
                pairs.append((s_name, t_key))

    logger.info("Built %d feature pairs: %s", len(pairs), pairs)
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Train Fetal VLM from scratch with KD on mixed data (exp2_mixed)"
    )
    parser.add_argument("--model-name", default=DEFAULTS["model_name"])
    parser.add_argument(
        "--train-data",
        default=str(PROJECT_ROOT / "processed_data" / "mixed_train.jsonl"),
    )
    parser.add_argument(
        "--val-data",
        default=str(PROJECT_ROOT / "processed_data" / "val.jsonl"),
    )
    parser.add_argument(
        "--cache-path",
        default=str(PROJECT_ROOT / "processed_data" / "teacher_feature_cache.h5"),
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs_mixed" / "exp2_mixed"))
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["num_train_epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["per_device_train_batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["gradient_accumulation_steps"])
    parser.add_argument("--save-steps", type=int, default=DEFAULTS["save_steps"])
    parser.add_argument("--eval-steps", type=int, default=DEFAULTS["eval_steps"])
    parser.add_argument("--max-seq-length", type=int, default=DEFAULTS["max_seq_length"])
    parser.add_argument("--w-feat", type=float, default=0.5, help="KD feature loss weight")
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
    logger.info("  Fetal VLM exp2_mixed (from scratch, KD fusion_mse, mixed data)")
    logger.info("=" * 60)
    logger.info("Training data:  %s", args.train_data)
    logger.info("Feature cache:  %s", args.cache_path)
    logger.info("Output dir:     %s", args.output_dir)
    logger.info("Loss type:      fusion_mse")
    logger.info("Teachers:       %s", list(TEACHER_LAYER_MAP.keys()))
    logger.info("w_feat:         %.2f", args.w_feat)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s (%.1f GB total)", gpu.name, gpu.total_memory / 1e9)
    else:
        logger.error("No GPU available!")
        sys.exit(1)

    # ---- Verify teacher feature cache ----
    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        logger.error("Teacher feature cache not found at %s", cache_path)
        sys.exit(1)
    logger.info("Feature cache: %s (%.1f MB)", cache_path, cache_path.stat().st_size / 1e6)

    # ---- Load model ----
    logger.info("Loading base model: %s", args.model_name)
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
    )

    # ---- Apply fresh LoRA (no adapter loading -- from scratch) ----
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
    logger.info("Starting with fresh LoRA weights (no pre-trained adapter)")

    # ---- Load data ----
    logger.info("Loading training data...")
    train_data = load_dataset_from_jsonl(args.train_data)
    logger.info("Loading validation data...")
    val_data = load_dataset_from_jsonl(args.val_data)

    train_dataset = train_data
    val_dataset = val_data

    logger.info("Train: %d conversations", len(train_dataset))
    logger.info("Val:   %d conversations", len(val_dataset))

    # ---- Load teacher feature cache ----
    logger.info("Loading teacher feature cache...")
    from fetal_vlm_kd.data.feature_cache import TeacherFeatureCache

    feature_cache = TeacherFeatureCache(args.cache_path, mode="r")
    logger.info("Cache loaded: %d images", len(feature_cache))

    # ---- Set up student feature hooks ----
    student_layers = STUDENT_LAYER_INDICES
    logger.info("Attaching hooks to student vision blocks %s", student_layers)
    from fetal_vlm_kd.models.feature_hooks import FeatureHookManager

    hook_manager = FeatureHookManager()

    vision_blocks = None
    for attr_path in [
        "model.model.visual.blocks",
        "model.visual.blocks",
        "base_model.model.model.visual.blocks",
    ]:
        obj = model
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            vision_blocks = obj
            logger.info("Found vision blocks at: %s (len=%d)", attr_path, len(vision_blocks))
            break
        except AttributeError:
            continue

    if vision_blocks is None:
        from fetal_vlm_kd.models.feature_hooks import _find_blocks
        if hasattr(model, 'model') and hasattr(model.model, 'model'):
            visual = getattr(model.model.model, 'visual', None)
            if visual is not None:
                vision_blocks = _find_blocks(visual)
                if vision_blocks is not None:
                    logger.info("Found vision blocks via _find_blocks: len=%d", len(vision_blocks))

    if vision_blocks is None:
        logger.error("Could not locate vision encoder blocks. Exiting.")
        sys.exit(1)

    for idx in student_layers:
        hook_name = f"student_layer{idx}"
        hook_manager.attach(vision_blocks[idx], hook_name)
        logger.info("  Attached hook: %s -> block[%d]", hook_name, idx)

    # ---- Set up projector bank ----
    from fetal_vlm_kd.models.projectors import ProjectorBank

    projector_bank = ProjectorBank()
    feature_pairs = build_feature_pairs()

    for s_name, t_key in feature_pairs:
        parts = t_key.rsplit("_layer", 1)
        teacher_name = parts[0]
        teacher_dim = TEACHER_LAYER_MAP[teacher_name]["feature_dim"]

        proj_name = f"{s_name}_to_{t_key}"
        if STUDENT_DIM != teacher_dim:
            projector_bank.add_projector(proj_name, STUDENT_DIM, teacher_dim)
        else:
            logger.info("  Dims match for %s, no projector needed", proj_name)

    total_proj_params = sum(p.numel() for p in projector_bank.parameters())
    logger.info(
        "ProjectorBank: %d projectors, %d total parameters",
        len(projector_bank.projector_names), total_proj_params,
    )

    # ---- Set up reverse projectors for fusion_mse ----
    fusion_reverse_projectors = ProjectorBank()
    for s_name, t_key in feature_pairs:
        parts = t_key.rsplit("_layer", 1)
        teacher_name = parts[0]
        teacher_dim = TEACHER_LAYER_MAP[teacher_name]["feature_dim"]

        if teacher_dim != STUDENT_DIM:
            rev_name = f"{t_key}_to_{s_name}"
            fusion_reverse_projectors.add_projector(rev_name, teacher_dim, STUDENT_DIM)
    rev_params = sum(p.numel() for p in fusion_reverse_projectors.parameters())
    logger.info("Reverse ProjectorBank: %d projectors, %d params",
                 len(fusion_reverse_projectors.projector_names), rev_params)

    # ---- Estimate steps ----
    steps_per_epoch = len(train_dataset) // (args.batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    logger.info("Steps/epoch: %d | Total steps: %d", steps_per_epoch, total_steps)

    # ---- Set up trainer ----
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from fetal_vlm_kd.training.collator import KDVisionDataCollator
    from fetal_vlm_kd.training.kd_trainer import OfflineDistillationSFTTrainer

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

    # Inject KD collator into train dataloader
    kd_collator = KDVisionDataCollator(trainer.data_collator, feature_cache)

    _orig_get_train_dl = trainer.get_train_dataloader

    def _kd_get_train_dataloader():
        saved = trainer.data_collator
        trainer.data_collator = kd_collator
        try:
            dl = _orig_get_train_dl()
        finally:
            trainer.data_collator = saved
        logger.info("Injected KDVisionDataCollator into train DataLoader")
        return dl

    trainer.get_train_dataloader = _kd_get_train_dataloader

    # Wrap with offline distillation trainer
    kd_trainer = OfflineDistillationSFTTrainer(
        sft_trainer=trainer,
        hook_manager=hook_manager,
        projector_bank=projector_bank,
        feature_pairs=feature_pairs,
        w_feat=args.w_feat,
        kd_warmup_ratio=0.0,
        normalize_features=True,
        teacher_weights=TEACHER_WEIGHTS,
        loss_type="fusion_mse",
        fusion_reverse_projectors=fusion_reverse_projectors,
    )

    # ---- Memory info ----
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    max_memory = round(gpu_stats.total_memory / 1024**3, 3)
    logger.info("GPU memory reserved: %.1f / %.1f GB", start_gpu_memory, max_memory)

    # ---- Train ----
    logger.info("Starting exp2_mixed KD training (fusion_mse, w_feat=%.2f)...", args.w_feat)
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

    trainer_stats = kd_trainer.train(resume_from_checkpoint=resume_from)

    elapsed = time.time() - start_time

    # ---- Final stats ----
    used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    used_for_training = round(used_memory - start_gpu_memory, 3)

    logger.info("=" * 60)
    logger.info("  exp2_mixed KD Training Complete!")
    logger.info("=" * 60)
    logger.info("  Duration: %.0f seconds (%.1f hours)", elapsed, elapsed / 3600)
    logger.info("  Final train loss: %.4f", trainer_stats.metrics.get("train_loss", -1))
    logger.info("  Peak GPU memory: %.1f GB (%.1f GB for training)", used_memory, used_for_training)
    logger.info("  Output dir: %s", args.output_dir)

    # Log KD-specific stats
    loss_history = kd_trainer.loss_history
    if loss_history:
        last_10 = loss_history[-10:]
        avg_task = sum(d["task"] for d in last_10) / len(last_10)
        avg_feat = sum(d["feat"] for d in last_10) / len(last_10)
        avg_total = sum(d["total"] for d in last_10) / len(last_10)
        logger.info("  Last 10 steps avg: task=%.4f, feat=%.4f, total=%.4f",
                     avg_task, avg_feat, avg_total)

    # ---- Save final model ----
    final_dir = str(Path(args.output_dir) / "final")
    logger.info("Saving final model to %s", final_dir)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save projector bank
    if list(projector_bank.parameters()):
        import os
        proj_path = os.path.join(final_dir, "projector_bank.pt")
        torch.save(projector_bank.state_dict(), proj_path)
        logger.info("Saved projector bank to %s", proj_path)

    # Save experiment metadata
    metadata = {
        "loss_type": "fusion_mse",
        "w_feat": args.w_feat,
        "teachers": list(TEACHER_LAYER_MAP.keys()),
        "teacher_weights": TEACHER_WEIGHTS,
        "student_layers": STUDENT_LAYER_INDICES,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "skip_v2_adapter": True,
        "duration_seconds": elapsed,
        "final_train_loss": trainer_stats.metrics.get("train_loss", -1),
        "num_feature_pairs": len(feature_pairs),
        "data": "mixed_train.jsonl (det/seg/cls/keypoint + interpretation)",
    }
    meta_path = Path(args.output_dir) / "experiment_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved experiment metadata to %s", meta_path)

    # Clean up hooks
    hook_manager.remove_all()

    logger.info("Done!")


if __name__ == "__main__":
    main()

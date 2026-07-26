#!/usr/bin/env python3
"""LoRA fine-tuning for Gemma 4 E2B racing coach.

Adapted from finRegTechPipeline/training/train_lora.py with all lessons
from TRAINING_NOTES.md applied:
  - peft_config passed to SFTTrainer (NOT get_peft_model)
  - modules_to_save + ensure_weight_tying
  - Gemma4ClippableLinear patch (peft#3129)
  - lr=1e-5, r=8, 1 epoch (v2 recipe)

Usage:
    python training/train_racing_coach.py \
        --config configs/gemma4_e2b_racing_lora_v1.yaml \
        --train-data data/racing_sft_train.jsonl \
        --val-data data/racing_sft_val.jsonl

    # Smoke test (10 steps)
    python training/train_racing_coach.py \
        --config configs/gemma4_e2b_racing_lora_v1.yaml \
        --train-data data/racing_sft_train.jsonl \
        --max-steps 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def _patch_gemma4_clippable_linear() -> None:
    """Monkey-patch Gemma4ClippableLinear to inherit from nn.Linear.
    PEFT only recognizes nn.Linear. See: https://github.com/huggingface/peft/issues/3129
    """
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        return

    class PatchedClippableLinear(nn.Linear):
        def __init__(self, config, in_features, out_features):
            nn.Linear.__init__(self, in_features, out_features, bias=False)
            self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
            if self.use_clipped_linears:
                self.register_buffer("input_min", torch.tensor(-float("inf")))
                self.register_buffer("input_max", torch.tensor(float("inf")))
                self.register_buffer("output_min", torch.tensor(-float("inf")))
                self.register_buffer("output_max", torch.tensor(float("inf")))

        def forward(self, x):
            if self.use_clipped_linears:
                x = torch.clamp(x, self.input_min, self.input_max)
            out = nn.Linear.forward(self, x)
            if self.use_clipped_linears:
                out = torch.clamp(out, self.output_min, self.output_max)
            return out

    modeling_gemma4.Gemma4ClippableLinear = PatchedClippableLinear
    print("Patched Gemma4ClippableLinear (peft#3129)")


def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print(f"Config: {json.dumps(cfg, indent=2)}")

    # --- Patch + Load model ---
    _patch_gemma4_clippable_linear()

    model_name = cfg["model_name"]
    torch_dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16

    model_kwargs = dict(torch_dtype=torch_dtype, device_map={"": 0})

    if cfg.get("load_in_4bit", False):
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch_dtype,
        )
        print("QLoRA: 4-bit quantization enabled")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- LoRA config (CRITICAL: pass to SFTTrainer, not get_peft_model) ---
    peft_config = LoraConfig(
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.0),
        r=cfg["lora_r"],
        bias="none",
        target_modules=cfg.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
        modules_to_save=cfg.get("modules_to_save", ["lm_head", "embed_tokens"]),
        ensure_weight_tying=cfg.get("ensure_weight_tying", True),
    )

    # --- Datasets ---
    train_ds = load_jsonl(args.train_data)
    print(f"Train: {len(train_ds)} examples")

    val_ds = None
    if args.val_data and Path(args.val_data).exists():
        val_ds = load_jsonl(args.val_data)
        print(f"Val: {len(val_ds)} examples")

    # --- SFTConfig ---
    output_dir = cfg["output_dir"]
    max_steps = args.max_steps if args.max_steps > 0 else -1

    sft_args = SFTConfig(
        output_dir=output_dir,
        max_length=cfg.get("max_length", 4096),
        num_train_epochs=cfg["num_epochs"] if max_steps == -1 else 1,
        max_steps=max_steps,
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        optim="adamw_torch_fused",
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        max_grad_norm=cfg.get("max_grad_norm", 0.3),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        weight_decay=cfg.get("weight_decay", 0.01),
        bf16=cfg.get("bf16", True),
        fp16=cfg.get("fp16", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_total_limit=cfg.get("save_total_limit", 2),
        eval_strategy=cfg.get("eval_strategy", "epoch") if val_ds else "no",
        seed=cfg.get("seed", 42),
        report_to=cfg.get("report_to", "tensorboard"),
        logging_dir=str(Path(output_dir) / "runs"),
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": True},
    )

    # --- Train ---
    trainer = SFTTrainer(
        model=model, args=sft_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        peft_config=peft_config, processing_class=tokenizer,
    )

    print(f"\nStarting training: {len(train_ds)} examples, {cfg['num_epochs']} epoch(s)")
    print(f"Effective batch: {cfg['batch_size'] * cfg['gradient_accumulation_steps']}")
    print(f"Output: {output_dir}")
    if max_steps > 0:
        print(f"SMOKE TEST: max_steps={max_steps}")

    trainer.train()

    final_path = Path(output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nSaved to: {final_path}")


if __name__ == "__main__":
    main()

"""QLoRA workflow script for the "Price Is Right" fine-tuning project.

This script consolidates the notebook steps into a single CLI with three stages:

1) prepare: build prompts + push dataset to Hugging Face
2) train: fine-tune with QLoRA using TRL's SFTTrainer
3) eval: evaluate a fine-tuned model on the test split

Example usage:
  python app.py prepare --lite
  python app.py train --lite --hf-user <your-hf-user>
  python app.py eval --run-name 2025-11-30_15.10.55-lite

Environment variables:
  HF_TOKEN (required)
  WANDB_API_KEY (optional, only for training if --log-wandb)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)


# Default configuration mirrors the notebooks for the "Price Is Right" project.
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B"
DEFAULT_PROJECT_NAME = "price"
DEFAULT_DATA_USER = "ed-donner"


@dataclass
class RunConfig:
    """Shared config bundle passed through each pipeline stage."""

    base_model: str = DEFAULT_BASE_MODEL
    project_name: str = DEFAULT_PROJECT_NAME
    data_user: str = DEFAULT_DATA_USER
    lite_mode: bool = False


def get_env(name: str, required: bool = False) -> Optional[str]:
    """Fetch an environment variable and optionally enforce its presence."""
    value = os.getenv(name)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def login_huggingface() -> None:
    """Authenticate to Hugging Face using the HF_TOKEN env var."""
    token = get_env("HF_TOKEN", required=True)
    login(token, add_to_git_credential=True)


def detect_bf16_support() -> bool:
    """Return True if the GPU supports bf16 (A100+), else False."""
    capability = torch.cuda.get_device_capability()
    return capability[0] >= 8


def build_quant_config(use_4bit: bool, use_bf16: bool) -> BitsAndBytesConfig:
    """Build a BitsAndBytesConfig for 4-bit or 8-bit quantization."""
    if use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    return BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )


def prepare_prompts(config: RunConfig, cutoff: int, hf_dataset_user: str) -> None:
    """Prepare prompt/completion datasets and push to the HF hub.

    This mirrors the "Prompt Data and Base Model" notebook:
    - Load the Item dataset
    - Tokenize summaries and build prompt/completion pairs
    - Push the prompt dataset to Hugging Face
    """
    from pricer.items import Item

    login_huggingface()

    # Base dataset (raw item descriptions/prices)
    dataset = (
        f"{config.data_user}/items_lite"
        if config.lite_mode
        else f"{config.data_user}/items_full"
    )
    train, val, test = Item.from_hub(dataset)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    # Build prompts for training/validation (include completion),
    # and prompts for test (no completion).
    for item in train + val:
        item.make_prompts(tokenizer, cutoff, True)
    for item in test:
        item.make_prompts(tokenizer, cutoff, False)

    # Push prompt dataset to the hub for training/eval.
    target_dataset = (
        f"{hf_dataset_user}/items_prompts_lite"
        if config.lite_mode
        else f"{hf_dataset_user}/items_prompts_full"
    )
    Item.push_prompts_to_hub(target_dataset, train, val, test)
    print(f"Uploaded prompts to: {target_dataset}")


def train_model(
    config: RunConfig,
    hf_user: str,
    log_wandb: bool,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    """Fine-tune with QLoRA and push to HF hub.

    This mirrors the "Train" notebook:
    - Load the prompt dataset
    - Configure quantization and LoRA adapters
    - Train with TRL's SFTTrainer
    - Push checkpoints + final model to Hugging Face
    """
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    login_huggingface()

    # Optional Weights & Biases logging for training runs.
    if log_wandb:
        import wandb

        wandb_api_key = get_env("WANDB_API_KEY", required=True)
        os.environ["WANDB_API_KEY"] = wandb_api_key
        os.environ["WANDB_PROJECT"] = config.project_name
        os.environ["WANDB_LOG_MODEL"] = "false"
        os.environ["WANDB_WATCH"] = "false"
        wandb.login()

    # Create a unique run name (and match the notebook naming convention).
    run_name = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
    if config.lite_mode:
        run_name += "-lite"
    project_run_name = f"{config.project_name}-{run_name}"
    hub_model_name = f"{hf_user}/{project_run_name}"

    if log_wandb:
        import wandb

        wandb.init(project=config.project_name, name=run_name)

    dataset_name = (
        f"{config.data_user}/items_prompts_lite"
        if config.lite_mode
        else f"{config.data_user}/items_prompts_full"
    )
    dataset = load_dataset(dataset_name)
    train = dataset["train"]
    val_size = 500 if config.lite_mode else 1000
    val = dataset["val"].select(range(val_size))

    # Build the quantized base model (QLoRA relies on 4-bit quantization).
    use_bf16 = detect_bf16_support()
    quant_config = build_quant_config(use_4bit=True, use_bf16=use_bf16)

    # Tokenizer setup mirrors the notebook (pad token = eos token).
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    # LoRA hyperparameters: smaller ranks for lite, larger for full dataset.
    lora_r = 32 if config.lite_mode else 256
    attention_layers = ["q_proj", "v_proj", "k_proj", "o_proj"]
    mlp_layers = ["gate_proj", "up_proj", "down_proj"]
    target_modules = attention_layers if config.lite_mode else attention_layers + mlp_layers

    lora_parameters = LoraConfig(
        lora_alpha=lora_r * 2,
        lora_dropout=0.1,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # Training configuration mirrors the notebook defaults.
    train_parameters = SFTConfig(
        output_dir=project_run_name,
        num_train_epochs=epochs or (1 if config.lite_mode else 3),
        per_device_train_batch_size=batch_size or (32 if config.lite_mode else 256),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=100 if config.lite_mode else 200,
        save_total_limit=10,
        logging_steps=5 if config.lite_mode else 10,
        learning_rate=1e-4,
        weight_decay=0.001,
        fp16=not use_bf16,
        bf16=use_bf16,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.01,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="wandb" if log_wandb else None,
        run_name=run_name,
        max_length=128,
        save_strategy="steps",
        hub_strategy="every_save",
        push_to_hub=True,
        hub_model_id=hub_model_name,
        hub_private_repo=True,
        eval_strategy="steps",
        eval_steps=100 if config.lite_mode else 200,
    )

    if seed is not None:
        set_seed(seed)

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train,
        eval_dataset=val,
        peft_config=lora_parameters,
        args=train_parameters,
    )
    trainer.train()
    trainer.model.push_to_hub(project_run_name, private=True)
    print(f"Saved to the hub: {project_run_name}")

    if log_wandb:
        import wandb

        wandb.finish()


def evaluate_model(
    config: RunConfig,
    hf_user: str,
    run_name: str,
    revision: Optional[str],
    seed: Optional[int],
) -> None:
    """Evaluate a fine-tuned model against the test split.

    Mirrors the "Eval" notebook:
    - Load the quantized base model + LoRA adapters
    - Generate predictions on test data
    - Use util.evaluate to compute error metrics + plots
    """
    from datasets import load_dataset
    from peft import PeftModel
    from util import evaluate

    login_huggingface()

    dataset_name = (
        f"{config.data_user}/items_prompts_lite"
        if config.lite_mode
        else f"{config.data_user}/items_prompts_full"
    )
    dataset = load_dataset(dataset_name)
    test = dataset["test"]

    # Load the quantized base model and attach the LoRA adapters.
    use_bf16 = detect_bf16_support()
    quant_config = build_quant_config(use_4bit=True, use_bf16=use_bf16)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    hub_model_name = f"{hf_user}/{config.project_name}-{run_name}"
    if revision:
        fine_tuned_model = PeftModel.from_pretrained(base_model, hub_model_name, revision=revision)
    else:
        fine_tuned_model = PeftModel.from_pretrained(base_model, hub_model_name)

    def model_predict(item):
        """Generate a price prediction for a single test item."""
        inputs = tokenizer(item["prompt"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_ids = fine_tuned_model.generate(**inputs, max_new_tokens=8)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:]
        return tokenizer.decode(generated_ids)

    if seed is not None:
        set_seed(seed)

    evaluate(model_predict, test)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI for the QLoRA workflow."""
    parser = argparse.ArgumentParser(description="QLoRA workflow for price prediction.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--project-name", default=DEFAULT_PROJECT_NAME)
    parser.add_argument("--data-user", default=DEFAULT_DATA_USER)
    parser.add_argument("--lite", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare prompt datasets.")
    prepare.add_argument("--cutoff", type=int, default=110)
    prepare.add_argument("--hf-dataset-user", default=DEFAULT_DATA_USER)

    train = subparsers.add_parser("train", help="Fine-tune with QLoRA.")
    train.add_argument("--hf-user", required=True)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--log-wandb", action="store_true")
    train.add_argument("--seed", type=int)

    evaluate = subparsers.add_parser("eval", help="Evaluate a fine-tuned model.")
    evaluate.add_argument("--hf-user", required=True)
    evaluate.add_argument("--run-name", required=True)
    evaluate.add_argument("--revision")
    evaluate.add_argument("--seed", type=int, default=42)

    return parser


def main() -> None:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args()

    # Shared config used across all stages.
    config = RunConfig(
        base_model=args.base_model,
        project_name=args.project_name,
        data_user=args.data_user,
        lite_mode=args.lite,
    )

    if args.command == "prepare":
        prepare_prompts(config, cutoff=args.cutoff, hf_dataset_user=args.hf_dataset_user)
    elif args.command == "train":
        train_model(
            config,
            hf_user=args.hf_user,
            log_wandb=args.log_wandb,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    elif args.command == "eval":
        evaluate_model(
            config,
            hf_user=args.hf_user,
            run_name=args.run_name,
            revision=args.revision,
            seed=args.seed,
        )
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
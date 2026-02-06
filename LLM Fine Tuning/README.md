# LLM Fine Tuning — QLoRA Workflow (`app.py`)

This README explains **every section of `app.py`**, the CLI script that merges the
QLoRA notebooks into a single workflow for the “Price Is Right” project.

The script provides three stages:

1. **prepare** — build prompt/completion datasets and push to Hugging Face.
2. **train** — fine-tune the base model using QLoRA (TRL `SFTTrainer`).
3. **eval** — evaluate the fine-tuned model on the test split.

---

## Table of Contents

- [Overview](#overview)
- [Environment Variables](#environment-variables)
- [CLI Commands](#cli-commands)
- [Code Walkthrough](#code-walkthrough)
  - [Module Docstring](#module-docstring)
  - [Imports](#imports)
  - [Defaults](#defaults)
  - [`RunConfig` dataclass](#runconfig-dataclass)
  - [`get_env`](#get_env)
  - [`login_huggingface`](#login_huggingface)
  - [`detect_bf16_support`](#detect_bf16_support)
  - [`build_quant_config`](#build_quant_config)
  - [`prepare_prompts`](#prepare_prompts)
  - [`train_model`](#train_model)
  - [`evaluate_model`](#evaluate_model)
  - [`build_parser`](#build_parser)
  - [`main`](#main)

---

## Overview

`app.py` is a **single entry-point** that follows the notebooks in this folder:

- `Prompt Data and Base Model.ipynb` → data prep + prompt creation
- `Train.ipynb` → QLoRA fine-tuning via TRL
- `Eval.ipynb` → evaluation using `util.evaluate`

It supports **lite mode** (small data + lower LoRA rank) and **full mode**
(larger dataset + higher LoRA rank).

---

## Environment Variables

| Variable | Required | Used By | Description |
|---|---|---|---|
| `HF_TOKEN` | ✅ | prepare/train/eval | Hugging Face access token |
| `WANDB_API_KEY` | Optional | train | Weights & Biases token for logging |

---

## CLI Commands

```bash
# Prepare prompt dataset
python app.py prepare --lite --hf-dataset-user <your-hf-user>

# Train with QLoRA
python app.py train --lite --hf-user <your-hf-user> --log-wandb --seed 42

# Evaluate fine-tuned model
python app.py eval --lite --hf-user <your-hf-user> --run-name 2025-11-30_15.10.55-lite --seed 42
```

---

## Code Walkthrough

### Module Docstring
The top-level docstring explains the **three stages** and shows **usage examples**.
It also lists required environment variables.

### Imports
`app.py` uses:

- **Standard library**: `argparse`, `os`, `dataclasses`, `datetime`, `typing`.
- **PyTorch**: to check GPU capability and run inference.
- **Hugging Face**: `login` + `transformers` for models/tokenizers/quantization.

### Defaults
```python
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B"
DEFAULT_PROJECT_NAME = "price"
DEFAULT_DATA_USER = "ed-donner"
```
These mirror the notebooks so the script behaves the same without extra flags.

### `RunConfig` dataclass
```python
@dataclass
class RunConfig:
    base_model: str
    project_name: str
    data_user: str
    lite_mode: bool
```
Encapsulates shared options that are passed to each pipeline stage.

### `get_env`
```python
def get_env(name: str, required: bool = False) -> Optional[str]:
```
Fetches environment variables and optionally fails fast if required.
Used for `HF_TOKEN` and `WANDB_API_KEY`.

### `login_huggingface`
```python
def login_huggingface() -> None:
```
Logs in using `HF_TOKEN` so model/dataset read/write works.

### `detect_bf16_support`
```python
def detect_bf16_support() -> bool:
```
Checks GPU capability (`>= 8`) to decide whether bf16 is safe
(A100+ supports bf16; T4 does not).

### `build_quant_config`
```python
def build_quant_config(use_4bit: bool, use_bf16: bool) -> BitsAndBytesConfig:
```
Creates the `BitsAndBytesConfig` used for quantization.
By default, **QLoRA uses 4-bit** (nf4) with double quantization.

### `prepare_prompts`
```python
def prepare_prompts(config: RunConfig, cutoff: int, hf_dataset_user: str) -> None:
```
Implements the logic from **Prompt Data and Base Model.ipynb**:

1. Loads the base dataset (`items_lite` or `items_full`).
2. Builds prompt/completion pairs using `Item.make_prompts`.
3. Pushes the prepared dataset (`items_prompts_*`) to Hugging Face.

### `train_model`
```python
def train_model(config: RunConfig, hf_user: str, log_wandb: bool, ...) -> None:
```
Implements the QLoRA training pipeline from **Train.ipynb**:

1. Loads the prompt dataset from HF.
2. Builds a quantized base model (4-bit).
3. Creates LoRA config (`r=32` lite, `r=256` full).
4. Trains with TRL `SFTTrainer`.
5. Optionally sets a reproducibility seed.
6. Pushes checkpoints and final model to the HF Hub.

**Key hyperparameters (mirroring notebooks):**
- `EPOCHS`: 1 (lite) / 3 (full)
- `BATCH_SIZE`: 32 (lite) / 256 (full)
- `LEARNING_RATE`: 1e-4
- `SAVE_STEPS`: 100 (lite) / 200 (full)

### `evaluate_model`
```python
def evaluate_model(config: RunConfig, hf_user: str, run_name: str, revision: Optional[str]) -> None:
```
Implements the evaluation from **Eval.ipynb**:

1. Loads the test split from the prompt dataset.
2. Loads the base model + LoRA adapters via PEFT.
3. Generates predictions with `generate(..., max_new_tokens=8)`.
4. Optionally sets a reproducibility seed (default 42).
5. Calls `util.evaluate` to compute error metrics + charts.

### `build_parser`
```python
def build_parser() -> argparse.ArgumentParser:
```
Creates the CLI interface with three subcommands:

- `prepare`
- `train`
- `eval`

Each subcommand adds only the flags it needs.

### `main`
```python
def main() -> None:
```
Entry point:

1. Parses CLI args.
2. Builds a `RunConfig`.
3. Dispatches to the chosen stage.

---

## Notes & Gotchas

- **GPU required** for training and evaluation.
- **Lite mode** is intended for Colab T4 GPUs.
- **Full mode** usually needs A100-class GPUs.
- Ensure `HF_TOKEN` is set before running any stage.

---

If you want, I can also add a `requirements.txt` or a diagram illustrating the flow.
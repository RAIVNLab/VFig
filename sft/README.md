# VFig SFT Training

Official SFT training code for **VFig: Vectorizing Complex Figures with Vision-Language Models**. This directory contains supervised fine-tuning code for training vision-language models to generate SVG figures from scientific paper images, built on top of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

## Table of Contents

- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)

## Installation

**Requirements:** CUDA 12.4, Python 3.10+

**Option 1 (recommended): restore exact environment**

```bash
cd sft/LLaMA-Factory
conda env create -f environment.yml
conda activate llama
```

**Option 2: manual install**

```bash
cd sft/LLaMA-Factory
conda create -n llama python=3.10
conda activate llama
pip install -e ".[torch,metrics]"
```

## Training

We provide training configs for **Qwen2.5-VL**, **InternVL3**, and **Qwen3-VL** under `LLaMA-Factory/examples/train_lora/`. Training follows a two-stage pipeline:

- **Stage 1** — warm-up training on a subset
- **Stage 2** — full training on all data

### Run manually

```bash
cd sft/LLaMA-Factory
conda activate llama

# Qwen2.5-VL (Stage 1 → Stage 2)
llamafactory-cli train examples/train_lora/qwen2.5vl_lora_sft_v2_1_stage1.yaml
llamafactory-cli train examples/train_lora/qwen2.5vl_lora_sft_v2_1_stage2.yaml

# InternVL3 (Stage 1 → Stage 2)
llamafactory-cli train examples/train_lora/internvl3_lora_sft_v2_1_stage1.yaml
llamafactory-cli train examples/train_lora/internvl3_lora_sft_v2_1_stage2.yaml

# Qwen3-VL (Stage 1 → Stage 2)
llamafactory-cli train examples/train_lora/qwen3vl_lora_sft_v2_1_stage1.yaml
llamafactory-cli train examples/train_lora/qwen3vl_lora_sft_v2_1_stage2.yaml
```

### Run on SLURM

```bash
cd sft/LLaMA-Factory

# Qwen2.5-VL
sbatch jobs/qwen2.5vl_lora_sft_v2_1_stage1.sh
sbatch jobs/qwen2.5vl_lora_sft_v2_1_stage2.sh

# InternVL3
sbatch jobs/internvl3_lora_sft_v2_1_stage1.sh
sbatch jobs/internvl3_lora_sft_v2_1_stage2.sh

# Qwen3-VL
sbatch jobs/qwen3vl_lora_sft_v2_1_stage1.sh
sbatch jobs/qwen3vl_lora_sft_v2_1_stage2.sh
```

> Note: Job scripts are configured for SLURM. Update `--account`, `--partition`, `--gres`, conda path, and CUDA module for your cluster before running.

Checkpoints are saved to `LLaMA-Factory/output/`.

## Inference

Run inference with a fine-tuned checkpoint using the scripts under `inference/`:

```bash
cd sft/inference

# Qwen2.5-VL
python eval_qwen2.5_model_sft.py
# or via SLURM:
sbatch eval_qwen2.5_model_sft.sh

# InternVL3
python eval_intern3.5_model_sft.py
# or via SLURM:
sbatch eval_intern3.5_model_sft.sh
```

## Evaluation

Evaluate generated SVGs using the scripts under `eval/`:

```bash
cd eval
export GEMINI_API_KEY="your_gemini_api_key"
export OPENAI_API_KEY="your_openai_api_key"

# Score SVG quality using Gemini and GPT judges
python eval_metrics_gemini_gpt_white.py

# Compute SVG code cleanliness metric
python code_cleanliness.py
```

## Acknowledgements

This project builds on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We thank the authors for their excellent work.
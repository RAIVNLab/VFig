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

## Data Preparation

We provide `data/prepare_fine_tune_data.py` to convert raw SVG files into a LLaMA-Factory compatible training dataset.

**What it does:**
1. Reads `.svg` files from a source directory
2. Renders each SVG to a PNG image (via `cairosvg`)
3. Pairs each image with the original SVG code and a randomly sampled instruction prompt
4. Outputs a JSON file in LLaMA-Factory's ShareGPT multimodal format

**Usage:**

```bash
# Edit the CONFIG section at the top of the script:
#   SVG_DIR  — path to your SVG source directory
#   OUTPUT_JSON — output filename (e.g. finetune_data_arxiv.json)
#   RENDER_DIR  — directory to save rendered PNGs

python data/prepare_fine_tune_data.py
```

**After running**, place the output JSON and rendered images under `LLaMA-Factory/data/`, then register the dataset in `LLaMA-Factory/data/dataset_info.json`.

Then reference `my_dataset` in your training YAML under `dataset:`.

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

# Qwen3-VL-4B
python eval_qwen_model_sft.py
# or via SLURM:
sbatch eval_qwen_model_sft.sh
```

## Acknowledgements

This project builds on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We thank the authors for their excellent work.
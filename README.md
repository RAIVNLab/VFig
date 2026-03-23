# VFig-RL: Reinforcement Learning for Scientific Figure Generation

This repository contains the RL training code for **VFig**, a vision-language model trained to generate SVG figures from scientific paper images using GRPO-based reinforcement learning.

## Repository Structure

```
Vfig_RL/
├── data/                          # Training data (parquet format)
│   └── combined_axiv_molmo_star/
│       ├── train.parquet
│       └── test.parquet
├── data_preprocessing/
│   └── create_parquet.py          # Converts SFT JSON annotations to RL parquet format
├── rewards/
│   └── reward_full_gemini.py      # Gemini-based visual reward function
└── scripts/
    └── train_4b_2stage_full_gemini.sh  # GRPO training launch script
```

---

## Installation

### 1. Create Conda Environment

```bash
conda create -n vfig_rl python==3.12
conda activate vfig_rl
```

### 2. Install verl

verl is the RL training framework used for GRPO. We use FSDP (no Megatron required). See the [official installation guide](https://verl.readthedocs.io/en/latest/start/install.html) for full details.

```bash
git clone https://github.com/volcengine/verl.git
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
cd ..
```

> **Note:** `USE_MEGATRON=0` skips Megatron-LM installation. If you want Megatron support, run `bash scripts/install_vllm_sglang_mcore.sh` without the flag.
> Requires **Python >= 3.10** and **CUDA >= 12.8**.

#### Known Fix: Qwen3-VL + LoRA with vLLM

If you encounter errors when using Qwen3-VL with LoRA ([verl issue #3922](https://github.com/verl-project/verl/issues/3922)), apply this fix to your vLLM installation:

In `vllm/model_executor/models/qwen3_vl.py`, find `get_mm_mapping()` and change:

```python
# Before
connector="model.visual.merger",
tower_model="model.visual.",

# After
connector="visual.merger",
tower_model="visual.",
```

### 3. Install Additional Dependencies

```bash
pip install google-genai cairosvg func_timeout
pip install Pillow numpy opencv-python transformers
pip install datasets tqdm
```

### 4. Set API Keys

The reward function calls the Gemini API. Export your key before training:

```bash
export GEMINI_API_KEY=<your_gemini_api_key>
export WANDB_API_KEY=<your_wandb_api_key>   # optional, for logging
```

---

## Data Preparation

Edit `data_preprocessing/create_parquet.py` to set `IMAGE_ROOT` and `SOURCE_FILES` to point to your image directory and annotation JSON files, then run:

```bash
python data_preprocessing/create_parquet.py --output_dir data/combined_axiv_molmo_star
```

This produces `train.parquet` and `test.parquet` with a 90/10 split.

---

## Training

Edit `scripts/train_4b_2stage_full_gemini.sh` to fill in the placeholder paths:

| Placeholder | Description |
|---|---|
| `/path/to/Vfig_RL` | Root of this repository |
| `/path/to/miniconda/bin/activate` | Your conda installation |
| `/path/to/your/sft_model_checkpoint` | SFT model to initialize RL from |
| `/path/to/checkpoints/vfig_rl` | Where to save RL checkpoints |
| `/path/to/.caches` | Cache directory for HF, vLLM, etc. |

Then launch:

```bash
bash scripts/train_4b_2stage_full_gemini.sh
```

The script runs GRPO training with:
- **Model:** Qwen3-VL 4B with LoRA (rank 64, vision tower frozen)
- **Reward:** Gemini visual similarity scored on 4 axes (presence, layout, connectivity, details)
- **Rollout:** vLLM with tensor parallelism 2, 8 samples per prompt
- **Hardware:** 8× A100/L40S GPUs (1 node)

---

## Reward Function

`rewards/reward_full_gemini.py` implements the `compute_score` function expected by verl's custom reward interface. For each rollout:

1. Extracts the SVG from the model's output
2. Renders it to PNG via CairoSVG
3. Sends the rendered PNG and ground-truth PNG to Gemini for scoring
4. Returns a weighted average of presence (0.25), layout (0.25), connectivity (0.25), and details (0.25)


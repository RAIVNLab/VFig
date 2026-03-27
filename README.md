# VFig: Vectorizing Complex Figures with Vision-Language Models

This repository contains the official training code for **VFig**, a vision-language model trained to generate SVG figures from scientific paper images.

## Repository Structure

```
VFig/
├── rl/         # GRPO-based reinforcement learning pipeline for reward-driven SVG generation
├── sft/        # Supervised fine-tuning pipeline built on LLaMA-Factory (Qwen2.5-VL, InternVL3, Qwen3-VL)
├── inference/  # Quick inference script for single-image SVG generation using the pretrained model
└── eval/       # Benchmark evaluation scripts using Gemini/GPT judges and code cleanliness metrics
```

## Getting Started

- For SFT training, see [sft/README.md](sft/README.md).
- For RL training, see [rl/README.md](rl/README.md).

## Inference

There are two inference paths depending on your use case:

**1. Quick inference on a single image** — use [`inference/inference.py`](inference/inference.py) with the pretrained [VFIG-4B](https://huggingface.co/XunmeiLiu/VFIG-4B) model directly from HuggingFace. No training setup required.

**Installation:**

```bash
pip install transformers torch Pillow
```

```bash
cd inference

# Run on default example image
python inference.py

# Run on a custom image
python inference.py images/your_figure.png

# Save output to SVG file
python inference.py images/your_figure.png --output output.svg

# Use a local model checkpoint
python inference.py images/your_figure.png --model /path/to/model
```

**2. Batch inference on a dataset** — if you have trained your own checkpoint using the SFT/RL pipeline, use the scripts under [`sft/inference/`](sft/inference/) to run batch inference over a full dataset and save all generated SVGs. See [sft/README.md](sft/README.md) for details.

## Evaluation

[`eval/`](eval/) contains scripts for benchmarking SVG generation quality on the full VFig test set. This is intended for evaluating model outputs at scale, not individual images.

- **`eval_metrics_gemini_gpt_white.py`** — scores generated SVGs using Gemini and GPT as judges, measuring visual similarity to the reference figure
- **`code_cleanliness.py`** — measures SVG code quality (structure, validity, redundancy)

```bash
cd eval
export GEMINI_API_KEY="your_gemini_api_key"
export OPENAI_API_KEY="your_openai_api_key"

# Score SVG quality using Gemini and GPT judges
python eval_metrics_gemini_gpt_white.py

# Compute SVG code cleanliness metric
python code_cleanliness.py
```

## Citation
```
@misc{he2026vfigvectorizingcomplexfigures,
      title={VFIG: Vectorizing Complex Figures in SVG with Vision-Language Models}, 
      author={Qijia He and Xunmei Liu and Hammaad Memon and Ziang Li and Zixian Ma and Jaemin Cho and Jason Ren and Daniel S Weld and Ranjay Krishna},
      year={2026},
      eprint={2603.24575},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.24575}, 
}
```

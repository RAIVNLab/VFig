#!/bin/bash
#SBATCH --job-name=<job_name>
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --gres=<gpu_type>:<num_gpus>
#SBATCH --cpus-per-task=<num_cpus>
#SBATCH --mem=<memory>
#SBATCH --time=<time_limit>
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --mail-user=<your_email>
#SBATCH --mail-type=ALL

# ---- Environment setup ----
module purge
module load cuda/12.4.1  # ✅ valid CUDA version on Hyak

source $(conda info --base)/etc/profile.d/conda.sh
conda activate llama

echo "Activated conda env: $CONDA_DEFAULT_ENV"
which python
which llamafactory-cli

cd sft/LLaMA-Factory
# ---- Launch fine-tuning ----
llamafactory-cli train examples/train_lora/qwen2.5vl_lora_sft_v2_1_stage2.yaml

#!/bin/bash
#SBATCH --job-name=<job_name>
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --gres=<gpu_type>:<num_gpus>
#SBATCH --cpus-per-task=<num_cpus>
#SBATCH --mem=<memory>
#SBATCH --time=<time_limit>

#SBATCH --array=0-9%10
#SBATCH --output=logs/eval_qwen3vl/%x_%A_%a.out
#SBATCH --error=logs/eval_qwen3vl/%x_%A_%a.err
#SBATCH --mail-user=<your_email>
#SBATCH --mail-type=ALL

module purge
module load cuda/12.4.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llama

echo "Python:" $(which python)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi

cd sft/LLaMA-Factory/output || exit 1

# Keep sharding consistent with the Slurm array size.
# SLURM_ARRAY_TASK_COUNT is set for array jobs; fall back to 1 when not present.
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID}

echo "=============================="
echo "Shard ${TASK_ID} / ${NUM_JOBS}"
echo "=============================="

# -u ensures logs show up immediately in Slurm output (no buffering).
python -u eval_qwen2.5_model_sft.py \
  --num_shards ${NUM_JOBS} \
  --shard_id ${TASK_ID}

echo "Shard ${TASK_ID} finished."

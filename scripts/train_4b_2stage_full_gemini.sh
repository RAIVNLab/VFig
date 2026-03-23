#!/bin/bash
echo "Job running on node: $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
nvidia-smi

module load cuda/12.8.1
module load gcc/11.2.0

cd /path/to/Vfig_RL

# Activate conda environment
source /path/to/miniconda/bin/activate vfig_rl

# Environment variables
export WANDB_API_KEY=<your_wandb_api_key>
export GEMINI_API_KEY=<your_gemini_api_key>
export ENGINE=vllm

unset ROCR_VISIBLE_DEVICES
export TORCH_CUDA_ARCH_LIST="8.9"

# Cache paths
export CACHE_BASE=/path/to/.caches
export XDG_CACHE_HOME=$CACHE_BASE/xdg
export FLASHINFER_CACHE_DIR=$CACHE_BASE/flashinfer
export TRITON_CACHE_DIR=$CACHE_BASE/triton
export TORCH_HOME=$CACHE_BASE/torch
export HF_HOME=$CACHE_BASE/huggingface
export RAY_TMPDIR=$CACHE_BASE/ray_tmp
export WANDB_DIR=$CACHE_BASE/wandb
export TMPDIR=/path/to/tmp

# Ray & NCCL stability settings
export RAY_health_check_timeout_s=1200
export NCCL_TIMEOUT=7000
export NCCL_ASYNC_ERROR_HANDLING=1

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/combined_axiv_molmo_star/train.parquet \
    data.val_files=data/combined_axiv_molmo_star/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.val_max_samples=50 \
    data.max_prompt_length=9000 \
    data.max_response_length=8500 \
    actor_rollout_ref.rollout.max_model_len=17500 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=/path/to/your/sft_model_checkpoint \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.exclude_modules='.*visual.*' \
    actor_rollout_ref.actor.optim.lr=9e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='vfig_rl' \
    trainer.experiment_name='qwen3vl_4b_2stage' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=500 \
    trainer.total_epochs=1 \
    trainer.resume_mode="disable" \
    trainer.default_local_dir=/path/to/checkpoints/vfig_rl \
    trainer.rollout_data_dir=/path/to/rollout_responses \
    trainer.log_val_generations=40 \
    trainer.validation_data_dir=/path/to/val_responses \
    custom_reward_function.path=rewards/reward_full_gemini.py

#!/bin/bash
#SBATCH --job-name=weibo-debate
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/yxfeng/cscwd27_loopfuse/logs/%x-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_loopfuse/logs/%x-%j.err

# Sample every Weibo item for iter0_real, then iter0_fake, then the two rebuttals.
# XLM-RoBERTa-large then writes (N, 4, 1024) pooled CLS features.
set -euo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs /data3/yxfeng/cscwd27_loopfuse/weibo_debate

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com

PY=/home/yxfeng/miniconda3/envs/verl1/bin/python
"$PY" scripts/sample_weibo_debate.py
"$PY" scripts/encode_debate_hidden.py

#!/bin/bash
#SBATCH --job-name=weibo-loop2-debate
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --nodelist=node5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:4
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/yxfeng/cscwd27_loopfuse/logs/%x-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_loopfuse/logs/%x-%j.err

# Looped Chinese CLIP (2 loops) with the debate branch enabled.
# Main branch: both towers looped twice, pretrained weights, Fuse over 4 slots.
# Debate branch: 4 offline debate turns pooled by XLM-R-large, (B, 4, 1024),
# fused by a second Fuse and concatenated with the CLIP feature.
set -uo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
# Single-node 4090s. Peer access stays on; disabling it made the
# allreduce at step 35 of epoch 0 time out. Workers stay at 0 because
# forking DataLoader workers after NCCL init deadlocks the same step.
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
NPROC="${NPROC:-4}"
SEEDS="${SEEDS:-0 1 2}"
OUT="${OUT:-outputs/weibo_cnclip_loop2_debate}"
LOOPS="${LOOPS:-2}"
LR="${LR:-2e-5}"
MIN_LR="${MIN_LR:-2e-6}"
# Batch 32 filled the 24GB 4090s (one card sat at 24.0GB and blocked
# in the allocator). 8 x accum 4 keeps the global batch at 128.
MICRO="${MICRO:-8}"
ACCUM="${ACCUM:-4}"
SUMMARY="logs/weibo_loop2_debate.log"

echo "START $(date) host=$(hostname) loops=${LOOPS} seeds=${SEEDS} lr=${LR} micro=${MICRO} global_batch=$((MICRO * NPROC * ACCUM)) debate=on" | tee "$SUMMARY"

for seed in ${SEEDS}; do
  log="logs/weibo_loop2_debate_seed${seed}.log"
  echo "===== loop=${LOOPS} seed=${seed} $(date) =====" | tee -a "$SUMMARY"
  rm -rf "${OUT}/seed${seed}"
  "$PY" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT:-29891}" \
    trainer.py --config=weibo_cnclip_loop2_debate \
    output_dir="${OUT}/seed${seed}" \
    run_name="weibo_loop2_debate_seed${seed}" \
    num_loops="${LOOPS}" epochs=50 patience=5 max_text_length=512 \
    batch_size="${MICRO}" grad_accum_steps="${ACCUM}" num_workers=0 \
    model_type=chinese backbone=chinese debate_dim=1024 d_model=512 \
    freeze_backbone=false lr="${LR}" min_lr="${MIN_LR}" weight_decay=1e-4 \
    seed="${seed}" > "${log}" 2>&1
  status=$?
  if [ "${status}" -ne 0 ]; then
    echo "FAIL loop=${LOOPS} seed=${seed} exit=${status}" | tee -a "$SUMMARY"
    tail -n 20 "${log}" | tee -a "$SUMMARY"
    exit 1
  fi
  acc=$(grep -o 'test accuracy=[0-9.]*' "${log}" | tail -1 | cut -d= -f2)
  f1=$(grep -o 'macro_f1=[0-9.]*' "${log}" | tail -1 | cut -d= -f2)
  echo "OK loop=${LOOPS} seed=${seed} test_acc=${acc} macro_f1=${f1}" | tee -a "$SUMMARY"
  echo "${seed},${acc},${f1}" >> "${OUT}/seeds.csv"
done

echo "ALL_DONE $(date)" | tee -a "$SUMMARY"

#!/bin/bash
#SBATCH --job-name=weibo-cnclip-loop2
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

# Chinese CLIP looped twice on Weibo, starting from the pretrained weights.
# No mid-training checkpoint and no debate branch: both towers simply run twice
# with shared weights, then Fuse and classify.
# 4 GPUs, global batch 32 x 4 = 128, lr 2e-5 cosine to 2e-6, 50 epochs, patience 5.
set -uo pipefail
cd /home/yxfeng/cscwd27_loopfuse
mkdir -p logs

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/yxfeng/miniconda3/envs/www27/bin/python
NPROC="${NPROC:-4}"
SEEDS="${SEEDS:-0 1 2}"
OUT="${OUT:-outputs/weibo_cnclip_loop2}"
LOOPS="${LOOPS:-2}"
LR="${LR:-2e-5}"
MIN_LR="${MIN_LR:-2e-6}"
MICRO="${MICRO:-32}"
ACCUM="${ACCUM:-1}"
SUMMARY="logs/weibo_cnclip_loop2.log"

echo "START $(date) host=$(hostname) loops=${LOOPS} seeds=${SEEDS} nproc=${NPROC} lr=${LR} micro=${MICRO} global_batch=$((MICRO * NPROC * ACCUM))" | tee "$SUMMARY"

for seed in ${SEEDS}; do
  log="logs/weibo_cnclip_loop2_seed${seed}.log"
  echo "===== loop=${LOOPS} seed=${seed} $(date) =====" | tee -a "$SUMMARY"
  # A fresh run each time: delete any partial save from an earlier attempt.
  rm -rf "${OUT}/seed${seed}"
  "$PY" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT:-29881}" \
    trainer.py --config=weibo_cnclip_loop2 \
    output_dir="${OUT}/seed${seed}" \
    run_name="weibo_cnclip_loop2_seed${seed}" \
    num_loops="${LOOPS}" epochs=50 patience=5 max_text_length=512 \
    batch_size="${MICRO}" grad_accum_steps="${ACCUM}" \
    model_type=chinese backbone=chinese debate_dim=0 d_model=512 \
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

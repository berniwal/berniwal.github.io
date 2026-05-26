#!/usr/bin/env bash
# Runs ON a RunPod GPU pod (via launch.py --gpu --cmd) to train the PyTorch GRPO LLM
# proposer (Layer 1). Unlike the MLX arm (Apple-only), this runs on NVIDIA. Installs
# the torch extra on top of the sia package, confirms the GPU, runs the driver, and
# leaves results under ./results (synced to gs://.../<exp>/results by the worker).
#
# Env knobs (set in launch.py --cmd):
#   TARGET=medium ARM=risk MODE=quantile ROUNDS=40 BATCH=16 MAXNEW=48 TEMP=1.0
#   LR=1e-6 EPSILON=0.25 MODEL=Qwen/Qwen2.5-0.5B-Instruct EXTRA="--std-normalize"
#   REASONING=1  -> append --reasoning, bump MAXNEW to 2048 unless caller set it.
#                   Use with a thinking-capable model (e.g. MODEL=Qwen/Qwen3-1.7B).
set -uxo pipefail

echo "[l1] nproc=$(nproc)"
nvidia-smi -L || echo "[l1] WARNING: no nvidia-smi (no GPU?)"

# torch built for the host's CUDA: default PyPI torch is a cu130 build that needs a
# newer driver than RunPod hosts provide (12.7) and silently falls back to CPU. Pin
# the cu124 build (driver >=12.4). Then transformers <5 (5.x changed generate
# internals -> KeyError) + peft/accelerate/scipy. Large (~GB); launch with --disk 40.
python3 -m pip install --break-system-packages -q torch --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install --break-system-packages -q "transformers>=4.44,<5" "peft>=0.11" "accelerate>=0.30" scipy 2>&1 | tail -8
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; \
  print('[l1] torch', torch.__version__, 'cuda', torch.cuda.get_device_name(0))" \
  || { echo '[l1] FATAL: torch/CUDA not available'; exit 1; }

REASONING_FLAG=""
if [ "${REASONING:-0}" = "1" ]; then
  REASONING_FLAG="--reasoning"
  MAXNEW="${MAXNEW:-2048}"   # CoT traces need room; only set if caller did not.
fi

for SEED in $(echo "${SEEDS:-0}" | tr ',' ' '); do
  START=$(date +%s)
  python3 run_layer1_torch.py \
      --model "${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" \
      --target "${TARGET:-medium}" --arm "${ARM:-risk}" --mode "${MODE:-quantile}" \
      --rounds "${ROUNDS:-60}" --batch "${BATCH:-32}" --max-new-tokens "${MAXNEW:-48}" \
      --temperature "${TEMP:-1.0}" --lr "${LR:-1e-5}" --epsilon "${EPSILON:-0.25}" \
      --x-range="${XRANGE:--1,1}" --n-points "${NPOINTS:-20}" \
      --seed "$SEED" --out results/layer1-torch ${REASONING_FLAG} ${EXTRA:-} 2>&1 | tail -45
  echo "[l1] ELAPSED=$(( $(date +%s) - START ))s  TARGET=${TARGET:-medium} ARM=${ARM:-risk} MODE=${MODE:-quantile} seed=$SEED reasoning=${REASONING:-0}"
done
echo "[l1] done"

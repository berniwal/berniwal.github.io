#!/usr/bin/env bash
# Runs ON a RunPod GPU pod (via launch.py --gpu --cmd) to train the PyTorch GRPO LLM
# proposer (Layer 1). Unlike the MLX arm (Apple-only), this runs on NVIDIA. Installs
# the torch extra on top of the sia package, confirms the GPU, runs the driver, and
# leaves results under ./results (synced to gs://.../<exp>/results by the worker).
#
# Env knobs (set in launch.py --cmd):
#   TARGET=medium ARM=risk MODE=quantile ROUNDS=40 BATCH=16 MAXNEW=48 TEMP=1.0
#   LR=1e-6 EPSILON=0.25 MODEL=Qwen/Qwen2.5-0.5B-Instruct EXTRA="--std-normalize"
set -uxo pipefail

echo "[l1] nproc=$(nproc)"
nvidia-smi -L || echo "[l1] WARNING: no nvidia-smi (no GPU?)"

# torch+CUDA + transformers + peft + accelerate (the layer1-gpu extra). Large (~GB);
# launch with --disk 40.
python3 -m pip install --break-system-packages -q -e ".[layer1-gpu]" 2>&1 | tail -15
python3 -c "import torch; print('[l1] torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
  (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-ONLY'))" \
  || { echo "[l1] FATAL: torch import failed"; exit 1; }

START=$(date +%s)
python3 run_layer1_torch.py \
    --model "${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" \
    --target "${TARGET:-medium}" --arm "${ARM:-risk}" --mode "${MODE:-quantile}" \
    --rounds "${ROUNDS:-40}" --batch "${BATCH:-16}" --max-new-tokens "${MAXNEW:-48}" \
    --temperature "${TEMP:-1.0}" --lr "${LR:-1e-6}" --epsilon "${EPSILON:-0.25}" \
    --out results/layer1-torch ${EXTRA:-} 2>&1 | tail -80
echo "[l1] ELAPSED=$(( $(date +%s) - START ))s  TARGET=${TARGET:-medium} ARM=${ARM:-risk}"
echo "[l1] done"

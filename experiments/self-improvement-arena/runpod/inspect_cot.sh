#!/usr/bin/env bash
# Pod-side: install minimal torch+transformers (Qwen3-capable) and run the CoT
# inspector. Separate file because launch.py's --cmd doesn't survive shell-quoted
# pip constraints like "transformers>=4.44,<5".
set -uxo pipefail
python3 -m pip install --break-system-packages -q torch --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install --break-system-packages -q "transformers>=4.51,<5" "peft>=0.11" "accelerate>=0.30" scipy 2>&1 | tail -8
python3 -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.get_device_name(0), 'tf', transformers.__version__)"
python3 runpod/inspect_cot.py 2>&1 | tee inspect.log

#!/usr/bin/env bash
# Crash-safe, resumable launcher for the long sweeps.
#
#   ./run_overnight.sh             # Layer 0 budget-scaling sweep (default)
#   ./run_overnight.sh layer0      # same
#   ./run_overnight.sh layer1      # Layer 1 LLM evolution sweep
#   ./run_overnight.sh layer1_lora # Layer 1 three-arm sweep (evolution + LoRA)
#
# Safe to re-run: completed runs are detected on disk and skipped, so if the job
# crashes, the machine reboots, or you stop it, just run this again and it
# continues.
#
# IMPORTANT for MLX (Layer 1): `caffeinate -i` alone is NOT enough -- when the
# display sleeps / the Mac is locked, the Metal GPU down-clocks and background
# work runs ~10x slower. We use `caffeinate -dimsu` (-d also prevents DISPLAY
# sleep, which keeps the GPU at full clocks). Also: keep it plugged into power
# and turn OFF Low Power Mode (System Settings > Battery) -- caffeinate can't
# override those. The screen will stay on; just dim the brightness.
cd "$(dirname "$0")" || exit 1
LAYER="${1:-layer0}"

case "$LAYER" in
  layer0)
    OUT=results_scaling; LOG=results_scaling_run.log
    CMD="python3 run_layer0.py --config configs/scaling.yaml --out $OUT --parallel"
    DESC="Layer 0 budget-scaling sweep (300 runs, parallel CPU)";;
  layer1)
    OUT=results_layer1; LOG=results_layer1_run.log
    CMD="python3 run_layer1.py --config configs/layer1.yaml --out $OUT"
    DESC="Layer 1 LLM evolution sweep (serial, MLX/Metal)";;
  layer1_lora)
    # Fresh dir (not results_layer1_lora) so this KL + scale=2 run does NOT mix with
    # the earlier scale=20 / non-KL degenerate medium runs kept there as evidence.
    OUT=results_layer1_lora_kl; LOG=results_layer1_lora_kl_run.log
    CMD="python3 run_layer1.py --config configs/layer1_lora.yaml --out $OUT"
    DESC="Layer 1 four-arm sweep: best_of_n + evolution + greedy/risk LoRA (serial, MLX)";;
  *)
    echo "usage: $0 [layer0|layer1|layer1_lora]"; exit 1;;
esac

mkdir -p "$OUT"
echo "$DESC -> $OUT/   (resumable)"
echo "Console log: $LOG"
caffeinate -dimsu nohup $CMD >> "$LOG" 2>&1 &

echo "Started in background, PID $!"
echo
echo "Monitor progress:   tail -f $LOG"
echo "Count completed:    ls $OUT/logs | wc -l"
echo "Resume after stop:  ./run_overnight.sh $LAYER"

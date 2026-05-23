#!/usr/bin/env bash
# Crash-safe, resumable launcher for the long sweeps.
#
#   ./run_overnight.sh            # Layer 0 budget-scaling sweep (default)
#   ./run_overnight.sh layer0     # same
#   ./run_overnight.sh layer1     # Layer 1 LLM evolution sweep
#
# Safe to re-run: completed runs are detected on disk and skipped, so if the job
# crashes, the machine reboots, or you stop it, just run this again and it
# continues. `caffeinate -i` keeps the Mac awake while it runs (keep it plugged
# in; a closed lid on battery can still sleep -- harmless, just resume).
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
  *)
    echo "usage: $0 [layer0|layer1]"; exit 1;;
esac

mkdir -p "$OUT"
echo "$DESC -> $OUT/   (resumable)"
echo "Console log: $LOG"
caffeinate -i nohup $CMD >> "$LOG" 2>&1 &

echo "Started in background, PID $!"
echo
echo "Monitor progress:   tail -f $LOG"
echo "Count completed:    ls $OUT/logs | wc -l"
echo "Resume after stop:  ./run_overnight.sh $LAYER"

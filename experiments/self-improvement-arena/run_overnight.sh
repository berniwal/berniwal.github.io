#!/usr/bin/env bash
# Crash-safe, resumable overnight budget-scaling sweep.
#
#   ./run_overnight.sh            # start (or resume) the 2M-call, 20-seed sweep
#
# Safe to re-run: completed runs are detected on disk and skipped, so if the job
# crashes, the machine reboots, or you stop it, just run this script again and it
# continues from where it left off. `caffeinate -i` keeps the Mac awake while it
# runs (keep it plugged in; a closed lid on battery can still sleep -- harmless,
# the sweep just resumes next time you run this).
cd "$(dirname "$0")" || exit 1
mkdir -p results_scaling

echo "Budget-scaling sweep -> results_scaling/  (300 runs, ~5-6h, resumable)"
echo "Console log: results_scaling_run.log"
caffeinate -i nohup python3 run_layer0.py \
    --config configs/scaling.yaml --out results_scaling \
    >> results_scaling_run.log 2>&1 &

echo "Started in background, PID $!"
echo
echo "Monitor progress:   tail -f results_scaling_run.log"
echo "Count completed:    ls results_scaling/logs | wc -l    # out of 300"
echo "Resume after crash: ./run_overnight.sh"

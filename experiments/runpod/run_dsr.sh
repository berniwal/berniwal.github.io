#!/usr/bin/env bash
# Runs ON a RunPod pod (via launch.py --cmd) to cross-check DSR's OWN implementation:
# does their vanilla PG (VPG) collapse on Nguyen where our greedy does not?
#
# DSR pins a 2019 stack (tensorflow==1.14, numpy<=1.19, numba==0.53.1, Cython) needing
# Python 3.7 -- impossible on the cloud-sdk image's modern Python -- so we build an
# isolated Miniconda py37 env while the harness's system gsutil handles GCS sync. DSR's
# output is copied into ./results so the worker syncs it to gs://.../<exp>/results/.
#
# Env knobs (set in launch.py --cmd):
#   TASKS=Nguyen-3,Nguyen-4   ARMS=vpg,risk   RUNS=25 (seeds)   NSAMPLES=2000000
#   NCORES=25 (DSR parallel runs)   START_SEED=0
# Calibration smoke: TASKS=Nguyen-3 ARMS=vpg RUNS=1 NSAMPLES=200000 NCORES=1
set -uxo pipefail   # -x traces every step into worker.log

TASKS="${TASKS:-Nguyen-3,Nguyen-4}"
ARMS="${ARMS:-vpg,risk}"
RUNS="${RUNS:-25}"
NSAMPLES="${NSAMPLES:-2000000}"
NCORES="${NCORES:-25}"
START_SEED="${START_SEED:-0}"

# Synced results dir, captured BEFORE any cd (cwd = the harness workdir).
RESULTS_DIR="$PWD/results/dsr-collapse"
mkdir -p "$RESULTS_DIR"
echo "[dsr] nproc=$(nproc)  TASKS=$TASKS ARMS=$ARMS RUNS=$RUNS NSAMPLES=$NSAMPLES NCORES=$NCORES"

apt-get update -qq && apt-get install -y -qq build-essential wget git >/dev/null 2>&1 || true

# 1) Miniconda + isolated Python 3.7 env (TF1.14 has no wheels for modern Python).
if [ ! -x /opt/conda/bin/conda ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p /opt/conda
fi
set +u; source /opt/conda/etc/profile.d/conda.sh
# conda-forge + --override-channels avoids the `defaults`-channel ToS block; ship pip.
conda create -y -n dsr -c conda-forge --override-channels python=3.7 pip
conda activate dsr; set -u
command -v python >/dev/null || { echo "[dsr] FATAL: py37 env not created"; conda info --envs; exit 1; }
python -m pip --version || { echo "[dsr] FATAL: pip missing in env"; exit 1; }

# 2) Clone + install DSR.
rm -rf /workspace/dsr
git clone --depth 1 https://github.com/dso-org/deep-symbolic-optimization.git /workspace/dsr
cd /workspace/dsr/dso
python -m pip install --upgrade "pip<24" "setuptools<60" wheel "cython<3" "numpy<=1.19"
python -m pip install -e . 2>&1 | tail -20
# TF1.14's generated code breaks under protobuf>=3.20 ("Descriptors cannot not be created").
python -m pip install "protobuf<3.20"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python -c "import dso, tensorflow as tf; print('[dsr] import OK: tf', tf.__version__)" \
  || { echo "[dsr] FATAL: DSR/TF import failed"; exit 1; }

# 3) For each arm x task: build the arm's config and run RUNS seeds.
IFS=',' read -ra ARM_LIST <<< "$ARMS"
IFS=',' read -ra TASK_LIST <<< "$TASKS"
for arm in "${ARM_LIST[@]}"; do
  CFG="/workspace/cfg-${arm}.json"
  python - "$arm" "$NSAMPLES" "/workspace/dsrlog/${arm}" "$CFG" <<'PY'
import sys, json, commentjson
arm, nsamp, logdir, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
cfg = commentjson.load(open("dso/config/config_regression.json"))
tr = cfg.setdefault("training", {})
if arm in ("vpg", "vpg_nops"):  # vanilla PG: no quantile filter; EWMA-of-mean baseline
    tr["epsilon"] = None         # (default baseline "R_e" sets b=quantile -> crashes for VPG)
    tr["baseline"] = "ewma_R"
    if arm == "vpg_nops":        # ABLATION: drop parent/sibling tree obs, observe prev-action
        sm = commentjson.load(open("dso/config/config_common.json")).get("state_manager", {})
        sm["observe_parent"] = False
        sm["observe_sibling"] = False
        sm["observe_action"] = True   # must observe something (their assert)
        cfg["state_manager"] = sm
else:                            # risk-seeking (DSR): top-5% filter, quantile baseline
    tr["epsilon"] = 0.05
tr["n_samples"] = nsamp
cfg.setdefault("experiment", {})["logdir"] = logdir
json.dump(cfg, open(out, "w"), indent=1)
sm = cfg.get("state_manager", {})
print("[dsr] %s config: epsilon=%r baseline=%s n_samples=%d parent=%s sibling=%s action=%s"
      % (arm, tr["epsilon"], tr.get("baseline", "R_e"), nsamp,
         sm.get("observe_parent", "default"), sm.get("observe_sibling", "default"),
         sm.get("observe_action", "default")))
PY
  for task in "${TASK_LIST[@]}"; do
    START=$(date +%s)
    python -m dso.run "$CFG" --b "$task" --runs "$RUNS" --seed "$START_SEED" \
        --n_cores_task "$NCORES" 2>&1 | tail -15
    echo "[dsr] ELAPSED=$(( $(date +%s) - START ))s  arm=$arm task=$task runs=$RUNS n=$NSAMPLES"
  done
  mkdir -p "$RESULTS_DIR/${arm}"
  cp -r /workspace/dsrlog/${arm}/* "$RESULTS_DIR/${arm}/" 2>/dev/null || true
done

# 4) Summarize recovery (success = exact symbolic, DSR's criterion) into worker.log.
python - "$RESULTS_DIR" <<'PY'
import glob, csv, os, sys
root = sys.argv[1]
print("\n===== DSR recovery: success=symbolic (rows=seeds) =====")
for f in sorted(glob.glob(os.path.join(root, "**", "summary.csv"), recursive=True)):
    rows = list(csv.DictReader(open(f)))
    if not rows:
        continue
    n = len(rows)
    succ = sum(str(r.get("success")).strip().lower() in ("true", "1") for r in rows)
    tag = f.replace(root, "").strip("/").split("/summary.csv")[0]
    print("  %-50s success=%d/%d" % (tag, succ, n))
PY
echo "[dsr] done"

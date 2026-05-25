#!/usr/bin/env bash
# Runs ON a RunPod pod (via launch.py --cmd) to cross-check DSR's OWN implementation.
#
# DSR pins a 2019 stack (tensorflow==1.14, numpy<=1.19, numba==0.53.1, Cython), which
# needs Python 3.7 -- impossible on the cloud-sdk image's modern Python. So we install
# Miniconda and build an isolated py37 env for DSR, while the harness's system gsutil
# still handles the GCS sync. DSR's output (summary.csv etc.) is copied into ./results
# so the worker syncs it to gs://runpod-experiments/<exp>/results/.
#
# Env knobs (set in launch.py --cmd):
#   TASK=Nguyen-3   EPSILON=null (VPG) | 0.05 (risk)   NSAMPLES=200000   SEED=0
#   RUNS=1          NCORES=1   (DSR's per-task parallelism)
#
# Calibration: small NSAMPLES, 1 run -> confirms the env builds + times one job so we
# can extrapolate the full 2,000,000-sample sweep before committing to it.
set -uxo pipefail   # -x traces every step into worker.log (this IS the calibration output)

TASK="${TASK:-Nguyen-3}"
EPSILON="${EPSILON:-null}"          # null -> vanilla PG; 0.05 -> risk-seeking
NSAMPLES="${NSAMPLES:-200000}"      # full benchmark uses 2000000
SEED="${SEED:-0}"
RUNS="${RUNS:-1}"
NCORES="${NCORES:-1}"

# Capture the harness's synced results dir BEFORE we cd anywhere (cwd = the workdir).
RESULTS_DIR="$PWD/results/dsr-${TASK}-eps${EPSILON}-n${NSAMPLES}"
mkdir -p "$RESULTS_DIR"
echo "[dsr] nproc=$(nproc)  results->$RESULTS_DIR"

apt-get update -qq && apt-get install -y -qq build-essential wget git >/dev/null 2>&1 || true

# 1) Miniconda + isolated Python 3.7 env (TF1.14 has no wheels for modern Python).
if [ ! -x /opt/conda/bin/conda ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p /opt/conda
fi
set +u; source /opt/conda/etc/profile.d/conda.sh
conda create -y -n dsr python=3.7 >/dev/null 2>&1
conda activate dsr; set -u
python --version

# 2) Clone + install DSR into the py37 env.
rm -rf /workspace/dsr
git clone --depth 1 https://github.com/dso-org/deep-symbolic-optimization.git /workspace/dsr
cd /workspace/dsr/dso
python -m pip install --upgrade "pip<24" "setuptools<60" wheel "cython<3" "numpy<=1.19" >/dev/null
python -m pip install -e . 2>&1 | tail -25

# 3) Build a config: chosen epsilon (VPG=null), reduced n_samples, our logdir.
#    DSR configs use // comments -> load with commentjson (installed with DSR).
python - "$EPSILON" "$NSAMPLES" <<'PY'
import sys, json, commentjson
eps_s, nsamp = sys.argv[1], int(sys.argv[2])
cfg = commentjson.load(open("dso/config/config_regression.json"))
cfg.setdefault("training", {})["epsilon"] = None if eps_s == "null" else float(eps_s)
cfg["training"]["n_samples"] = nsamp
cfg.setdefault("experiment", {})["logdir"] = "/workspace/dsrlog"
json.dump(cfg, open("/workspace/calib.json", "w"), indent=1)
print("[dsr] config: epsilon=%r n_samples=%d" % (cfg["training"]["epsilon"], nsamp))
PY

# 4) Run + time it (this number drives the full-run sizing decision).
START=$(date +%s)
python -m dso.run /workspace/calib.json --b "$TASK" --runs "$RUNS" --seed "$SEED" \
    --n_cores_task "$NCORES" 2>&1 | tail -50
echo "[dsr] ELAPSED=$(( $(date +%s) - START ))s  for TASK=$TASK EPSILON=$EPSILON NSAMPLES=$NSAMPLES"

# 5) Copy DSR's output into the synced results dir (summary.csv + config).
cp -r /workspace/dsrlog/* "$RESULTS_DIR/" 2>/dev/null || true
ls -R "$RESULTS_DIR" 2>/dev/null | head -40
echo "[dsr] done"

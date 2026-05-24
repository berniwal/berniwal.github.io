# RunPod experiment harness (GCS-backed, fire-and-forget)

Launch experiments on a RunPod pod, let it run in the background, and pull results
from Google Cloud Storage when it's done. The pod **terminates itself** on
completion; results live in `gs://runpod-experiments/<exp>/`, so nothing is lost if
the pod dies and runs are resumable.

Generic over **CPU now** (Layer 0 — pure numpy, embarrassingly parallel, no GPU) and
**GPU later** (Layer 1, after it's ported off MLX — MLX is Apple-only and does not
run on RunPod). Same tool, just `--gpu`.

```
launch.py        create a pod, inject secrets, boot the worker (laptop-side)
worker_entry.sh  runs ON the pod: auth -> clone -> resume from GCS -> run -> push -> self-terminate
fetch.py         pull results from GCS to the local repo (laptop-side, no pod)
config.yaml      bucket, repo, image, default pod flavors
```

## One-time setup

1. **Google Cloud SDK** on the laptop (`gsutil`): https://cloud.google.com/sdk/docs/install
2. **Service-account key** with object access to the bucket:
   ```bash
   # objectAdmin on just gs://runpod-experiments
   gcloud iam service-accounts keys create ~/keys/runpod-experiments-sa.json \
       --iam-account=<sa>@<project>.iam.gserviceaccount.com
   export GOOGLE_APPLICATION_CREDENTIALS=~/keys/runpod-experiments-sa.json
   ```
3. **RunPod API key** (RunPod → Settings → API Keys):
   ```bash
   export RUNPOD_API_KEY=...
   ```
4. **Launcher deps:** `pip install -r experiments/runpod/requirements.txt`

> The keys live only on the laptop and are injected into the pod at launch (GCP key
> base64-encoded into an env var; RunPod key passed so the pod can remove itself).
> `*.json` and `*.log` are gitignored here so a key never lands in the repo.

## Run an experiment

```bash
cd experiments/runpod
# Layer 0 (CPU). NOTE: pass the direct python command, NOT ./run_overnight.sh
# (that wrapper is macOS/caffeinate-only).
python launch.py --exp ablation-2026-05-24 \
    --cmd "python3 run_ablation.py --budget 2000000 --seeds 20"
```

The pod will: download `worker_entry.sh` from the repo, clone the repo, `pip install`,
pull any prior `results/` from GCS (resume), run the command (pushing `results/` to
GCS every `push_interval` seconds), then push the final results + `worker.log` and
terminate itself.

```bash
# watch it (the log appears after the first push):
gsutil cat gs://runpod-experiments/ablation-2026-05-24/worker.log

# when done, pull results into experiments/self-improvement-arena/results/ and commit:
python fetch.py --exp ablation-2026-05-24
python fetch.py --list            # see all experiments in the bucket
```

GPU (future, after Layer 1 is ported off MLX):
```bash
python launch.py --gpu --exp l1-smoke --cmd "python3 run_layer1.py --config ..."
```

## Behavior notes

- **Resumable:** each run checkpoints per-JSON to `results/.../logs/`. The worker
  pulls those down on start, so a crashed/relaunched pod continues where it left off.
- **Crash-safe:** the worker pushes state + `worker.log` to GCS on *any* exit before
  terminating, so failures are inspectable in the bucket (read `worker.log`).
- **`--no-terminate`** leaves the pod up after the command (for debugging).
- **Self-termination** uses the RunPod GraphQL `podTerminate` mutation; swap for
  `runpodctl remove pod` if your account prefers that.

## Caveats / TODO

- `--cmd` must be a **foreground** command (the pod is the background). Don't use
  `run_overnight.sh` (caffeinate/nohup are macOS-only).
- The bootstrap fetches `worker_entry.sh` from `repo_ref` (default `main`), so push
  changes to that branch before launching with them.
- `create_pod` parameters can drift between `runpod` SDK versions; if launch fails,
  check the SDK version and that the CPU `instance_id` / GPU `gpu_type_id` are valid
  for your account. Not end-to-end tested in CI (needs live RunPod + GCP creds).

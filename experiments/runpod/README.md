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

## Check availability + pricing first

RunPod's managed CPU instances are frequently sold out, while cheap **community
GPUs** (whose 4-16 vCPUs run the CPU-only Layer-0 sweeps fine, GPU idle) are usually
available and often *cheaper*. Query before launching:

```bash
python availability.py gpu                        # cheapest in-stock GPUs (price + stock)
python availability.py gpu --all --max-price 0.3  # incl. out-of-stock, <= $0.30/hr
python availability.py cpu                         # CPU flavor specs (API has no CPU price/stock)
python availability.py cpu --probe --vcpu 8,16,32  # real CPU availability (create+terminate)
```

GPU `stockStatus` is High/Medium/Low (None = sold out). To run a CPU sweep on a GPU
pod, `launch.py --gpu --gpu-type "<id>"` and set `--workers` to the pod's vCPUs.

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

## GPU selection guide (lesson learned 2026-05-27)

For the kinds of workloads in this repo (Qwen3-1.7B LoRA + GRPO, batch ≤ 32,
maxnew ≤ 2048), **the H100 and A100 are not worth their premium.** Measured
per-round wall time across a real 5-pod parallel run (`batch=16 maxnew=2048+96`):

| GPU            | $/hr  | per-round | $ per round  | notes                       |
|----------------|------:|----------:|-------------:|-----------------------------|
| A40 (44 GB)    | 0.44  | 151 s     | $0.018       | reference                   |
| RTX A6000      | 0.49  | 139 s     | $0.019       | ≈ A40                       |
| A100 80GB PCIe | 1.39  | 139 s     | $0.054       | **3× cost, ≈ A40 speed**    |
| H100 PCIe      | 2.89  | 127 s     | $0.102       | **6× cost, 1.15× speed**    |

The workload is bottlenecked by **autoregressive decode + KV cache management +
tokenizer + BFGS const-fit**, not by raw matmul. H100/A100's tensor-core
advantage is mostly wasted; we'd need much larger batch (≥ 64) or much longer
sequences to start seeing the speedup that justifies the price.

**Rule of thumb:**

1. **Prefer A40 (secure cloud, $0.44/hr) for any LLM-rollout-heavy run.** RTX
   A6000 is a near-perfect equivalent if A40 is sold out.
2. **Do NOT escalate to A100/H100 just because A40 is briefly unavailable.**
   A40 secure-cloud capacity comes back within ~10 min in our experience. Wait
   and retry, or accept fewer parallel seeds, before paying 3-6× the rate.
3. **Sanity-check the per-round wall time against expectations** before
   committing to a long run. If a "fast" GPU is barely faster than A40 for
   your workload, you're not getting the speedup you paid for.
4. **The fallback ladder in `launch.py --gpu-type` should be A40 → RTX A6000,
   then stop.** Anything more expensive needs explicit justification (e.g.
   "the model genuinely doesn't fit in 44 GB" — but with gradient
   checkpointing on, Qwen3-1.7B at batch=32 fits comfortably on A40).
5. **If you do need >44 GB VRAM** (e.g. Qwen3-8B+ models without aggressive
   sharding), prefer A100 80GB PCIe SECURE over H100. H100 only pays back its
   premium when FlashAttention-3 + large-batch + long-seq are all in play
   simultaneously, which is rarely true for our small-model RL loops.

This rule cost us ~$5 of avoidable spend the first time we hit it (one H100
+ two A100s where two more A40s would have been ~$2). Worth writing down so
the next experiment doesn't repeat it.

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

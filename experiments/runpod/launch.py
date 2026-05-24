#!/usr/bin/env python3
"""Launch a fire-and-forget experiment pod on RunPod.

The pod boots, downloads worker_entry.sh from the repo, clones the repo, pulls any
prior results from GCS (resume), runs your command, pushes results + log back to
gs://<bucket>/<exp>/, and terminates itself. You pull results later with fetch.py.

    export RUNPOD_API_KEY=...                      # from RunPod settings
    export GOOGLE_APPLICATION_CREDENTIALS=~/keys/runpod-experiments-sa.json
    python launch.py --exp ablation-2026-05-24 \
        --cmd "python3 run_ablation.py --budget 2000000 --seeds 20"
    # GPU (future Layer 1, after porting off MLX):
    python launch.py --exp l1-test --gpu --cmd "python3 run_layer1.py ..."

Cannot be exercised without a RunPod account + GCP key; this is the laptop-side
control script. Nothing here runs the experiment locally.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def _raw_worker_url(repo_url: str, ref: str) -> str:
    """Derive the raw.githubusercontent URL of worker_entry.sh from the clone URL."""
    slug = repo_url.removeprefix("https://github.com/").removesuffix(".git")
    return (f"https://raw.githubusercontent.com/{slug}/{ref}/"
            "experiments/runpod/worker_entry.sh")


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment name -> gs://<bucket>/<exp>/")
    ap.add_argument("--cmd", required=True,
                    help="foreground command to run in the repo workdir, e.g. "
                         "'python3 run_ablation.py --budget 2000000 --seeds 20' "
                         "(NOT run_overnight.sh -- that's macOS/caffeinate only)")
    ap.add_argument("--gpu", action="store_true", help="GPU pod (default: CPU)")
    ap.add_argument("--cpu-instance", default=cfg["cpu_instance_id"])
    ap.add_argument("--gpu-type", default=cfg["gpu_type_id"])
    ap.add_argument("--repo-ref", default=cfg["repo_ref"])
    ap.add_argument("--push-interval", type=int, default=cfg["push_interval"])
    ap.add_argument("--no-terminate", action="store_true",
                    help="leave the pod running after the command (for debugging)")
    ap.add_argument("--sa-key", default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
                    help="path to the GCP service-account JSON key (or set "
                         "GOOGLE_APPLICATION_CREDENTIALS)")
    args = ap.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("error: set RUNPOD_API_KEY (RunPod -> Settings -> API Keys)")
    if not args.sa_key or not Path(args.sa_key).is_file():
        sys.exit("error: GCP key not found; pass --sa-key or set "
                 "GOOGLE_APPLICATION_CREDENTIALS to the service-account JSON")

    import runpod  # imported here so -h works without the SDK installed
    runpod.api_key = api_key

    sa_b64 = base64.b64encode(Path(args.sa_key).read_bytes()).decode()
    gcs_dest = f"{cfg['bucket'].rstrip('/')}/{args.exp}"
    env = {
        "GCS_DEST": gcs_dest,
        "REPO_URL": cfg["repo_url"],
        "REPO_REF": args.repo_ref,
        "WORKDIR": cfg["workdir"],
        "RUN_CMD": args.cmd,
        "GCP_SA_KEY_B64": sa_b64,
        "PUSH_INTERVAL": str(args.push_interval),
        "AUTO_TERMINATE": "0" if args.no_terminate else ("1" if cfg["auto_terminate"] else "0"),
        "RUNPOD_API_KEY": api_key,  # so the pod can terminate itself
    }
    raw = _raw_worker_url(cfg["repo_url"], args.repo_ref)
    docker_args = ("bash -c 'apt-get update -qq && apt-get install -y -qq curl >/dev/null "
                   f"&& curl -fsSL {raw} -o /worker_entry.sh && bash /worker_entry.sh'")

    common = dict(image_name=cfg["image"], env=env, docker_args=docker_args,
                  container_disk_in_gb=cfg["container_disk_gb"])
    print(f"launching {'GPU' if args.gpu else 'CPU'} pod '{args.exp}' -> {gcs_dest}/")
    try:
        if args.gpu:
            pod = runpod.create_pod(name=args.exp, gpu_type_id=args.gpu_type,
                                    gpu_count=1, cloud_type="COMMUNITY", **common)
        else:
            pod = runpod.create_pod(name=args.exp, instance_id=args.cpu_instance, **common)
    except Exception as e:  # SDK signatures drift between versions -> show the error
        sys.exit(f"create_pod failed ({e!r}). Check `runpod` SDK version and that the "
                 f"{'GPU type' if args.gpu else 'CPU instance id'} is valid for your account.")

    pid = pod.get("id", "?") if isinstance(pod, dict) else pod
    print(f"  pod id: {pid}")
    print(f"  watch:  gsutil cat {gcs_dest}/worker.log   (appears after first push)")
    print(f"  fetch:  python fetch.py --exp {args.exp}")


if __name__ == "__main__":
    main()

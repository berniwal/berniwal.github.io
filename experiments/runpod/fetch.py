#!/usr/bin/env python3
"""Pull experiment results from GCS to the local repo. No pod needed -- reads the
bucket directly with gsutil.

    python fetch.py --list                       # list experiments in the bucket
    python fetch.py --exp ablation-2026-05-24    # rsync its results/ into the sia results dir
    gsutil cat gs://runpod-experiments/<exp>/worker.log   # peek at a running/finished log

Requires the Google Cloud SDK (`gsutil`) on the laptop, authenticated to the bucket.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_DEST = HERE.parent / "self-improvement-arena" / "results"


def _run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    bucket = cfg["bucket"].rstrip("/")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", help="experiment name to pull")
    ap.add_argument("--list", action="store_true", help="list experiments in the bucket")
    ap.add_argument("--dest", default=str(DEFAULT_DEST),
                    help=f"local destination dir (default: {DEFAULT_DEST})")
    args = ap.parse_args()

    if args.list:
        sys.exit(_run(["gsutil", "ls", bucket + "/"]))
    if not args.exp:
        sys.exit("error: pass --exp <name> (or --list)")

    src = f"{bucket}/{args.exp}/results"
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"pulling {src}/  ->  {dest}/")
    rc = _run(["gsutil", "-m", "rsync", "-r", src, str(dest)])
    # also grab the worker log for convenience (non-fatal if absent)
    _run(["gsutil", "cp", f"{bucket}/{args.exp}/worker.log",
          str(dest / f"{args.exp}_worker.log")])
    if rc == 0:
        print(f"\ndone. Review {dest}/ and commit the artifacts (logs are gitignored).")
    sys.exit(rc)


if __name__ == "__main__":
    main()

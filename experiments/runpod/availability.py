#!/usr/bin/env python3
"""Query RunPod GPU/CPU availability + pricing from the laptop (no pod created,
unless you ask to probe CPU). Handy before `launch.py` to pick a flavor that is
actually in stock -- RunPod CPU instances are often sold out while cheap community
GPUs (whose vCPUs run our CPU-only Layer-0 sweeps fine) are available.

    # cheapest in-stock GPUs (sorted by on-demand price):
    python availability.py gpu
    python availability.py gpu --all --max-price 0.30      # incl. out-of-stock, <= $0.30/hr

    # CPU flavor specs (the API exposes NO price/stock for CPU):
    python availability.py cpu
    # real CPU availability = try to create+terminate a tiny pod per size:
    python availability.py cpu --probe --vcpu 8,16,32

GPU `stockStatus` is High/Medium/Low (None = out of stock). For GPU pods the vCPU/
RAM count is host-dependent (not in the API); community GPUs typically ship 4-16
vCPU, which is what our numpy runs use.

Auth: set RUNPOD_API_KEY, or pass --api-key-file (defaults to ~/runpod/
runpod-experiments.txt if present).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
import yaml

HERE = Path(__file__).resolve().parent
GQL = "https://api.runpod.io/graphql"


def get_api_key(api_key_file: str | None) -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key.strip()
    candidates = [api_key_file] if api_key_file else []
    candidates.append(os.path.expanduser("~/runpod/runpod-experiments.txt"))
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c).read_text().strip()
    sys.exit("error: set RUNPOD_API_KEY or pass --api-key-file <path>")


def gql(key: str, query: str) -> dict:
    r = requests.post(f"{GQL}?api_key={key}", json={"query": query}, timeout=30)
    data = r.json()
    if "errors" in data:
        sys.exit(f"GraphQL error: {data['errors'][0]['message']}")
    return data["data"]


def cmd_gpu(key: str, args) -> None:
    # minVcpu/minMemory = a LOWER BOUND on the host CPU/RAM you get with this GPU at
    # gpuCount (host-dependent, NOT tied to VRAM; the actual pod can give more -- e.g.
    # a "28 vCPU" pod landed on a 32-core EPYC). Decisive for CPU-bound sweeps on a
    # GPU pod: pick by this floor, then size --workers to the pod's real nproc.
    #
    # PRICE depends on the cloud tier: lowestPrice WITHOUT a cloud filter returns a
    # cross-tier floor (often the community price), which under-quotes a secure launch
    # (a "$0.34" RTX PRO 4500 actually bills $0.74 on secure). Pass secureCloud to get
    # the price for the tier you will actually launch on.
    cloud_filter = {"SECURE": ", secureCloud:true",
                    "COMMUNITY": ", secureCloud:false", "ANY": ""}[args.cloud_type]
    q = """query { gpuTypes {
      id displayName memoryInGb communityCloud secureCloud
      lowestPrice(input:{gpuCount:%d%s}) {
        uninterruptablePrice minimumBidPrice stockStatus minVcpu minMemory }
    } }""" % (args.gpu_count, cloud_filter)
    rows = []
    for g in gql(key, q)["gpuTypes"]:
        lp = g.get("lowestPrice") or {}
        price = lp.get("uninterruptablePrice")
        rows.append(dict(price=price, bid=lp.get("minimumBidPrice"),
                         stock=lp.get("stockStatus"), vram=g.get("memoryInGb"),
                         vcpu=lp.get("minVcpu"), ram=lp.get("minMemory"),
                         comm=g.get("communityCloud"), name=g["displayName"], id=g["id"]))
    rows = [r for r in rows if r["price"] is not None]
    if not args.all:
        rows = [r for r in rows if r["stock"]]
    if args.max_price is not None:
        rows = [r for r in rows if r["price"] <= args.max_price]
    rows.sort(key=lambda r: -(r["vcpu"] or 0) if args.by_vcpu else r["price"])

    print(f"{'$/hr≥':>9} {'spot':>6} {'vCPU≥':>5} {'RAM≥':>6} {'vram':>6} {'stock':>7} "
          f"{'cloud':>9}  GPU  [id]")
    for r in rows:
        # when filtered by tier the price IS that tier's; label it so (the per-type
        # communityCloud flag would mislabel a secure-priced row as "community").
        cloud = args.cloud_type.lower() if args.cloud_type != "ANY" \
            else ("community" if r["comm"] else "secure")
        bid = f"${r['bid']}" if r["bid"] else "-"
        print(f"{'$'+str(r['price']):>9} {bid:>6} {str(r['vcpu']):>5} {str(r['ram'])+'GB':>6} "
              f"{str(r['vram'])+'GB':>6} {str(r['stock'] or '-'):>7} {cloud:>9}  "
              f"{r['name']}  [{r['id']}]")
    print(f"\n{len(rows)} GPU type(s)"
          f"{' in stock' if not args.all else ''}; gpuCount={args.gpu_count}; "
          f"tier={args.cloud_type}.")
    print("NOTE: $/hr is the cheapest-datacenter floor FOR THIS TIER; vCPU/RAM are "
          "floors too. The placed pod can cost a touch more / have more cores -- "
          "launch.py prints the real costPerHr after create.")
    print("Launch:  python launch.py --gpu --gpu-type \"<id>\" --exp <name> --cmd \"...\"")


def cmd_cpu(key: str, args) -> None:
    flavors = gql(key, """query { cpuFlavors {
        id displayName minVcpu maxVcpu ramMultiplier diskLimitPerVcpu } }""")["cpuFlavors"]
    print(f"{'flavor':8} {'class':18} {'vCPU':>9} {'RAM/vCPU':>9} {'disk/vCPU':>9}")
    for f in flavors:
        print(f"{f['id']:8} {f['displayName']:18} {str(f['minVcpu'])+'-'+str(f['maxVcpu']):>9} "
              f"{str(f['ramMultiplier'])+'GB':>9} {str(f['diskLimitPerVcpu'])+'GB':>9}")
    print("\nNOTE: the API exposes NO price/stock for CPU. instance_id = "
          "<flavor>-<vCPU>-<vCPU*RAMmult>, e.g. cpu5c-16-32.")
    if not args.probe:
        print("Run with --probe --vcpu 8,16,32 to test real availability "
              "(creates + immediately terminates a tiny pod per size).")
        return

    import runpod
    runpod.api_key = key
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    img, disk = cfg["image"], cfg["container_disk_gb"]
    sizes = [int(v) for v in args.vcpu.split(",") if v.strip()]
    print(f"\nprobing (create+terminate) image={img} disk={disk}GB ...")
    avail = []
    for f in flavors:
        for vc in sizes:
            if not (f["minVcpu"] <= vc <= f["maxVcpu"]):
                continue
            inst = f"{f['id']}-{vc}-{vc * f['ramMultiplier']}"
            try:
                pod = runpod.create_pod(name=f"probe-{inst}", image_name=img,
                                        instance_id=inst, container_disk_in_gb=disk)
                pid = pod.get("id") if isinstance(pod, dict) else pod
                avail.append(inst)
                print(f"  AVAILABLE  {inst}  (pod {pid}) -> terminating")
                try:
                    runpod.terminate_pod(pid)
                except Exception as e:
                    print(f"     WARN: terminate failed for {pid}: {e!r}")
            except Exception as e:
                msg = str(e)
                tag = "no-capacity" if "no longer any instances" in msg else msg[:60]
                print(f"  --         {inst}  ({tag})")
    print(f"\nAVAILABLE NOW: {avail or 'none'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key-file", help="file containing the RunPod API key "
                    "(else $RUNPOD_API_KEY, else ~/runpod/runpod-experiments.txt)")
    sub = ap.add_subparsers(dest="cmd")

    g = sub.add_parser("gpu", help="list GPU types with price + stock (default)")
    g.add_argument("--all", action="store_true", help="include out-of-stock types")
    g.add_argument("--max-price", type=float, help="only show <= this on-demand $/hr")
    g.add_argument("--gpu-count", type=int, default=1)
    g.add_argument("--by-vcpu", action="store_true",
                   help="sort by vCPU desc (best for CPU-bound sweeps) instead of price")
    g.add_argument("--cloud-type", default="SECURE",
                   choices=["SECURE", "COMMUNITY", "ANY"],
                   help="price/stock for this tier (default SECURE, matching launch). "
                        "ANY = cross-tier floor (under-quotes a secure launch).")

    c = sub.add_parser("cpu", help="list CPU flavor specs; --probe for availability")
    c.add_argument("--probe", action="store_true",
                   help="create+terminate a tiny pod per size to test availability")
    c.add_argument("--vcpu", default="8,16,32", help="vCPU sizes to probe (CSV)")

    args = ap.parse_args()
    key = get_api_key(args.api_key_file)
    if args.cmd == "cpu":
        cmd_cpu(key, args)
    else:  # default to gpu
        if args.cmd is None:
            args = ap.parse_args(["gpu"])
            args.api_key_file = None
        cmd_gpu(key, args)


if __name__ == "__main__":
    main()

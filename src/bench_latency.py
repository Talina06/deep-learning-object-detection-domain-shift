"""Controlled latency benchmark (separate from the accuracy runs).

The accuracy runs log latency too, but they are long and can overlap with background macOS load
(e.g. mediaanalysisd competing for the GPU). This script measures every configuration back to back
on the SAME 300 images per dataset, batch size 1, after a 20-image warm-up, so the speed numbers in
the report are comparable with each other.

Latency = end-to-end per image: image decode + preprocess/resize + forward + NMS/postprocess + copy to CPU.

usage: python src/bench_latency.py            -> results/latency_bench.json
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import detectors as D

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = [  # (model key, dataset, imgsz)
    ("yolo11n", "coco", 640), ("yolo11s", "coco", 640), ("yolo11m", "coco", 640),
    ("rtdetr_l", "coco", 640), ("yoloworld_s", "coco", 640), ("frcnn_r50v2", "coco", None), ("fcos_r50", "coco", None),
    ("yolo11n", "kitti", 640), ("yolo11s", "kitti", 640), ("yolo11m", "kitti", 640),
    ("rtdetr_l", "kitti", 640), ("yoloworld_s", "kitti", 640), ("frcnn_r50v2", "kitti", None), ("fcos_r50", "kitti", None),
    ("yolo11s", "kitti", 1280), ("yolo11m", "kitti", 1280), ("rtdetr_l", "kitti", 1280), ("yoloworld_s", "kitti", 1280),
]
N_WARM, N = 20, 300
IMG_DIR = {"coco": ROOT / "data/coco/val2017", "kitti": ROOT / "data/kitti/images/val"}


def bench_one(key, ds, sz):
    """Runs in a fresh process so MPS driver memory reflects only this model."""
    os.chdir(ROOT / "weights")  # checkpoints live here
    det = D.build(key, sz or 640)
    if key.startswith("yoloworld"):
        det.set_prompts(D.KITTI_PROMPTS if ds == "kitti" else _coco_names())
    paths = [str(p) for p in sorted(IMG_DIR[ds].iterdir())[: N_WARM + N]]
    for p in paths[:N_WARM]:
        det.predict(p)
    lat, mem = [], 0.0
    for p in paths[N_WARM:]:
        _, ms = D.timed_predict(det, p)
        lat.append(ms)
        mem = max(mem, D.mem_mb(det.dev))
    return dict(model=det.name, key=key, dataset=ds, imgsz=str(det.imgsz), n=N,
                latency_ms_median=float(np.median(lat)), latency_ms_p90=float(np.percentile(lat, 90)),
                fps=1000 / float(np.median(lat)), mem_mb=mem)


def main():
    out = []
    for key, ds, sz in CONFIGS:
        r = subprocess.run([sys.executable, __file__, "--one", key, ds, str(sz or 0)],
                           capture_output=True, text=True, check=True)
        rec = json.loads(r.stdout.strip().splitlines()[-1])
        print(f"{key:<12} {ds:<6} {rec['imgsz']:<16} median {rec['latency_ms_median']:6.1f} ms  "
              f"p90 {rec['latency_ms_p90']:6.1f} ms  {rec['fps']:5.1f} FPS  MPS mem {rec['mem_mb']:.0f} MB", flush=True)
        out.append(rec)
    meta = dict(hardware="Apple M4, 16 GB unified memory", platform=platform.platform(), torch=torch.__version__,
                warmup=N_WARM, n_images=N, batch=1,
                memory_note="torch.mps.driver_allocated_memory() max over timed images, fresh process per config",
                results=out)
    (ROOT / "results/latency_bench.json").write_text(json.dumps(meta, indent=2))
    print("LATENCY BENCH FINISHED")


def _coco_names():
    from ultralytics.utils import ROOT as UROOT, YAML
    return list(YAML.load(UROOT / "cfg/datasets/coco.yaml")["names"].values())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        k, d, z = sys.argv[2], sys.argv[3], int(sys.argv[4])
        print(json.dumps(bench_one(k, d, z or None)))
    else:
        main()

"""Run one detector over one dataset and save raw detections + latency/memory log.

Detections are stored with the detector's *own* label names (COCO names or text prompts) so the
taxonomy mapping can be changed at evaluation time without re-running inference.

usage:
  python src/run_infer.py --model yolo11s --dataset coco  [--imgsz 640] [--limit N]
  python src/run_infer.py --model yoloworld_s --dataset kitti --imgsz 1280
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import detectors as D

DATA = Path(__file__).resolve().parent.parent / "data"
GT = {"coco": DATA / "coco/annotations/instances_val2017.json", "kitti": DATA / "kitti/kitti_val_coco.json"}
IMG_DIR = {"coco": DATA / "coco/val2017", "kitti": DATA / "kitti/images/val"}
N_WARMUP, N_LAT = 10, 300  # latency = median over the first N_LAT images after warm-up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", choices=["coco", "kitti"], required=True)
    ap.add_argument("--imgsz", type=int, default=640, help="ultralytics only; torchvision uses its own resize")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--prompts", default="", help="YOLO-World only: comma-separated prompts (overrides KITTI defaults)")
    a = ap.parse_args()

    det = D.build(a.model, a.imgsz)
    if a.model.startswith("yoloworld"):
        if a.prompts:
            det.set_prompts([p.strip() for p in a.prompts.split(",")])
        elif a.dataset == "kitti":
            det.set_prompts(D.KITTI_PROMPTS)
        else:
            from ultralytics.utils import YAML
            from ultralytics.utils import ROOT
            det.set_prompts(list(YAML.load(ROOT / "cfg/datasets/coco.yaml")["names"].values()))

    gt = json.loads(GT[a.dataset].read_text())
    images = gt["images"][: a.limit or None]
    run = f"{a.model}_{a.dataset}_{det.imgsz if isinstance(det.imgsz, int) else 'tv'}{a.tag}"
    out_dir = Path(__file__).resolve().parent.parent / "results" / run
    out_dir.mkdir(parents=True, exist_ok=True)

    # warm-up (kernel compilation / MPS graph caching)
    for im in images[:N_WARMUP]:
        det.predict(str(IMG_DIR[a.dataset] / im["file_name"]))

    dets, lat, mem_peak = [], [], 0.0
    t_start = time.time()
    for i, im in enumerate(images):
        (boxes, scores, labels), ms = D.timed_predict(det, str(IMG_DIR[a.dataset] / im["file_name"]))
        if i < N_LAT:
            lat.append(ms)
            mem_peak = max(mem_peak, D.mem_mb(det.dev))
        for b, s, l in zip(boxes.tolist(), scores.tolist(), labels):
            dets.append(dict(image_id=im["id"], label=l, score=round(s, 5),
                             bbox=[round(b[0], 2), round(b[1], 2), round(b[2] - b[0], 2), round(b[3] - b[1], 2)]))
        if i % 100 == 0 or i == len(images) - 1:
            el = time.time() - t_start
            eta = el / (i + 1) * (len(images) - i - 1)
            print(f"[{run}] {i + 1}/{len(images)}  median latency {np.median(lat):.1f} ms  "
                  f"elapsed {el / 60:.1f} min  ETA {eta / 60:.1f} min", flush=True)

    meta = dict(
        run=run, model=det.name, family=det.family, checkpoint=det.checkpoint, dataset=a.dataset,
        split="val2017" if a.dataset == "coco" else "kitti-val (ultralytics split, 1496 imgs)",
        n_images=len(images), imgsz=det.imgsz, conf_thresh=D.CONF_THRESH,
        nms_iou=("none (NMS-free)" if "rtdetr" in a.model else
                 D.NMS_IOU if isinstance(det, D.UltralyticsDetector) else
                 "torchvision default (FRCNN 0.5 / FCOS 0.6)"),
        max_det=D.MAX_DET, device=det.dev, hardware=platform.processor() + " " + platform.machine(),
        platform=platform.platform(), torch=torch.__version__,
        latency_ms_median=float(np.median(lat)), latency_ms_p90=float(np.percentile(lat, 90)),
        fps=1000.0 / float(np.median(lat)), mem_mb_peak=mem_peak, wall_s=time.time() - t_start,
        prompts=getattr(det, "prompts", None),
    )
    try:
        import torchvision, ultralytics
        meta.update(torchvision=torchvision.__version__, ultralytics=ultralytics.__version__)
    except ImportError:
        pass
    (out_dir / "detections.json").write_text(json.dumps(dets))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"DONE {run}  latency {meta['latency_ms_median']:.1f} ms  wall {meta['wall_s'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

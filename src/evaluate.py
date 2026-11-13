"""COCO-style evaluation (pycocotools) of saved detections on COCO val2017 or KITTI val.

Reports mAP@[.5:.95], AP50, AP75, AP_S/M/L, AR@100, per-class AP (and AP@0.7 on KITTI,
the official KITTI IoU for cars).

KITTI cyclist handling for closed-set COCO detectors (--cyclist):
  bicycle : use the COCO 'bicycle' box as the cyclist box (default)
  fuse    : cyclist box = union(bicycle box, best-overlapping person box), score = mean
            (a COCO 'person' riding a 'bicycle' is the closest COCO concept to a KITTI 'Cyclist',
             whose GT box covers rider + bike)

usage: python src/evaluate.py results/yolo11s_kitti_640 [--cyclist fuse]
"""
import argparse
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from taxonomy import COCO_TO_KITTI, EVAL_ID, PROMPT_TO_KITTI

DATA = Path(__file__).resolve().parent.parent / "data"
GT = {"coco": DATA / "coco/annotations/instances_val2017.json", "kitti": DATA / "kitti/kitti_val_coco.json"}


def _iou_matrix(a, b):  # xywh
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, None, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, None, 1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1) - inter + 1e-9)


def fuse_cyclists(dets):
    """bicycle + overlapping person -> one rider+bike box (only for closed-set COCO detectors)."""
    by_img = defaultdict(list)
    for d in dets:
        by_img[d["image_id"]].append(d)
    out = []
    for img, ds in by_img.items():
        bikes = [d for d in ds if d["label"] == "bicycle"]
        persons = [d for d in ds if d["label"] == "person" and d["score"] > 0.05]
        out += [d for d in ds if d["label"] != "bicycle"]
        if not bikes:
            continue
        if not persons:
            out += bikes
            continue
        # a rider overlaps the bike box and sits above it
        iou = _iou_matrix([b["bbox"] for b in bikes], [p["bbox"] for p in persons])
        for bi, b in enumerate(bikes):
            j = int(iou[bi].argmax())
            if iou[bi, j] > 0.1:
                p = persons[j]
                x1 = min(b["bbox"][0], p["bbox"][0]); y1 = min(b["bbox"][1], p["bbox"][1])
                x2 = max(b["bbox"][0] + b["bbox"][2], p["bbox"][0] + p["bbox"][2])
                y2 = max(b["bbox"][1] + b["bbox"][3], p["bbox"][1] + p["bbox"][3])
                out.append(dict(b, bbox=[x1, y1, x2 - x1, y2 - y1], score=(b["score"] + p["score"]) / 2))
            else:
                out.append(b)
    return out


def to_coco_results(dets, dataset, coco_gt, open_vocab):
    res = []
    if dataset == "coco":
        name2id = {c["name"]: c["id"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
        missing = {d["label"] for d in dets} - set(name2id)
        assert not missing, f"unmapped COCO labels: {missing}"
        for d in dets:
            res.append(dict(image_id=d["image_id"], category_id=name2id[d["label"]], bbox=d["bbox"], score=d["score"]))
    else:
        m = PROMPT_TO_KITTI if open_vocab else COCO_TO_KITTI
        for d in dets:
            if d["label"] in m:
                res.append(dict(image_id=d["image_id"], category_id=EVAL_ID[m[d["label"]]], bbox=d["bbox"], score=d["score"]))
    return res


def evaluate(run_dir: Path, cyclist="bicycle"):
    meta = json.loads((run_dir / "meta.json").read_text())
    dets = json.loads((run_dir / "detections.json").read_text())
    ds = meta["dataset"]
    open_vocab = meta.get("prompts") is not None and ds == "kitti"
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(GT[ds]))
    if ds == "kitti" and not open_vocab and cyclist == "fuse":
        dets = fuse_cyclists(dets)
    res = to_coco_results(dets, ds, gt, open_vocab)
    img_ids = [im["id"] for im in gt.dataset["images"]][: meta["n_images"]]  # same order as run_infer.py

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dt = gt.loadRes(res) if res else COCO()
        E = COCOeval(gt, dt, "bbox")
        E.params.imgIds = img_ids
        E.evaluate(); E.accumulate(); E.summarize()
    s = E.stats
    out = dict(run=meta["run"], model=meta["model"], dataset=ds, imgsz=str(meta["imgsz"]), cyclist_mode=cyclist if ds == "kitti" else None,
               mAP=s[0], AP50=s[1], AP75=s[2], APs=s[3], APm=s[4], APl=s[5], AR100=s[8],
               ARs=s[9], ARm=s[10], ARl=s[11],
               latency_ms=meta["latency_ms_median"], fps=meta["fps"], mem_mb=meta["mem_mb_peak"])

    # per-class AP from the precision array  [T, R, K, A, M]
    P = E.eval["precision"]
    t50, t70 = 0, int(np.argmin(np.abs(E.params.iouThrs - 0.7)))
    per = {}
    for k, cid in enumerate(E.params.catIds):
        name = gt.cats[int(cid)]["name"]
        def ap(pr):
            pr = pr[pr > -1]
            return float(pr.mean()) if pr.size else float("nan")
        per[name] = dict(AP=ap(P[:, :, k, 0, -1]), AP50=ap(P[t50, :, k, 0, -1]), AP70=ap(P[t70, :, k, 0, -1]),
                         APs=ap(P[:, :, k, 1, -1]), APm=ap(P[:, :, k, 2, -1]), APl=ap(P[:, :, k, 3, -1]))
    out["per_class"] = per
    suffix = f"_{cyclist}" if ds == "kitti" and not open_vocab else ""
    (run_dir / f"metrics{suffix}.json").write_text(json.dumps(out, indent=2))
    (run_dir / f"cocoeval{suffix}.txt").write_text(buf.getvalue())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--cyclist", choices=["bicycle", "fuse"], default="bicycle")
    a = ap.parse_args()
    for r in a.run_dirs:
        o = evaluate(Path(r), a.cyclist)
        print(f"{o['run']:<32} mAP {o['mAP']:.3f}  AP50 {o['AP50']:.3f}  AP75 {o['AP75']:.3f}  "
              f"S/M/L {o['APs']:.3f}/{o['APm']:.3f}/{o['APl']:.3f}  AR100 {o['AR100']:.3f}  {o['latency_ms']:.1f} ms")

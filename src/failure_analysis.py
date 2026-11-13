"""Failure analysis on KITTI val (operating point: per-model F1-optimal score threshold, IoU >= 0.5).

Detectors are calibrated very differently (median score of a true-positive car/person on KITTI:
Faster R-CNN 0.99, RT-DETR 0.74, YOLO11s 0.43, YOLO-World 0.08), so a single global threshold is
unfair. Each run gets the threshold that maximises F1 over all KITTI eval classes.

1. recall vs. GT box height (KITTI's own difficulty proxy: Easy >= 40 px, Moderate/Hard >= 25 px)
2. class confusion: which COCO label fires on each KITTI GT (car/van/truck/pedestrian/cyclist/tram)
3. qualitative grid of the images where YOLO11s@640 misses the most pedestrians/cyclists
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from evaluate import _iou_matrix
from taxonomy import COCO_TO_KITTI, EVAL_ID, KITTI_RAW, PROMPT_TO_KITTI

ROOT = Path(__file__).resolve().parent.parent
RES, OUT, FIG = ROOT / "results", ROOT / "results/summary", ROOT / "figs"
GT = json.loads((ROOT / "data/kitti/kitti_val_coco.json").read_text())
IMGS = {im["id"]: im for im in GT["images"]}
IOU = 0.5
RUNS = ["frcnn_r50v2_kitti_tv", "fcos_r50_kitti_tv", "yolo11s_kitti_640", "yolo11s_kitti_1280",
        "rtdetr_l_kitti_640", "rtdetr_l_kitti_1280", "yoloworld_s_kitti_640"]
BINS = [(0, 25), (25, 40), (40, 80), (80, 1e9)]
BIN_NAMES = ["<25 px", "25–40", "40–80", "≥80"]

# raw KITTI objects (one per physical object: take the first GT_MAP entry of each object)
objs = defaultdict(list)
seen = set()
for a in GT["annotations"]:
    key = (a["image_id"], tuple(a["bbox"]))
    if key in seen:
        continue
    seen.add(key)
    objs[a["image_id"]].append(a)


GT_EVAL = defaultdict(list)  # non-crowd eval GT per image
for a in GT["annotations"]:
    if not a["iscrowd"]:
        GT_EVAL[a["image_id"]].append(a)


def f1_threshold(run):
    """Greedy IoU>=0.5 matching of class-mapped detections vs non-crowd GT; sweep score threshold."""
    m = PROMPT_TO_KITTI if "world" in run else COCO_TO_KITTI
    dets = defaultdict(list)
    for x in json.loads((RES / run / "detections.json").read_text()):
        if x["label"] in m:
            dets[x["image_id"]].append((x["score"], EVAL_ID[m[x["label"]]], x["bbox"]))
    scores, is_tp = [], []
    for img_id, ds in dets.items():
        ds.sort(key=lambda t: -t[0])
        gts = GT_EVAL.get(img_id, [])
        used = set()
        for sc, c, b in ds:
            cand = [(j, g) for j, g in enumerate(gts) if g["category_id"] == c and j not in used]
            best = None
            if cand:
                iou = _iou_matrix([b], [g["bbox"] for _, g in cand])[0]
                k = int(iou.argmax())
                if iou[k] >= IOU:
                    best = cand[k][0]
            if best is not None:
                used.add(best)
            scores.append(sc); is_tp.append(best is not None)
    scores, is_tp = np.array(scores), np.array(is_tp)
    n_gt = sum(len(v) for v in GT_EVAL.values())
    best_t, best_f1 = 0.3, -1
    for t in np.arange(0.02, 0.995, 0.005):
        k = scores >= t
        tp = is_tp[k].sum(); prec = tp / max(k.sum(), 1); rec = tp / n_gt
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


THRESH = {}


def load(run):
    if run not in THRESH:
        THRESH[run] = f1_threshold(run)
    d = defaultdict(list)
    for x in json.loads((RES / run / "detections.json").read_text()):
        if x["score"] >= THRESH[run][0]:
            d[x["image_id"]].append(x)
    return d


recall_rows, conf_rows, miss_per_img = [], [], {}
for run in RUNS:
    if not (RES / run).exists():
        continue
    dets = load(run)
    hit = defaultdict(lambda: np.zeros(len(BINS))); tot = defaultdict(lambda: np.zeros(len(BINS)))
    conf = defaultdict(Counter)
    for img_id, gts in objs.items():
        ds = dets.get(img_id, [])
        iou = _iou_matrix([g["bbox"] for g in gts], [d["bbox"] for d in ds]) if ds else np.zeros((len(gts), 0))
        misses = 0
        for gi, g in enumerate(gts):
            raw = g["kitti_raw"]
            h = g["bbox"][3]
            b = next(i for i, (lo, hi) in enumerate(BINS) if lo <= h < hi)
            tot[raw][b] += 1
            if iou.shape[1] and iou[gi].max() >= IOU:
                hit[raw][b] += 1
                conf[raw][ds[int(iou[gi].argmax())]["label"]] += 1  # label of best-overlapping det
            else:
                conf[raw]["(missed)"] += 1
                misses += raw in ("pedestrian", "cyclist")
        if run == "yolo11s_kitti_640":
            miss_per_img[img_id] = misses
    for raw in ("car", "pedestrian", "cyclist"):
        r = {"run": run, "class": raw}
        for i, n in enumerate(BIN_NAMES):
            r[n] = 100 * hit[raw][i] / max(tot[raw][i], 1)
        r["n per bin"] = "/".join(str(int(t)) for t in tot[raw])
        recall_rows.append(r)
    for raw in ("car", "van", "truck", "pedestrian", "cyclist", "tram"):
        c = conf[raw]; n = sum(c.values())
        top = ", ".join(f"{k} {100*v/n:.0f}%" for k, v in c.most_common(4))
        conf_rows.append({"run": run, "KITTI GT": raw, "n": n, "best-overlap label (F1-opt thr, IoU≥0.5)": top})

rec = pd.DataFrame(recall_rows)
(OUT / "recall_by_height.md").write_text(rec.to_markdown(index=False, floatfmt=".1f"))
(OUT / "class_confusion.md").write_text(pd.DataFrame(conf_rows).to_markdown(index=False))
print(rec.to_string())

# ---- figure: recall vs height, pedestrian + car, small multiples
COL = {"frcnn_r50v2_kitti_tv": ("#2a78d6", "s", "Faster R-CNN"), "fcos_r50_kitti_tv": ("#eb6834", "D", "FCOS"),
       "yolo11s_kitti_640": ("#1baf7a", "o", "YOLO11s@640"), "yolo11s_kitti_1280": ("#1baf7a", "o", "YOLO11s@1280"),
       "rtdetr_l_kitti_640": ("#eda100", "^", "RT-DETR-L@640"), "rtdetr_l_kitti_1280": ("#eda100", "^", "RT-DETR-L@1280"),
       "yoloworld_s_kitti_640": ("#e87ba4", "P", "YOLO-World-S@640")}
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, cls in zip(axs, ("car", "pedestrian", "cyclist")):
    for run, (c, m, lab) in COL.items():
        r = rec[(rec.run == run) & (rec["class"] == cls)]
        if r.empty:
            continue
        ls = "--" if "1280" in run else "-"
        ax.plot(BIN_NAMES, r[BIN_NAMES].values[0], ls, color=c, marker=m, lw=2, ms=7, mec="white", label=lab)
    n = rec[rec["class"] == cls]["n per bin"].iloc[0]
    ax.set_title(f"{cls}  (n = {n})", loc="left", fontsize=10)
    ax.set_xlabel("GT box height (KITTI pixels)"); ax.grid(color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel(f"recall @ IoU≥{IOU}, F1-optimal threshold (%)")
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=7, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(FIG / "recall_by_height.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- qualitative grid: worst images for YOLO11s@640 on vulnerable road users
worst = sorted(miss_per_img, key=miss_per_img.get, reverse=True)[:2]
show = ["yolo11s_kitti_640", "yolo11s_kitti_1280", "rtdetr_l_kitti_640", "frcnn_r50v2_kitti_tv", "yoloworld_s_kitti_640"]
show = [s for s in show if (RES / s).exists()]
fig, axs = plt.subplots(len(show), len(worst), figsize=(7.2 * len(worst), 2.3 * len(show)))
axs = np.atleast_2d(axs).reshape(len(show), len(worst))
dets_cache = {s: load(s) for s in show}
SHOWN = set(COCO_TO_KITTI) | set(PROMPT_TO_KITTI) | {"motorcycle", "bus"}  # scored classes + key confusions
for j, img_id in enumerate(worst):
    im = Image.open(ROOT / "data/kitti/images/val" / IMGS[img_id]["file_name"])
    for i, s in enumerate(show):
        ax = axs[i, j]; ax.imshow(im); ax.axis("off")
        for g in objs[img_id]:
            x, y, w, h = g["bbox"]
            ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, ec="white", lw=1.2, ls="--"))
        for d in dets_cache[s].get(img_id, []):
            if d["label"] not in SHOWN:  # e.g. traffic light: not scored on KITTI, hidden for clarity
                continue
            x, y, w, h = d["bbox"]
            c = {"person": "#e34948", "pedestrian": "#e34948", "person sitting": "#e34948",
                 "bicycle": "#2a78d6", "cyclist": "#2a78d6"}.get(d["label"], "#1baf7a")
            ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, ec=c, lw=1.4))
            ax.text(x, y - 2, f'{d["label"]} {d["score"]:.2f}', color="white", fontsize=5,
                    bbox=dict(fc=c, ec="none", pad=0.5))
        ax.set_title(f'{COL[s][2]} — {IMGS[img_id]["file_name"]}', fontsize=8, loc="left")
fig.suptitle("White dashed = KITTI GT · red = person/pedestrian · blue = bicycle/cyclist · green = vehicles   (per-model F1-optimal threshold)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "qualitative_failures.png", dpi=170, bbox_inches="tight"); plt.close(fig)
print("worst images:", [IMGS[i]["file_name"] for i in worst])
thr = pd.DataFrame([{"run": r, "F1-optimal score threshold": t, "F1": f} for r, (t, f) in THRESH.items()])
(OUT / "operating_thresholds.md").write_text(thr.to_markdown(index=False, floatfmt=".2f"))
print(thr.to_string())

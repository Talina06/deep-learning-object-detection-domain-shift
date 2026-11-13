"""Aggregate metrics*.json into markdown/CSV tables and report figures (results/summary, figs/)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES, OUT, FIG = ROOT / "results", ROOT / "results/summary", ROOT / "figs"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)

# validated categorical palette (dataviz reference, light mode) - one slot per detector FAMILY
FAM = {"Faster R-CNN": ("#2a78d6", "s"), "FCOS": ("#eb6834", "D"), "YOLO11": ("#1baf7a", "o"),
       "RT-DETR": ("#eda100", "^"), "YOLO-World": ("#e87ba4", "P")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200})


def fam(model):
    return next(k for k in FAM if model.startswith(k))


BENCH = {}
if (RES / "latency_bench.json").exists():  # controlled back-to-back latency runs override in-run latency
    for b in json.loads((RES / "latency_bench.json").read_text())["results"]:
        BENCH[(b["model"], b["dataset"], b["imgsz"])] = b


def load(pattern):
    rows = []
    for f in sorted(RES.glob(pattern)):
        if "_prompts-" in f.parent.name:  # prompt ablation: reported separately (prompt_ablation.md)
            continue
        m = json.loads(f.read_text())
        m["family"] = fam(m["model"])
        b = BENCH.get((m["model"], m["dataset"], m["imgsz"]))
        if b:
            m.update(latency_ms=b["latency_ms_median"], fps=b["fps"], mem_mb=b["mem_mb"])
        rows.append(m)
    return rows


def table(rows, cols, path, fmt="{:.1f}"):
    df = pd.DataFrame(rows)
    out = df[["model", "imgsz"] + cols].copy()
    for c in cols:
        if c.startswith(("mAP", "AP", "AR")):
            out[c] = (out[c] * 100).map(fmt.format)
        elif c in ("latency_ms", "fps", "mem_mb"):
            out[c] = out[c].map("{:.0f}".format if c == "mem_mb" else "{:.1f}".format)
    out.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".md").write_text(out.to_markdown(index=False))
    return df


core = ["mAP", "AP50", "AP75", "APs", "APm", "APl", "AR100", "latency_ms", "fps", "mem_mb"]
coco = load("*_coco_*/metrics.json")
# KITTI: closed-set use cyclist-fusion variant as headline; open-vocab has plain metrics.json
kitti_b = load("*_kitti_*/metrics_bicycle.json") + load("*world*_kitti_*/metrics.json")
kitti_f = load("*_kitti_*/metrics_fuse.json") + load("*world*_kitti_*/metrics.json")
dc = table(coco, core, OUT / "coco")
dk = table(kitti_f, core, OUT / "kitti_fuse")
table(kitti_b, core, OUT / "kitti_bicycle")

# per-class tables
def per_class(rows, classes, path, key="AP"):
    recs = []
    for r in rows:
        d = {"model": r["model"], "imgsz": r["imgsz"]}
        for c in classes:
            d[c] = r["per_class"][c][key] * 100
        recs.append(d)
    df = pd.DataFrame(recs)
    path.with_suffix(".md").write_text(df.to_markdown(index=False, floatfmt=".1f"))
    df.to_csv(path.with_suffix(".csv"), index=False)
    return df

KC = ["car", "pedestrian", "cyclist", "truck", "tram"]
per_class(kitti_f, KC, OUT / "kitti_perclass_AP")
per_class(kitti_f, KC, OUT / "kitti_perclass_AP50", "AP50")
per_class(kitti_f, KC, OUT / "kitti_perclass_AP70", "AP70")
per_class(coco, ["person", "car", "bicycle", "truck", "train"], OUT / "coco_perclass_AP")

# cyclist mode ablation
cy = [{"model": b["model"], "imgsz": b["imgsz"], "cyclist AP (bicycle box)": b["per_class"]["cyclist"]["AP"] * 100,
       "cyclist AP (person+bicycle fused)": f["per_class"]["cyclist"]["AP"] * 100,
       "mAP bicycle": b["mAP"] * 100, "mAP fused": f["mAP"] * 100}
      for b, f in zip(load("*_kitti_*/metrics_bicycle.json"), load("*_kitti_*/metrics_fuse.json"))]
(OUT / "cyclist_ablation.md").write_text(pd.DataFrame(cy).to_markdown(index=False, floatfmt=".1f"))

# domain-shift table: same concept, COCO AP vs KITTI AP (both @640 / torchvision default)
PAIRS = [("person", "pedestrian"), ("car", "car"), ("bicycle", "cyclist"), ("truck", "truck"), ("train", "tram")]
ks = {r["model"]: r for r in kitti_f if r["imgsz"] in ("640", "min800/max1333")}
ds_rows = []
for r in coco:
    k = ks.get(r["model"])
    if not k:
        continue
    d = {"model": r["model"], "COCO mAP": r["mAP"] * 100, "KITTI mAP": k["mAP"] * 100}
    for cc, kc in PAIRS:
        d[f"{cc}->{kc}"] = f'{r["per_class"][cc]["AP"]*100:.1f} -> {k["per_class"][kc]["AP"]*100:.1f}'
    for s in ("APs", "APm", "APl"):
        d[f"{s} COCO->KITTI"] = f"{r[s]*100:.1f} -> {k[s]*100:.1f}"
    ds_rows.append(d)
(OUT / "domain_shift.md").write_text(pd.DataFrame(ds_rows).to_markdown(index=False, floatfmt=".1f"))

# resolution ablation
res = []
for r in kitti_f:
    if r["imgsz"] == "1280":
        lo = next(x for x in kitti_f if x["model"] == r["model"] and x["imgsz"] == "640")
        res.append({"model": r["model"], "mAP@640": lo["mAP"] * 100, "mAP@1280": r["mAP"] * 100,
                    "APs@640": lo["APs"] * 100, "APs@1280": r["APs"] * 100,
                    "ped AP@640": lo["per_class"]["pedestrian"]["AP"] * 100,
                    "ped AP@1280": r["per_class"]["pedestrian"]["AP"] * 100,
                    "lat@640 ms": lo["latency_ms"], "lat@1280 ms": r["latency_ms"]})
(OUT / "resolution_ablation.md").write_text(pd.DataFrame(res).to_markdown(index=False, floatfmt=".1f"))


# ------------------------------------------------------------------ figures
def scatter(ax, rows, title):
    for r in rows:
        c, mk = FAM[r["family"]]
        ax.scatter(r["latency_ms"], r["mAP"] * 100, s=60, color=c, marker=mk, edgecolor="white", linewidth=1.5, zorder=3)
        lab = r["model"].replace(" R50-FPN", "").replace("-v2-S", "") + ("@1280" if r["imgsz"] == "1280" else "")
        off, ha = (5, 3), "left"
        if r["dataset"] == "kitti" and lab == "RT-DETR-L":
            off = (5, -11)  # below its marker, clear of YOLO11m@1280
        if r["dataset"] == "kitti" and lab == "YOLO11m@1280":
            off, ha = (-6, 3), "right"  # left of its marker, clear of the RT-DETR-L marker
        ax.annotate(lab, (r["latency_ms"], r["mAP"] * 100), xytext=off, textcoords="offset points", fontsize=7.5, color=INK, ha=ha)
    ax.set_xscale("log"); ax.set_xlabel("median latency per image, ms (log scale, Apple M4 MPS, batch 1)")
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, ScalarFormatter
    ax.xaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 400]))
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("mAP@[.5:.95] (%)"); ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
scatter(axs[0], coco, "COCO val2017 (in-domain)")
scatter(axs[1], kitti_f, "KITTI val (zero-shot transfer)")
h = [plt.Line2D([], [], marker=m, color=c, linestyle="", markersize=7, label=k) for k, (c, m) in FAM.items()]
fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(FIG / "speed_accuracy.png", bbox_inches="tight"); plt.close(fig)

# domain shift per concept: paired bars COCO (blue) vs KITTI (orange)
models = [r["model"] for r in coco if r["model"] in ks]
fig, axs = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
for ax, (cc, kc) in zip(axs, PAIRS[:4]):
    x = np.arange(len(models)); w = 0.38
    a = [next(r for r in coco if r["model"] == m)["per_class"][cc]["AP"] * 100 for m in models]
    b = [ks[m]["per_class"][kc]["AP"] * 100 for m in models]
    ax.bar(x - w / 2 - 0.01, a, w, color="#2a78d6", label="COCO", zorder=3)
    ax.bar(x + w / 2 + 0.01, b, w, color="#eb6834", label="KITTI (zero-shot)", zorder=3)
    ax.set_xticks(x, [m.replace(" R50-FPN", "").replace(" v2", "").replace("-v2-S", "") for m in models], rotation=60, ha="right", fontsize=7.5)
    ax.set_title(f"{cc} → {kc}", loc="left", fontsize=10, color=INK); ax.grid(axis="y", color=GRID, zorder=0)
axs[0].set_ylabel("AP@[.5:.95] (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(); fig.savefig(FIG / "domain_shift_per_class.png", bbox_inches="tight"); plt.close(fig)

# object-size distribution: why KITTI is a small-object problem at 640
from pycocotools.coco import COCO  # noqa: E402
import contextlib, io  # noqa: E401,E402
with contextlib.redirect_stdout(io.StringIO()):
    gc = COCO(str(ROOT / "data/coco/annotations/instances_val2017.json"))
    gk = COCO(str(ROOT / "data/kitti/kitti_val_coco.json"))
def size_frac(g, scale=1.0):
    a = np.array([x["area"] * scale**2 for x in g.dataset["annotations"] if not x["iscrowd"]])
    return [(a < 32**2).mean(), ((a >= 32**2) & (a < 96**2)).mean(), (a >= 96**2).mean()]
sz = pd.DataFrame({"COCO val2017": size_frac(gc), "KITTI val (native 1224 px)": size_frac(gk),
                   "KITTI as seen @640 (×0.52)": size_frac(gk, 640 / 1224)}, index=["small <32²", "medium", "large ≥96²"]).T
(OUT / "size_distribution.md").write_text((sz * 100).to_markdown(floatfmt=".1f"))
print((sz * 100).round(1))
print(open(OUT / "coco.md").read()); print(open(OUT / "kitti_fuse.md").read())

# YOLO-World prompt ablation on KITTI @640
pa = []
for d in [RES / "yoloworld_s_kitti_640"] + sorted(RES.glob("yoloworld_s_kitti_640_prompts-*")):
    m = json.loads((d / "metrics.json").read_text()); meta = json.loads((d / "meta.json").read_text())
    pa.append({"run": d.name, "prompts": ", ".join(meta["prompts"]) if len(meta["prompts"]) < 12 else f"all {len(meta['prompts'])} COCO names",
               "mAP": m["mAP"] * 100, **{f"{c} AP": m["per_class"][c]["AP"] * 100 for c in KC}})
(OUT / "prompt_ablation.md").write_text(pd.DataFrame(pa).to_markdown(index=False, floatfmt=".1f"))


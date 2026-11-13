#!/usr/bin/env bash
# Full experiment log for HW: 2D Object Detection (Track A).  Every number in the report comes from here.
# Hardware: Apple M4 (10-core GPU, 16 GB unified memory), PyTorch MPS backend, batch size 1.
set -euo pipefail
cd "$(dirname "$0")"
PY="$(pwd)/.venv/bin/python"
mkdir -p weights logs && cd weights   # ultralytics downloads checkpoints into CWD

# ---- 0. data (already downloaded; see README) ----
# COCO val2017 images + instances_val2017.json  -> data/coco
# KITTI (ultralytics split, 7481 -> 5985 train / 1496 val) -> data/kitti
$PY ../src/kitti_to_coco.py ../data/kitti ../data/kitti/kitti_val_coco.json

# ---- 1. COCO val2017 (in-domain), 5000 images ----
for m in yolo11n yolo11s yolo11m rtdetr_l yoloworld_s; do
  $PY ../src/run_infer.py --model $m --dataset coco --imgsz 640
done
$PY ../src/run_infer.py --model frcnn_r50v2 --dataset coco
$PY ../src/run_infer.py --model fcos_r50    --dataset coco

# ---- 2. KITTI val (COCO-pretrained, NO retraining -> domain shift), 1496 images ----
for m in yolo11n yolo11s yolo11m rtdetr_l yoloworld_s; do
  $PY ../src/run_infer.py --model $m --dataset kitti --imgsz 640
done
$PY ../src/run_infer.py --model frcnn_r50v2 --dataset kitti
$PY ../src/run_infer.py --model fcos_r50    --dataset kitti

# ---- 3. KITTI input-resolution ablation (KITTI images are 1224x370; 640 => 0.52x downscale) ----
for m in yolo11s yolo11m rtdetr_l yoloworld_s; do
  $PY ../src/run_infer.py --model $m --dataset kitti --imgsz 1280
done

# ---- 3b. YOLO-World prompt ablation (KITTI @640) ----
$PY ../src/run_infer.py --model yoloworld_s --dataset kitti --imgsz 640 --tag _prompts-person \
    --prompts "car,van,truck,person,person sitting,cyclist,tram"
$PY ../src/run_infer.py --model yoloworld_s --dataset kitti --imgsz 640 --tag _prompts-rider \
    --prompts "car,van,truck,person,person riding a bicycle,bicycle,bus,tram"
$PY ../src/run_infer.py --model yoloworld_s --dataset kitti --imgsz 640 --tag _prompts-coco80 \
    --prompts "$($PY -c 'from ultralytics.utils import ROOT,YAML;print(",".join(YAML.load(ROOT/"cfg/datasets/coco.yaml")["names"].values()))')"

# ---- 4. evaluation ----
cd ..
$PY src/evaluate.py results/*_coco_640 results/*_coco_tv
$PY src/evaluate.py --cyclist bicycle results/*_kitti_640 results/*_kitti_1280 results/*_kitti_tv
$PY src/evaluate.py --cyclist fuse    results/*_kitti_640 results/*_kitti_1280 results/*_kitti_tv
$PY src/evaluate.py results/yoloworld_s_kitti_640_prompts-*
# ---- 5. controlled latency/memory benchmark (same 300 imgs per dataset, fresh process per config) ----
$PY src/bench_latency.py
$PY src/summarize.py
$PY src/failure_analysis.py
echo "ALL EXPERIMENTS FINISHED"

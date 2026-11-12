# Do AI Object Detectors Work on Self-Driving Data?

This project takes AI models trained to find objects in everyday photos and tests them, **without any retraining**, on images from a car's dashboard camera. It measures how much accuracy they lose, and explains why.

## The basics

### What is object detection?

An **object detector** is an AI model that looks at an image and draws a box around every object it recognizes, with a label and a confidence score:

```
image  →  detector  →  [ car 0.94 (box) , person 0.81 (box) , bicycle 0.40 (box) , ... ]
```

Self-driving cars use detectors to find other cars, pedestrians and cyclists.

### The two datasets

| Dataset | What it contains | Role in this project |
| --- | --- | --- |
| **COCO** | 120k everyday photos (kitchens, streets, sports, animals) labelled with 80 object types | The models were **trained** on it. We test on its 5,000-image validation set to confirm they work as advertised. |
| **KITTI** | Photos from a car driving around Karlsruhe, Germany, labelled with cars, vans, trucks, pedestrians, cyclists and trams | The **new environment**. We test on 1,496 images the models have never seen. |

Moving a model from one kind of data to another is called **domain shift**, and it usually costs accuracy. This project measures how much, and why.

### How accuracy is measured

- **IoU (Intersection over Union):** how much a predicted box overlaps the true box. 1.0 = perfect overlap, 0.5 = half.
- **AP (Average Precision):** for one object type, a score from 0 to 100 that rewards finding every object (recall) without false alarms (precision).
- **mAP:** AP averaged over all object types and over strictness levels from IoU 0.50 to 0.95. This is the standard headline number. **Higher is better.**
- **AP50:** the same but lenient: a box counts as correct if it overlaps at least 50%.
- **Latency:** milliseconds the model takes per image. **Lower is faster.**

## The models

Seven model configurations from five design families. They all solve the same task in different ways.

| Model | Family | How it works, in plain words |
| --- | --- | --- |
| **Faster R-CNN** | Two-stage | First proposes "regions that might contain an object", then classifies each region and refines its box. Accurate and well studied, but slow. Uses **anchors**: thousands of preset template boxes that it adjusts to fit objects. |
| **FCOS** | One-stage, anchor-free | Skips the proposal step. Every pixel of the feature map directly predicts "what object am I part of, and how far are its four edges from me?" No preset template boxes. |
| **YOLO11** (n, s, m = small to medium) | One-stage, real-time | "You Only Look Once": one fast pass over the image predicts all boxes at once. Built for speed. The n/s/m sizes trade speed for accuracy. |
| **RT-DETR** | Transformer | Uses **attention** (the mechanism behind ChatGPT-style models) to look at the whole image at once. It has 300 "object queries", slots that each claim one object, so it needs no clean-up step to remove duplicate boxes. |
| **YOLO-World** | Open-vocabulary | A YOLO model connected to a language model (CLIP). Instead of a fixed list of classes, you **type** what to look for, e.g. `"car, van, pedestrian, tram"`. It can find object types it was never explicitly trained on. |

**NMS (Non-Maximum Suppression):** most detectors predict several overlapping boxes for the same object, and NMS keeps only the best one. RT-DETR is designed so it doesn't need this step.

## What we found

| Model | COCO mAP | KITTI mAP | Speed on KITTI (ms/image) |
| --- | --- | --- | --- |
| RT-DETR-L | 52.6 | 23.7 | 70 |
| YOLO11m | 50.5 | 21.4 | 19 |
| Faster R-CNN | 46.9 | 22.8 | 243 |
| YOLO11s | 45.5 | 18.9 | 12 |
| FCOS | 39.2 | 21.8 | 139 |
| YOLO11n | 38.5 | 15.4 | 10 |
| YOLO-World-S | 36.6 | 22.8 | 14 |
| YOLO-World-S, larger input (1280 px) | n/a | **25.1** | 29 |

In plain words:

1. **Every model loses more than half its accuracy on driving images.** The best COCO model drops from 52.6 to 23.7 mAP.
2. **Cars transfer well; people and cyclists don't.** Cars score about the same on both datasets. Pedestrians drop to about 20 AP for every model, because KITTI's human annotators drew slightly bigger boxes than the models learned to predict. Cyclists score under 4 AP, because COCO has "person" and "bicycle" but no single "person riding a bicycle" label.
3. **The typed-label model (YOLO-World) adapts best.** It is the weakest on COCO but among the best on KITTI, because you can ask it for KITTI's own categories like "truck" and "van". The exact words matter: asking for "person" instead of "pedestrian" improved its pedestrian score by 4.7 points.
4. **Bigger input images help a lot.** KITTI photos are very wide (1224×370 px). Shrinking them to 640 px makes distant objects tiny. Feeding them in at 1280 px improved every model, and helped the small, fast models the most.
5. **Fast models aren't worse here.** YOLO-World-S matches Faster R-CNN's KITTI accuracy while running 18× faster.

Full tables are in [`results/summary/`](results/summary) and charts in [`figs/`](figs).

## Files

| File | What it does |
| --- | --- |
| `detectors.py` | Wraps all seven models so they take an image and return boxes in the same format |
| `taxonomy.py` | Translates KITTI's labels (car, van, cyclist, ...) into COCO's labels (car, truck, person, bicycle, ...) |
| `kitti_to_coco.py` | Converts KITTI's label files into the format the scoring tool expects |
| `run_infer.py` | Runs one model on one dataset and saves its boxes and settings |
| `evaluate.py` | Scores the saved boxes against the true labels (uses the official COCO scoring code, `pycocotools`) |
| `bench_latency.py` | Measures speed and memory fairly: same 300 images, one model at a time |
| `summarize.py` | Builds the result tables and charts |
| `failure_analysis.py` | Digs into mistakes: which objects are missed, what gets mislabelled, example images |
| `run_all.sh` | **Every command, in order.** Running it reproduces the whole study |
| `logs/` | The real console output from running the experiments |
| `results/<run>/meta.json` | The exact settings used for each run (model file, image size, thresholds, hardware, versions) |

## Run it yourself

You need Python 3.10+ and about 2 GB of disk for the data. A GPU helps; this was run on an Apple M4 laptop.

```bash
# 1. install
python3 -m venv .venv
.venv/bin/pip install torch torchvision ultralytics pycocotools pandas matplotlib tabulate

# 2. download data
mkdir -p data/coco data/kitti
curl -L -o val2017.zip http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip -d data/coco
curl -L -o ann.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q ann.zip -d data/coco
curl -L -o kitti.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip && unzip -q kitti.zip -d data/kitti

# 3. run everything (~1.5 hours on an Apple M4; model files download automatically)
./run_all.sh
```

To try just one model on one dataset:

```bash
cd weights
../.venv/bin/python ../run_infer.py --model yolo11s --dataset kitti --imgsz 640
cd .. && .venv/bin/python evaluate.py results/yolo11s_kitti_640 --cyclist fuse
```

The raw box files (`results/*/detections.json`, ~426 MB) are not stored in this repo; `run_all.sh` recreates them.

## Credits

- Models: [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO11, RT-DETR, YOLO-World) and [torchvision](https://github.com/pytorch/vision) (Faster R-CNN, FCOS).
- Data: [COCO](https://cocodataset.org) (CC BY 4.0) and the [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/) (CC BY-NC-SA 3.0); KITTI images appear in `figs/qualitative_failures.png` for non-commercial, educational use.
- Hardware: Apple M4 (10-core GPU, 16 GB), PyTorch 2.15.0, Ultralytics 8.4.203.

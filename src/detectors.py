"""Uniform wrappers around the detectors evaluated in this homework.

Every wrapper exposes   predict(img_path) -> (boxes_xyxy[N,4], scores[N], label_names[N])
in ORIGINAL image pixel coordinates, so evaluation code is model-agnostic.

Families covered
  * torchvision  Faster R-CNN R50-FPN v2   two-stage, anchor-based (RPN + RoIAlign), NMS
  * torchvision  FCOS R50-FPN              one-stage, anchor-free (per-pixel ltrb + centerness), NMS
  * ultralytics  YOLO11 n/s/m              one-stage, anchor-free, DFL box head, TAL assignment, NMS
  * ultralytics  RT-DETR-L                 query-based transformer decoder, Hungarian matching, NMS-free
  * ultralytics  YOLO-World v2 (s)         open-vocabulary: CLIP text embeddings replace the class head
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

# Shared evaluation thresholds (documented in the report, Section "Reproducibility")
CONF_THRESH = 0.001   # low threshold -> full precision/recall curve for mAP
NMS_IOU = 0.7         # ultralytics default for validation; torchvision keeps its own (see below)
MAX_DET = 100         # COCOeval maxDets=100


def device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(dev: str):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def mem_mb(dev: str) -> float:
    if dev == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if dev == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return float("nan")


@dataclass
class Detector:
    name: str
    family: str
    checkpoint: str
    imgsz: int | None = None
    dev: str = field(default_factory=device_name)

    def predict(self, path: str):  # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------- ultralytics
class UltralyticsDetector(Detector):
    def __init__(self, name, family, checkpoint, imgsz=640, prompts: list[str] | None = None):
        super().__init__(name, family, checkpoint, imgsz)
        from ultralytics import RTDETR, YOLO, YOLOWorld

        if "rtdetr" in checkpoint:
            self.model = RTDETR(checkpoint)
        elif "world" in checkpoint:
            self.model = YOLOWorld(checkpoint)
        else:
            self.model = YOLO(checkpoint)
        self.prompts = prompts
        if prompts is not None:  # open-vocabulary: class head = text embeddings of these prompts
            self.model.set_classes(prompts)
        self.names = self.model.names

    def set_prompts(self, prompts: list[str]):
        self.prompts = prompts
        self.model.set_classes(prompts)
        self.names = self.model.names

    def predict(self, path):
        r = self.model.predict(
            path, imgsz=self.imgsz, conf=CONF_THRESH, iou=NMS_IOU, max_det=MAX_DET,
            device=self.dev, verbose=False,
        )[0]
        b = r.boxes
        cls = b.cls.cpu().numpy().astype(int)
        return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), [self.names[c] for c in cls]


# --------------------------------------------------------------------------- torchvision
class TorchvisionDetector(Detector):
    def __init__(self, name, family, builder: str):
        super().__init__(name, family, builder)
        from torchvision.models import detection as D

        fn = getattr(D, builder)
        from torchvision.models import get_model_weights
        weights = get_model_weights(fn).DEFAULT
        self.checkpoint = f"torchvision {builder} / {weights}"
        # Lower the score threshold so mAP integrates over the full PR curve (same as ultralytics).
        kw = dict(weights=weights, box_score_thresh=CONF_THRESH, box_detections_per_img=MAX_DET)
        if builder.startswith("fcos"):
            kw = dict(weights=weights, score_thresh=CONF_THRESH, detections_per_img=MAX_DET)
        self.model = fn(**kw).eval().to(self.dev)
        self.categories = weights.meta["categories"]  # index == COCO category id
        t = self.model.transform
        self.imgsz = f"min{t.min_size[0]}/max{t.max_size}"

    @torch.inference_mode()
    def predict(self, path):
        from torchvision.io import ImageReadMode, read_image

        img = read_image(path, ImageReadMode.RGB).float().div_(255).to(self.dev)
        out = self.model([img])[0]  # GeneralizedRCNNTransform resizes, postprocess maps back to orig coords
        labels = out["labels"].cpu().numpy()
        keep = np.array([self.categories[i] not in ("N/A", "__background__") for i in labels], dtype=bool)  # unused ids / bg
        return (out["boxes"].cpu().numpy()[keep], out["scores"].cpu().numpy()[keep],
                [self.categories[i] for i in labels[keep]])


# --------------------------------------------------------------------------- registry
COCO_PROMPTS = None  # filled lazily from ultralytics coco names for YOLO-World on COCO
KITTI_PROMPTS = ["car", "van", "truck", "pedestrian", "person sitting", "cyclist", "tram"]


def build(key: str, imgsz: int = 640) -> Detector:
    reg = {
        "frcnn_r50v2": lambda: TorchvisionDetector("Faster R-CNN R50-FPN v2", "two-stage anchor-based", "fasterrcnn_resnet50_fpn_v2"),
        "fcos_r50": lambda: TorchvisionDetector("FCOS R50-FPN", "one-stage anchor-free (FCOS)", "fcos_resnet50_fpn"),
        "yolo11n": lambda: UltralyticsDetector("YOLO11n", "one-stage anchor-free (YOLO)", "yolo11n.pt", imgsz),
        "yolo11s": lambda: UltralyticsDetector("YOLO11s", "one-stage anchor-free (YOLO)", "yolo11s.pt", imgsz),
        "yolo11m": lambda: UltralyticsDetector("YOLO11m", "one-stage anchor-free (YOLO)", "yolo11m.pt", imgsz),
        "rtdetr_l": lambda: UltralyticsDetector("RT-DETR-L", "query-based DETR", "rtdetr-l.pt", imgsz),
        "yoloworld_s": lambda: UltralyticsDetector("YOLO-World-v2-S", "open-vocabulary", "yolov8s-worldv2.pt", imgsz),
    }
    return reg[key]()


def timed_predict(det: Detector, path: str):
    sync(det.dev)
    t0 = time.perf_counter()
    out = det.predict(path)  # includes preprocess + forward + NMS/postprocess + copy to CPU
    sync(det.dev)
    return out, (time.perf_counter() - t0) * 1000.0

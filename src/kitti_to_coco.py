"""Convert the Ultralytics KITTI val split (YOLO txt labels) into a COCO-format GT json.

usage: python src/kitti_to_coco.py data/kitti data/kitti/kitti_val_coco.json
"""
import json
import sys
from pathlib import Path

from PIL import Image

from taxonomy import EVAL_CATS, EVAL_ID, GT_MAP, KITTI_RAW


def main(root: str, out: str, split: str = "val"):
    root = Path(root)
    images, anns = [], []
    for img_id, p in enumerate(sorted((root / "images" / split).glob("*.png")), start=1):
        w, h = Image.open(p).size
        images.append(dict(id=img_id, file_name=p.name, width=w, height=h))
        lbl = root / "labels" / split / (p.stem + ".txt")
        for line in lbl.read_text().split("\n") if lbl.exists() else []:
            if not line.strip():
                continue
            c, xc, yc, bw, bh = line.split()[:5]
            xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
            box = [xc - bw / 2, yc - bh / 2, bw, bh]
            for cat, crowd in GT_MAP[KITTI_RAW[int(c)]]:
                anns.append(dict(id=len(anns) + 1, image_id=img_id, category_id=EVAL_ID[cat], bbox=box,
                                 area=bw * bh, iscrowd=crowd, kitti_raw=KITTI_RAW[int(c)]))
    cats = [dict(id=k, name=v) for k, v in EVAL_CATS.items()]
    Path(out).write_text(json.dumps(dict(images=images, annotations=anns, categories=cats)))
    n = {v: sum(a["category_id"] == k and not a["iscrowd"] for a in anns) for k, v in EVAL_CATS.items()}
    print(f"{len(images)} images, {len(anns)} anns; non-crowd per class: {n}")


if __name__ == "__main__":
    main(*sys.argv[1:])

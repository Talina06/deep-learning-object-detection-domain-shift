"""KITTI <-> COCO label/taxonomy mapping (documented in the report, Table "Taxonomy mapping").

KITTI-eval categories (what we score on KITTI):
    1 car   2 pedestrian   3 cyclist   4 truck   5 tram

Ground truth (KITTI raw class -> KITTI-eval category). Following KITTI's official
"neighbouring class" rule, confusable classes are added as COCO *crowd/ignore* regions
(iscrowd=1): a detection that lands on them is neither a TP nor an FP, and missing them is
not an FN.
    Car             -> car
    Van             -> ignore(car), ignore(truck)        (COCO has no "van"; detectors say car OR truck)
    Truck           -> truck
    Pedestrian      -> pedestrian
    Person_sitting  -> ignore(pedestrian)                (KITTI official neighbour of Pedestrian)
    Cyclist         -> cyclist, ignore(pedestrian)       (COCO detectors fire "person" on the rider)
    Tram            -> tram
    Misc            -> ignore(car), ignore(truck)        (trailers, segways, ... not a COCO class)

Predictions (detector label name -> KITTI-eval category)
    closed-set COCO detectors: person->pedestrian, car->car, truck->truck, train->tram,
                               bicycle->cyclist   (see CYCLIST_MODE below)
    open-vocab YOLO-World with KITTI prompts: car/van->car, truck->truck,
                               pedestrian/person sitting->pedestrian, cyclist->cyclist, tram->tram
All other COCO classes (bus, motorcycle, traffic light, ...) are dropped.
"""

KITTI_RAW = ["car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram", "misc"]
EVAL_CATS = {1: "car", 2: "pedestrian", 3: "cyclist", 4: "truck", 5: "tram"}
EVAL_ID = {v: k for k, v in EVAL_CATS.items()}

# raw KITTI class -> list of (eval category, iscrowd)
GT_MAP = {
    "car": [("car", 0)],
    "van": [("car", 1), ("truck", 1)],
    "truck": [("truck", 0)],
    "pedestrian": [("pedestrian", 0)],
    "person_sitting": [("pedestrian", 1)],
    "cyclist": [("cyclist", 0), ("pedestrian", 1)],
    "tram": [("tram", 0)],
    "misc": [("car", 1), ("truck", 1)],
}

COCO_TO_KITTI = {"person": "pedestrian", "car": "car", "truck": "truck", "train": "tram", "bicycle": "cyclist"}

PROMPT_TO_KITTI = {
    "car": "car", "van": "car", "truck": "truck", "pedestrian": "pedestrian",
    "person sitting": "pedestrian", "cyclist": "cyclist", "tram": "tram",
    # prompt-ablation synonyms (results/yoloworld_s_kitti_640_prompts-*)
    "person": "pedestrian", "person riding a bicycle": "cyclist", "bicycle": None, "bus": None, "train": "tram",
}
PROMPT_TO_KITTI = {k: v for k, v in PROMPT_TO_KITTI.items() if v is not None}  # None = prompt kept only as a distractor

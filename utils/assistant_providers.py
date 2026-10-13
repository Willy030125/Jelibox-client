"""
The two newer Label Assistant providers, as plain functions that take their settings explicitly (no workspace
globals), so they can be used by the G key, by Auto-annotate all, and in tests.

  - locate_predictor : LocateAnything-3B through HF Transformers (utils/locate_anything.py)
  - head_predictor   : frozen YOLO detector + trained head (utils/custom_head.py)

A predictor is a function  predict(bgr_image) -> (predictions, is_polygon)  where predictions is
[{'rect': (x1, y1, x2, y2), 'conf': float, 'cls': workspace class name}]  (the same shape the YOLO providers
produce, so annotation_merge works on all of them).

Heavy things (torch, transformers, the models) are imported and loaded only when a predictor is asked for.
"""
import os

from . import locate_parsing as lp


class AssistantError(Exception):
    """A problem the user can act on; the message is shown as is."""


# how every Ultralytics detector is run when it draws boxes: a big input so far-away objects are found, and a low
# NMS IoU so one object gets one box
AUTO_IMGSZ = 1280
AUTO_IOU = 0.3


# ------------------------------------------------------------------ LocateAnything
def locate_targets(settings):
    return [t for t in settings["target_classes"] if t["prompt"].strip() and t["map_to"]]


def locate_predictor(settings, status=None):
    """settings: the 'locate_anything' block of the workspace config. Loads the model (downloading it the first
    time) and returns predict(bgr). `status(text)` receives progress lines while that happens."""
    targets = locate_targets(settings)
    if not targets:
        raise AssistantError("LocateAnything can't work without a target class. "
                             "Open Label Assistant and add at least one target class.")
    from . import locate_anything as la
    problem = la.check_requirements()
    if problem:
        raise AssistantError(problem)
    try:
        la.engine.load(settings["device"], status)
    except Exception as exc:
        raise AssistantError(f"LocateAnything could not be loaded:\n{exc}") from exc

    categories = []
    for t in targets:
        if t["prompt"].strip() not in categories:
            categories.append(t["prompt"].strip())

    def predict(bgr):
        from PIL import Image
        import cv2
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        boxes, _raw = la.engine.detect(pil, categories, settings)
        return lp.map_to_workspace(boxes, targets), False

    return predict


# ------------------------------------------------------------------ custom head
_head_cache = {}   # key -> HeadDetector; at most one is kept in memory


def map_head_classes(head_names, class_map, workspace_classes):
    """{head class name: workspace class or None}. An explicit mapping wins; a head class that has the same
    name as a workspace class (ignoring case) maps to it by itself; everything else stays unmapped (None) and
    its boxes are skipped."""
    by_lower = {c.lower(): c for c in workspace_classes}
    out = {}
    for name in head_names:
        target = class_map.get(name)
        if target in workspace_classes:
            out[name] = target
        else:
            out[name] = by_lower.get(name.lower())
    return out


def resolve_head_path(path, base_dir):
    """Absolute path of the head checkpoint; a relative path is taken from the Jelibox folder."""
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def head_predictor(settings, confidence, workspace_classes, base_dir, device="auto"):
    """settings: the 'custom_head' block of the workspace config; confidence: the detector's threshold."""
    if not settings["head_path"]:
        raise AssistantError("No head checkpoint is chosen. Open Label Assistant and pick your head_best.pt.")
    head_path = resolve_head_path(settings["head_path"], base_dir)
    if not os.path.isfile(head_path):
        raise AssistantError(f"The head checkpoint was not found:\n{head_path}")
    try:
        from . import custom_head as ch
    except Exception as exc:
        raise AssistantError(f"Custom head models need torch and ultralytics ({exc}).") from exc
    problem = ch.ultralytics_problem()
    if problem:
        raise AssistantError(problem)

    key = (head_path, os.path.getmtime(head_path), settings["detector_weights"], device)
    detector = _head_cache.get(key)
    if detector is None:
        _head_cache.clear()
        try:
            detector = ch.HeadDetector(head_path, settings["detector_weights"],
                                       os.path.join(base_dir, "models", "_detectors"), device)
        except Exception as exc:
            raise AssistantError(f"The head model could not be loaded:\n{exc}") from exc
        _head_cache[key] = detector

    mapping = map_head_classes(detector.names, settings["class_map"], workspace_classes)
    if not any(mapping.values()):
        raise AssistantError("None of the head's classes (" + ", ".join(detector.names) + ") is mapped to a "
                             "workspace class. Open Label Assistant and map them.")

    def predict(bgr):
        found = detector.predict(bgr, conf=confidence, iou=settings["iou"], classes=settings["detector_classes"],
                                 min_head_conf=settings["min_head_conf"])
        preds = [{"rect": f["rect"], "conf": f["det_conf"], "cls": mapping[f["head"]]}
                 for f in found if mapping.get(f["head"])]
        return preds, False

    return predict


# ------------------------------------------------------------------ stock detector only (no trained head yet)
_pre_cache = {}   # (weights path, device) -> YOLO; at most one is kept in memory


def detector_file(weights, base_dir):
    """Where a stock detector (yolov8m.pt, ...) lives: a path that exists as given, otherwise Jelibox's detectors folder."""
    if os.path.isabs(weights) or os.path.exists(weights):
        return weights
    return os.path.join(base_dir, "models", "_detectors", os.path.basename(weights))


def detector_predictor(weights, confidence, class_ids, label, base_dir, iou=AUTO_IOU, imgsz=AUTO_IMGSZ):
    """Boxes of a stock COCO detector, every one given the workspace class `label`. For drawing the boxes quickly
    (e.g. persons) before any head is trained; the classes are then set by hand or by the trained head later.
    A stock name that is not on disk is downloaded by ultralytics into models/_detectors."""
    if not label:
        raise AssistantError("Choose the class the detected boxes should get.")
    try:
        from . import custom_head as ch
        import torch
    except Exception as exc:
        raise AssistantError(f"The detector needs torch and ultralytics ({exc}).") from exc
    problem = ch.ultralytics_problem()
    if problem:
        raise AssistantError(problem)
    path = detector_file(weights, base_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    key = (path, torch.cuda.is_available())
    model = _pre_cache.get(key)
    if model is None:
        _pre_cache.clear()
        try:
            from ultralytics import YOLO
            model = YOLO(path)
        except Exception as exc:
            raise AssistantError(f"The detector {os.path.basename(weights)} could not be loaded:\n{exc}") from exc
        _pre_cache[key] = model
    classes = [int(i) for i in class_ids] or None

    def predict(bgr):
        with torch.no_grad():
            res = model.predict(bgr, conf=confidence, iou=iou, imgsz=imgsz, classes=classes, verbose=False)
        preds = []
        for r in res:
            for box in (r.boxes if getattr(r, "boxes", None) is not None else []):
                x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    preds.append({"rect": (x1, y1, x2, y2), "conf": float(box.conf[0].item()), "cls": label})
        return preds, False

    return predict


def release_all():
    """Forget the cached models so their memory can be freed."""
    _head_cache.clear()
    _pre_cache.clear()
    try:
        from . import locate_anything as la
        if la.engine.loaded:
            la.engine.unload()
    except Exception:
        pass


# ------------------------------------------------------------------ browsed Ultralytics models
def model_class_names(path):
    """Class names of an Ultralytics .pt file as a list indexed by class id (gaps are ''). Used by the settings
    windows to build the class mapping after "Browse custom model". Raises AssistantError."""
    if not os.path.isfile(path):
        raise AssistantError(f"Model file not found:\n{path}")
    try:
        from ultralytics import YOLO
        names = dict(YOLO(path).names)
    except Exception as exc:
        raise AssistantError(f"{os.path.basename(path)} could not be read as an Ultralytics model:\n{exc}") from exc
    if not names:
        raise AssistantError(f"{os.path.basename(path)} has no class names.")
    return [str(names.get(i, "")) for i in range(max(names) + 1)]

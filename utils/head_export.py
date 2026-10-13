"""
Export a workspace's labels for the command-line head trainer (train.py of the custom-heads trainer), so a head can be
trained on another computer, for example one with a GPU.

Layout written to <out_dir> (the layout the trainer's config describes):

    images/train/*.jpg  images/val/*.jpg
    labels_<head>/train/<name>.txt  labels_<head>/val/<name>.txt      `class cx cy w h`, normalized
    <head>.yaml                                                       detector, data, head (classes, anchors, labels)

Same rules as the Train button: the split is by position (the last share of the sorted images is validation), a
classify head uses images that have boxes, a detect head also uses images with an empty label file (nothing to find).
Segmentation rows are written as their bounding box, because the trainer reads 4 numbers after the class.
No torch here.
"""
import json
import os
import re
import shutil

from . import head_data


def safe_name(text):
    return re.sub(r"[^\w.-]+", "_", str(text)).strip("_") or "head"


def label_text(label_path):
    rows = head_data.read_label_rows(label_path)
    return "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in rows)


def config_text(*, head_name, head_type, classes, anchors, weights, taps, imgsz, epochs, batch, lr, flip):
    """The trainer's yaml (JSON flow style inside, so any class name is quoted correctly)."""
    j = json.dumps
    lines = [
        "# written by Jelibox (Train head > Export). Run:  python train.py --config <this file> --head " + head_name,
        f"detector: {{weights: {j(os.path.basename(weights))}, taps: {j(list(taps))}, imgsz: {j(list(imgsz))}}}",
        "data: {path: ., images: \"images/{split}\", splits: {train: train, val: val}}",
        "heads:",
        f"  {head_name}:",
        f"    type: {head_type}",
        f"    names: {j(list(classes))}",
    ]
    if head_type == "detect":
        lines.append(f"    attach_to: {j(list(anchors))}          # COCO classes the detections belong to (0 = person)")
    else:
        lines.append(f"    anchors: {j(list(anchors))}           # COCO classes the head classifies (0 = person)")
    lines += [
        f"    labels: {{format: yolo_txt, path: \"labels_{head_name}/{{split}}\"}}",
        f"    augment: {{flip: {str(bool(flip)).lower()}}}",
        f"    train: {{epochs: {int(epochs)}, batch: {int(batch)}, lr: {float(lr)}}}",
    ]
    return "\n".join(lines) + "\n"


def export_for_trainer(pairs, out_dir, *, head_name, head_type, classes, anchors, weights, taps, imgsz,
                       epochs=50, batch=16, lr=0.001, val_fraction=0.2, flip=True, progress=None):
    """pairs: [(image path, label path)] in the order to split. Copies the images and writes the labels + yaml.
    Returns dict(config, head, train, val, command). Existing files with the same names are overwritten."""
    name = safe_name(head_name)
    train, val = head_data.split_contiguous(pairs, val_fraction)
    total, done = len(pairs), 0
    for split, items in (("train", train), ("val", val)):
        img_dir = os.path.join(out_dir, "images", split)
        lab_dir = os.path.join(out_dir, f"labels_{name}", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        for image, label in items:
            shutil.copy2(image, os.path.join(img_dir, os.path.basename(image)))
            base = os.path.splitext(os.path.basename(image))[0]
            with open(os.path.join(lab_dir, base + ".txt"), "w", encoding="utf-8") as f:
                f.write(label_text(label))
            done += 1
            if progress:
                progress(done, total)
    config = os.path.join(out_dir, f"{name}.yaml")
    with open(config, "w", encoding="utf-8") as f:
        f.write(config_text(head_name=name, head_type=head_type, classes=classes, anchors=anchors, weights=weights,
                            taps=taps, imgsz=imgsz, epochs=epochs, batch=batch, lr=lr, flip=flip))
    return dict(config=config, head=name, train=len(train), val=len(val),
                command=f"python train.py --config {os.path.basename(config)} --head {name}")

"""
Train a custom head on a frozen YOLO detector - the training half of the Custom head mode.

Ported from TOBEADDED/custom_dual_heads (train.py + dataset.py), then extended with the second head kind. It runs as
its own process so a long run never freezes the annotation window:

    python -m utils.head_train <job.json>          (run from the Jelibox folder; the GUI does this)

The job file says which images / labels / classes / detector to use (see HeadTrainDialog). `head_type` is "classify"
(the box comes from the detector, the head learns its class; the default) or "detect" (the head learns the boxes
too: they are the workspace's boxes, images with an empty label file count as "nothing there"). `anchors` are the
detector class ids stored in the checkpoint for the command-line trainer's inference (classify: the boxes it relabels,
detect: the boxes its detections attach to); `head_name` names the head there. Progress goes to
stdout as lines `@@ {json}` (events: status, start, epoch, done, error) that the GUI turns into a progress window;
anything else printed is plain log text. The detector is never changed: its weights are fingerprinted before and after.

Needs torch + ultralytics >= 8.4.68. Import this module only when a head is actually trained.
"""
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import head_data
from .custom_head import HEAD_KINDS, BehaviorModel, decode, letterbox, ultralytics_problem

PREFIX = "@@ "


# ------------------------------------------------------------------ data
class HeadDataset(Dataset):
    """Items are (image CHW RGB uint8, boxes xyxy in letterboxed pixels, class ids). min_box: smallest box side (in
    letterboxed pixels) kept after augmentation."""

    def __init__(self, pairs, imgsz, n_classes, train=True, hflip=True, min_box=8):
        self.pairs, self.imgsz, self.nc, self.train, self.hflip = list(pairs), tuple(imgsz), n_classes, train, hflip
        self.min_box = min_box

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, label_path = self.pairs[i]
        data = np.fromfile(img_path, dtype=np.uint8)                       # also works for non-ASCII paths
        img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if img is None:
            raise OSError(f"could not read {img_path}")
        h0, w0 = img.shape[:2]
        rows = [r for r in head_data.read_label_rows(label_path) if 0 <= r[0] < self.nc]
        lab = np.array(rows, np.float32).reshape(-1, 5)
        cls = lab[:, 0].astype(np.int64)
        cx, cy, bw, bh = lab[:, 1] * w0, lab[:, 2] * h0, lab[:, 3] * w0, lab[:, 4] * h0
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).astype(np.float32)
        img, r, left, top = letterbox(img, self.imgsz)
        boxes = boxes * r + np.array([left, top, left, top], np.float32)
        if self.train:
            img, boxes, cls = self._augment(img, boxes, cls)
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))      # BGR -> RGB, CHW
        return torch.from_numpy(img), torch.from_numpy(boxes), torch.from_numpy(cls)

    def _augment(self, img, boxes, cls):
        H, W = img.shape[:2]
        s = random.uniform(0.75, 1.25)
        tx, ty = random.uniform(-0.1, 0.1) * W, random.uniform(-0.1, 0.1) * H
        M = np.array([[s, 0, (1 - s) * W / 2 + tx], [0, s, (1 - s) * H / 2 + ty]], np.float32)
        img = cv2.warpAffine(img, M, (W, H), borderValue=(114, 114, 114))
        if len(boxes):
            b = boxes.copy()
            b[:, [0, 2]] = b[:, [0, 2]] * s + M[0, 2]
            b[:, [1, 3]] = b[:, [1, 3]] * s + M[1, 2]
            a0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
            bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
            keep = (bw * bh / np.maximum(a0, 1e-6) > 0.5) & (bw > self.min_box) & (bh > self.min_box)
            boxes, cls = b[keep], cls[keep]
        if self.hflip and random.random() < 0.5:
            img = img[:, ::-1]
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= random.uniform(0.6, 1.4)
        hsv[..., 2] *= random.uniform(0.6, 1.4)
        img = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img, boxes, cls


def collate(batch):
    imgs, rois, labels = [], [], []
    for i, (im, b, c) in enumerate(batch):
        imgs.append(im)
        if len(b):
            rois.append(torch.cat([torch.full((len(b), 1), float(i)), b], 1))
            labels.append(c)
    imgs = torch.stack(imgs)
    rois = torch.cat(rois) if rois else torch.zeros(0, 5)
    labels = torch.cat(labels) if labels else torch.zeros(0, dtype=torch.long)
    return imgs, rois, labels


def jitter(boxes, hw, amt=0.1):
    """Simulate detector box noise so the head doesn't overfit to perfect labelled boxes."""
    h, w = hw
    bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    d = (torch.rand(len(boxes), 4, device=boxes.device) * 2 - 1) * amt
    out = boxes + d * torch.stack([bw, bh, bw, bh], 1)
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, h)
    ok = ((out[:, 2] - out[:, 0]) > 4) & ((out[:, 3] - out[:, 1]) > 4)
    return torch.where(ok[:, None], out, boxes)


def per_image(rois, labels, n):
    """rois (K, 5) = [batch idx, x1, y1, x2, y2] -> boxes and classes as one list entry per image."""
    return [rois[rois[:, 0] == b, 1:] for b in range(n)], [labels[rois[:, 0] == b] for b in range(n)]


# ------------------------------------------------------------------ detect head: targets, loss, metric
def build_targets(boxes_list, cls_list, nc, grid_hw, stride, device="cpu"):
    """Gaussian heat maps (B, nc, gh, gw) plus one row per object: (image, class, cell y, cell x, log w, log h,
    offset x, offset y), sizes in strides."""
    gh, gw = grid_hw
    heat = torch.zeros(len(boxes_list), nc, gh, gw, device=device)
    rows = []
    for b, (boxes, classes) in enumerate(zip(boxes_list, cls_list)):
        for box, c in zip(boxes.tolist(), classes.tolist()):
            w, h = box[2] - box[0], box[3] - box[1]
            if w < 1 or h < 1:
                continue
            gx, gy = (box[0] + box[2]) / 2 / stride, (box[1] + box[3]) / 2 / stride
            ix, iy = min(max(int(gx), 0), gw - 1), min(max(int(gy), 0), gh - 1)
            sx, sy = max(w / stride / 6, 0.6), max(h / stride / 6, 0.6)
            rx, ry = math.ceil(3 * sx), math.ceil(3 * sy)
            x0, x1, y0, y1 = max(ix - rx, 0), min(ix + rx + 1, gw), max(iy - ry, 0), min(iy + ry + 1, gh)
            xs = torch.arange(x0, x1, device=device, dtype=torch.float32) - ix
            ys = torch.arange(y0, y1, device=device, dtype=torch.float32) - iy
            g = torch.exp(-(xs[None, :] ** 2 / (2 * sx ** 2) + ys[:, None] ** 2 / (2 * sy ** 2)))
            heat[b, c, y0:y1, x0:x1] = torch.maximum(heat[b, c, y0:y1, x0:x1], g)
            rows.append([b, c, iy, ix, math.log(w / stride), math.log(h / stride),
                         min(max(gx - ix, 0.0), 0.999), min(max(gy - iy, 0.0), 0.999)])
    objs = torch.tensor(rows, dtype=torch.float32, device=device) if rows else torch.zeros(0, 8, device=device)
    return heat, objs


def focal_loss(logits, gt, alpha=2.0, beta=4.0):
    p = logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    pos_loss = -(p.log() * (1 - p) ** alpha * pos).sum()
    neg_loss = -((1 - p).log() * p ** alpha * (1 - gt) ** beta * (1 - pos)).sum()
    return (pos_loss + neg_loss) / pos.sum().clamp(min=1)


def detect_loss(out, heat, objs):
    loss = focal_loss(out["heat"], heat)
    if len(objs):
        b, iy, ix = objs[:, 0].long(), objs[:, 2].long(), objs[:, 3].long()
        loss = loss + F.l1_loss(out["size"][b, :, iy, ix].float(), objs[:, 4:6]) \
            + F.l1_loss(out["off"][b, :, iy, ix].float().sigmoid(), objs[:, 6:8])
    else:
        loss = loss + out["size"].sum() * 0 + out["off"].sum() * 0     # keep the graph when a batch has no boxes
    return loss


def iou_xyxy(a, b):
    a, b = np.asarray(a, np.float32).reshape(-1, 4), np.asarray(b, np.float32).reshape(-1, 4)
    iw = (np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])).clip(0)
    ih = (np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])).clip(0)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


class DetMeter:
    """AP@0.5 per class, plus precision / recall at a score threshold."""

    def __init__(self, nc, iou_thr=0.5):
        self.nc, self.thr = nc, iou_thr
        self.scores = [[] for _ in range(nc)]
        self.hits = [[] for _ in range(nc)]
        self.n_gt = np.zeros(nc, np.int64)

    def update(self, pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls):
        pred_boxes, pred_scores, pred_cls = np.asarray(pred_boxes), np.asarray(pred_scores), np.asarray(pred_cls)
        gt_boxes, gt_cls = np.asarray(gt_boxes, np.float32).reshape(-1, 4), np.asarray(gt_cls)
        for c in range(self.nc):
            g = gt_boxes[gt_cls == c]
            self.n_gt[c] += len(g)
            m = pred_cls == c
            if not m.any():
                continue
            order = np.argsort(-pred_scores[m])
            pb, ps = pred_boxes[m][order], pred_scores[m][order]
            used = np.zeros(len(g), bool)
            iou = iou_xyxy(pb, g) if len(g) else np.zeros((len(pb), 0))
            for k in range(len(pb)):
                hit = False
                if len(g):
                    j = int(np.argmax(np.where(used, -1, iou[k])))
                    if not used[j] and iou[k, j] >= self.thr:
                        used[j], hit = True, True
                self.scores[c].append(float(ps[k]))
                self.hits[c].append(hit)

    def summary(self, conf=0.3):
        ap, prec_at, rec_at = [], [], []
        for c in range(self.nc):
            s, t = np.array(self.scores[c]), np.array(self.hits[c], bool)
            n = max(int(self.n_gt[c]), 1)
            if not len(s):
                ap.append(0.0); prec_at.append(0.0); rec_at.append(0.0)
                continue
            o = np.argsort(-s)
            tp, fp = np.cumsum(t[o]), np.cumsum(~t[o])
            rec, prec = tp / n, tp / np.maximum(tp + fp, 1e-9)
            mrec = np.concatenate([[0], rec, [1]])
            mpre = np.maximum.accumulate(np.concatenate([[1], prec, [0]])[::-1])[::-1]
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap.append(float(((mrec[i + 1] - mrec[i]) * mpre[i + 1]).sum()))
            k = s >= conf
            tp_k = int(t[k].sum())
            prec_at.append(tp_k / max(int(k.sum()), 1))
            rec_at.append(tp_k / n)
        present = self.n_gt > 0                                   # a class with no validation boxes says nothing
        mean = lambda v: float(np.mean(np.array(v)[present])) if present.any() else 0.0
        return dict(ap50=mean(ap), precision=mean(prec_at), recall=mean(rec_at), recall_per_class=rec_at,
                    support=self.n_gt.tolist())


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, dl, nc, dev, amp):
    model.eval()
    cm = torch.zeros(nc, nc, dtype=torch.long)
    for imgs, rois, labels in dl:
        if len(rois) == 0:
            continue
        imgs = imgs.to(dev).float() / 255
        with torch.autocast(dev.type, enabled=amp):
            logits = model(imgs, rois[:, 0].long().to(dev), rois[:, 1:].to(dev))
        pred = logits.argmax(1).cpu()
        for t, p in zip(labels, pred):
            cm[t, p] += 1
    tp = cm.diag().float()
    rec = tp / cm.sum(1).clamp(min=1)
    prec = tp / cm.sum(0).clamp(min=1)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
    present = cm.sum(1) > 0                                   # a class with no validation boxes says nothing
    f1m = f1[present].mean().item() if present.any() else 0.0
    return dict(score=f1m, acc=(tp.sum() / cm.sum().clamp(min=1)).item(), f1=f1m,
                recall=rec.tolist(), support=cm.sum(1).tolist(), cm=cm)


@torch.no_grad()
def evaluate_detect(model, dl, nc, dev, amp, conf=0.3):
    model.eval()
    meter = DetMeter(nc)
    for imgs, rois, labels in dl:
        imgs = imgs.to(dev).float() / 255
        with torch.autocast(dev.type, enabled=amp):
            out = model(imgs)
        gb, gc = per_image(rois, labels, len(imgs))
        for (pb, ps, pc), b, c in zip(decode(out, model.head.stride, 0.05), gb, gc):
            meter.update(pb.cpu().numpy(), ps.cpu().numpy(), pc.cpu().numpy(), b.numpy(), c.numpy())
    s = meter.summary(conf)
    return dict(score=s["ap50"], ap50=s["ap50"], precision=s["precision"], recall=s["recall_per_class"],
                recall_mean=s["recall"], support=s["support"])


# ------------------------------------------------------------------ training
def emit(event, **fields):
    print(PREFIX + json.dumps({"event": event, **fields}), flush=True)


def resolve_weights(weights, detectors_dir):
    """A stock name (yolov8m.pt) that is not on disk is fetched into the detectors folder, like at annotation time."""
    if detectors_dir and not os.path.isabs(weights) and not os.path.exists(weights):
        os.makedirs(detectors_dir, exist_ok=True)
        return os.path.join(detectors_dir, os.path.basename(weights))
    return weights


def train(job, emit=emit, should_stop=lambda: False):
    """Run one training job (a dict, see the module doc). Returns the summary dict sent with the `done` event."""
    problem = ultralytics_problem()
    if problem:
        raise RuntimeError(problem)
    classes = [str(c) for c in job["classes"]]
    nc = len(classes)
    kind = job.get("head_type", "classify")
    if kind not in HEAD_KINDS:
        raise ValueError(f"Unknown head type {kind!r}; use one of {', '.join(HEAD_KINDS)}.")
    detect = kind == "detect"
    imgsz = tuple(int(v) for v in job["imgsz"])
    if len(imgsz) != 2 or any(v < 32 or v % 32 for v in imgsz):
        raise ValueError(f"Image size {imgsz[0]}x{imgsz[1]} must be two multiples of 32 (height, width).")
    out = job["out_dir"]
    os.makedirs(out, exist_ok=True)

    pairs = head_data.labelled_pairs(job["images_folders"], job["labels_folder"], allow_empty=detect)
    boxed = sum(1 for _, label in pairs if head_data.read_label_rows(label)) if detect else len(pairs)
    if boxed < head_data.MIN_LABELLED:
        raise ValueError(f"Only {boxed} {'images with boxes' if detect else 'labelled images'}; "
                         f"at least {head_data.MIN_LABELLED} are needed.")
    train_pairs, val_pairs = head_data.split_contiguous(pairs, float(job.get("val_fraction", 0.2)))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    batch = max(1, min(int(job["batch"]), len(train_pairs)))
    workers = int(job.get("workers", 2))
    min_box = 3 if detect else 8
    tr = HeadDataset(train_pairs, imgsz, nc, True, bool(job.get("hflip", True)), min_box)
    va = HeadDataset(val_pairs, imgsz, nc, False, min_box=min_box)
    tdl = DataLoader(tr, batch, shuffle=True, num_workers=workers, collate_fn=collate,
                     drop_last=len(tr) > batch, pin_memory=amp)
    vdl = DataLoader(va, batch, shuffle=False, num_workers=workers, collate_fn=collate)

    emit("status", text="Loading the detector ...")
    weights = resolve_weights(job["weights"], job.get("detectors_dir"))
    taps = tuple(int(t) for t in job["taps"])
    model = BehaviorModel(weights, nc, taps=taps, kind=kind).to(dev)
    if not model.backbone.strides_ok():
        raise ValueError(f"Layers {list(taps)} of {os.path.basename(weights)} are not at strides 8 / 16 / 32. "
                         f"Pick the neck layers that match this detector (see Taps).")
    fingerprint = model.backbone.fingerprint()

    counts = head_data.class_counts(train_pairs, nc)
    if not any(counts):
        raise ValueError("The training labels use none of the workspace classes.")
    crit = None
    if not detect:
        weight = None
        if job.get("class_weights", True):
            c = torch.tensor(counts, dtype=torch.float)
            present = c > 0                                    # a class with no examples must not soak up weight
            weight = torch.zeros(nc)
            weight[present] = (1 / c[present]).sqrt()
            weight = (weight / weight[present].mean()).to(dev)
        crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)

    epochs = int(job["epochs"])
    opt = torch.optim.AdamW(model.head.parameters(), lr=float(job["lr"]), weight_decay=0.01)
    total, warm = epochs * len(tdl), min(300, len(tdl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except AttributeError:                                      # torch older than 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    emit("start", kind=kind, train_images=len(train_pairs), val_images=len(val_pairs), classes=classes, counts=counts,
         epochs=epochs, device=dev.type, batch=batch, imgsz=list(imgsz),
         head_params_m=round(sum(p.numel() for p in model.head.parameters()) / 1e6, 2))

    best, best_epoch, stopped, last_metrics = -1.0, 0, False, None
    for ep in range(epochs):
        if should_stop():
            stopped = True
            break
        model.train()                                          # the detector inside stays in eval() by design
        t0, run, n = time.time(), 0.0, 0
        for imgs, rois, labels in tdl:
            if not detect and len(rois) == 0:                  # a detect head also learns from images with no boxes
                continue
            imgs = imgs.to(dev, non_blocking=True).float() / 255
            rois, labels = rois.to(dev), labels.to(dev)
            if detect:
                with torch.autocast(dev.type, enabled=amp):
                    pred = model(imgs)
                gb, gc = per_image(rois, labels, len(imgs))
                heat, objs = build_targets(gb, gc, nc, pred["heat"].shape[-2:], model.head.stride, dev)
                loss = detect_loss(pred, heat, objs)
            else:
                boxes = jitter(rois[:, 1:], imgs.shape[-2:])
                with torch.autocast(dev.type, enabled=amp):
                    logits = model(imgs, rois[:, 0].long(), boxes)
                loss = crit(logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            n += 1
        m = evaluate_detect(model, vdl, nc, dev, amp) if detect else evaluate(model, vdl, nc, dev, amp)
        last_metrics = m
        ids = [int(a) for a in job.get("anchors", [])]          # classify: the detector classes it relabels; detect: the
        spec = dict(type=kind, names=classes, anchors=[] if detect else ids,    # ones its boxes attach to (train.py)
                    attach_to=ids if detect else [], head_args={})
        ck = dict(head=model.head.state_dict(), name=str(job.get("head_name") or "head"), spec=spec, names=classes,
                  cfg=dict(weights=os.path.basename(job["weights"]), taps=list(taps), imgsz=list(imgsz),
                           strides=[8, 16, 32], nc=nc))
        torch.save(ck, os.path.join(out, "head_last.pt"))
        is_best = m["score"] > best
        if is_best:
            best, best_epoch = m["score"], ep + 1
            torch.save(ck, os.path.join(out, "head_best.pt"))
        fields = dict(ap50=m["ap50"], precision=m["precision"]) if detect else dict(acc=m["acc"], f1=m["f1"])
        emit("epoch", kind=kind, epoch=ep + 1, epochs=epochs, loss=run / max(n, 1), score=m["score"], best=is_best,
             recall=m["recall"], support=m["support"], seconds=time.time() - t0, **fields)

    if model.backbone.fingerprint() != fingerprint:
        raise RuntimeError("The detector's weights changed during training - this must never happen.")
    summary = dict(kind=kind, metric="AP50" if detect else "macro-F1", best_f1=max(best, 0.0), best_epoch=best_epoch,
                   epochs_run=ep + (0 if stopped else 1), stopped=stopped, out_dir=out,
                   has_head=os.path.isfile(os.path.join(out, "head_best.pt")), classes=classes,
                   last_recall=(last_metrics or {}).get("recall"), last_support=(last_metrics or {}).get("support"))
    emit("done", **summary)
    return summary


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m utils.head_train <job.json>")
        return 2
    with open(argv[0], "r", encoding="utf-8") as f:
        job = json.load(f)
    try:
        train(job)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        emit("error", message=str(exc) or exc.__class__.__name__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

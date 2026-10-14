"""Augmentation for the dataset exporter, in the style of Roboflow's augmentation step.

Every augmentation is one entry of OPS: a label, a short description, and a few parameters (sliders and
checkboxes) that set the *range* it may draw from. For each augmented copy the enabled augmentations each draw a
random value inside their range (`sample`) and are applied (`apply`); the preview in the export dialog uses the
same two functions with fixed values (`variants`).

Colour and effect augmentations (brightness, exposure, hue, saturation, grayscale, blur, cutout) use
torchvision.transforms.v2.functional. Geometry (flip, 90 rotate, rotation, shear, crop) is warped with the matrix
of augment_geometry, so the labels move exactly with the pixels. Noise and motion blur are plain numpy / OpenCV.
torch and torchvision are imported on first use only.

The exporter writes the outputs as <name>_aug1, ... for the Train split only ("Nx" as in Roboflow: the outputs
stand in for the image, unless the original is kept as well).
"""
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .augment_geometry import CropTransform, GeoTransform, ResizeTransform, compose

MAX_COPIES = 10
MAX_REROLLS = 20


# ----------------------------------------------------------------------
#  Catalogue
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Param:
    name: str
    label: str
    kind: str                     # "slider" or "check"
    default: float
    lo: float = 0
    hi: float = 0
    unit: str = ""


@dataclass(frozen=True)
class Op:
    key: str
    label: str
    group: str                    # "geometry", "color" or "effects"
    description: str
    params: tuple


OPS = {op.key: op for op in (
    Op("flip", "Flip", "geometry",
       "Add horizontal or vertical flips to help your model be insensitive to subject orientation.",
       (Param("horizontal", "Horizontal", "check", 1), Param("vertical", "Vertical", "check", 0))),
    Op("rot90", "90° Rotate", "geometry",
       "Add 90-degree rotations to help your model be insensitive to camera orientation.",
       (Param("clockwise", "Clockwise", "check", 1), Param("counter", "Counter-Clockwise", "check", 1),
        Param("upside_down", "Upside Down", "check", 0))),
    Op("crop", "Crop", "geometry",
       "Add variability to positioning and size to help your model be more resilient to subject translations "
       "and camera position.",
       (Param("min_zoom", "Minimum zoom", "slider", 0, 0, 90, "%"),
        Param("max_zoom", "Maximum zoom", "slider", 20, 1, 90, "%"))),
    Op("rotation", "Rotation", "geometry",
       "Add variability to rotations to help your model be more resilient to camera roll.",
       (Param("degrees", "Rotate between -N° and +N°", "slider", 15, 1, 90, "°"),)),
    Op("shear", "Shear", "geometry",
       "Add variability to perspective to help your model be more resilient to camera and subject pitch and yaw.",
       (Param("horizontal", "Horizontal", "slider", 10, 0, 45, "°"),
        Param("vertical", "Vertical", "slider", 10, 0, 45, "°"))),
    Op("grayscale", "Grayscale", "color",
       "Turn a share of the images grayscale to help your model focus on shapes rather than colours.",
       (Param("percent", "Apply to this share of the images", "slider", 15, 1, 100, "%"),)),
    Op("hue", "Hue", "color",
       "Randomly shift the colours to help your model be more resilient to colour changes.",
       (Param("degrees", "Shift between -N° and +N°", "slider", 25, 1, 180, "°"),)),
    Op("saturation", "Saturation", "color",
       "Add variability to colour saturation to help your model be more resilient to camera settings.",
       (Param("percent", "Between -N% and +N%", "slider", 25, 1, 100, "%"),)),
    Op("brightness", "Brightness", "color",
       "Add variability to image brightness to help your model be more resilient to lighting and camera "
       "setting changes.",
       (Param("percent", "Between -N% and +N%", "slider", 15, 1, 99, "%"),
        Param("brighten", "Brighten", "check", 1), Param("darken", "Darken", "check", 1))),
    Op("exposure", "Exposure", "color",
       "Add variability to image exposure to help your model be more resilient to lighting and camera "
       "setting changes.",
       (Param("percent", "Between -N% and +N%", "slider", 15, 1, 100, "%"),)),
    Op("blur", "Blur", "effects",
       "Add random Gaussian blur to help your model be more resilient to camera focus.",
       (Param("pixels", "Up to this many pixels", "slider", 3, 1, 25, "px"),)),
    Op("noise", "Noise", "effects",
       "Add salt-and-pepper noise to help your model be more resilient to camera artifacts.",
       (Param("percent", "Up to this share of the pixels", "slider", 5, 1, 25, "%"),)),
    Op("cutout", "Cutout", "effects",
       "Black out random boxes to help your model be more resilient to objects that are partly hidden. "
       "The labels are not changed.",
       (Param("boxes", "Number of boxes", "slider", 3, 1, 10, ""),
        Param("size", "Box size (share of the image side)", "slider", 10, 1, 50, "%"))),
    Op("motion_blur", "Motion Blur", "effects",
       "Add motion blur in a random direction to help your model be more resilient to camera and subject movement.",
       (Param("length", "Up to this many pixels", "slider", 15, 3, 51, "px"),)),
)}

# pixel-level steps first, geometry last (so the black corners of a rotation stay black)
_PIXEL_ORDER = ("brightness", "exposure", "hue", "saturation", "grayscale", "blur", "motion_blur", "noise", "cutout")
_GEOMETRY_ORDER = ("crop", "flip", "rot90", "shear", "rotation")


def default_params(key):
    return {p.name: (bool(p.default) if p.kind == "check" else p.default) for p in OPS[key].params}


def clean_params(key, raw):
    """`raw` (maybe incomplete or out of range) -> a full, clamped parameter dict for OPS[key]."""
    out = {}
    for p in OPS[key].params:
        value = (raw or {}).get(p.name, p.default)
        if p.kind == "check":
            out[p.name] = bool(value)
        else:
            try:
                out[p.name] = max(p.lo, min(p.hi, float(value)))
            except (TypeError, ValueError):
                out[p.name] = p.default
    if key == "crop" and out["min_zoom"] > out["max_zoom"]:
        out["min_zoom"] = out["max_zoom"]
    return out


def summary(key, params):
    """A short description of the enabled augmentation, e.g. 'Brightness ±15%'."""
    p = clean_params(key, params)
    label = OPS[key].label
    if key == "flip":
        return label + " (" + "/".join(n for n, on in (("horizontal", p["horizontal"]), ("vertical", p["vertical"])) if on) + ")"
    if key == "rot90":
        names = [n for n, k in (("clockwise", "clockwise"), ("counter-clockwise", "counter"),
                                ("upside down", "upside_down")) if p[k]]
        return label + " (" + ", ".join(names) + ")"
    if key == "crop":
        return f"{label} {p['min_zoom']:g}-{p['max_zoom']:g}% zoom"
    if key == "shear":
        return f"{label} ±{p['horizontal']:g}° / ±{p['vertical']:g}°"
    if key == "grayscale":
        return f"{label} {p['percent']:g}% of images"
    if key == "cutout":
        return f"{label} {p['boxes']:g} × {p['size']:g}%"
    if key == "brightness":
        sign = "±" if p["brighten"] and p["darken"] else ("+" if p["brighten"] else "-")
        return f"{label} {sign}{p['percent']:g}%"
    unit = OPS[key].params[0].unit
    sign = "up to " if key in ("blur", "noise", "motion_blur") else "±"
    return f"{label} {sign}{p[OPS[key].params[0].name]:g}{unit}"


# ----------------------------------------------------------------------
#  Config
# ----------------------------------------------------------------------
@dataclass
class AugmentConfig:
    ops: dict = field(default_factory=dict)       # key -> parameter dict; present = enabled
    copies: int = 1                                # augmented outputs per Train image
    keep_original: bool = False                    # also write the original next to its outputs
    seed: int = None                               # None = different every run
    resize: object = None                          # export_preprocess.ResizeConfig: copies are brought back to this size

    @property
    def enabled(self):
        return self.copies > 0 and any(k in OPS for k in self.ops)


# ----------------------------------------------------------------------
#  torch helpers (imported on first use)
# ----------------------------------------------------------------------
def _tv():
    import torch
    from torchvision.transforms.v2 import functional as F
    return torch, F


def _to_tensor(bgr):
    torch, _ = _tv()
    return torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).permute(2, 0, 1).contiguous()


def _to_bgr(tensor):
    return np.ascontiguousarray(tensor.permute(1, 2, 0).numpy()[..., ::-1])


def _odd(n):
    n = int(round(n))
    return n if n % 2 else n + 1


# ----------------------------------------------------------------------
#  sample: draw the random values of one copy
# ----------------------------------------------------------------------
def sample(key, params, rng, size):
    """Random values for one copy. `size` = (w, h) of the image (crop and cutout place things in it)."""
    p = clean_params(key, params)
    w, h = size
    if key == "flip":
        axes = [(True, False)] * p["horizontal"] + [(False, True)] * p["vertical"] + \
               [(True, True)] * (p["horizontal"] and p["vertical"])
        flip_h, flip_v = axes[int(rng.integers(len(axes)))] if axes else (False, False)
        return {"h": flip_h, "v": flip_v}
    if key == "rot90":
        turns = [k for k, on in ((-1, p["clockwise"]), (1, p["counter"]), (2, p["upside_down"])) if on]
        return {"k": turns[int(rng.integers(len(turns)))] if turns else 0}
    if key == "crop":
        return {"zoom": rng.uniform(p["min_zoom"], p["max_zoom"]) / 100.0, "ux": rng.random(), "uy": rng.random()}
    if key == "rotation":
        return {"angle": rng.uniform(-p["degrees"], p["degrees"])}
    if key == "shear":
        return {"sx": rng.uniform(-p["horizontal"], p["horizontal"]), "sy": rng.uniform(-p["vertical"], p["vertical"])}
    if key == "grayscale":
        return {"on": rng.random() * 100.0 < p["percent"]}
    if key == "hue":
        return {"shift": rng.uniform(-p["degrees"], p["degrees"])}
    if key == "saturation":
        return {"pct": rng.uniform(-p["percent"], p["percent"])}
    if key == "brightness":
        lo = -p["percent"] if p["darken"] else 0.0
        hi = p["percent"] if p["brighten"] else 0.0
        return {"pct": rng.uniform(lo, hi) if hi > lo else 0.0}
    if key == "exposure":
        return {"pct": rng.uniform(-p["percent"], p["percent"])}
    if key == "blur":
        top = _odd(max(3, p["pixels"]))
        return {"k": int(rng.choice(list(range(3, top + 1, 2))))}
    if key == "noise":
        return {"pct": rng.uniform(p["percent"] / 4.0, p["percent"]), "seed": int(rng.integers(2 ** 31))}
    if key == "cutout":
        side = p["size"] / 100.0
        boxes = [(rng.random(), rng.random(), side) for _ in range(int(p["boxes"]))]
        return {"boxes": boxes}
    if key == "motion_blur":
        return {"length": int(rng.integers(3, max(3, int(p["length"])) + 1)), "angle": rng.uniform(0, 180)}
    raise KeyError(key)


# ----------------------------------------------------------------------
#  apply: values -> (image, transform or None). `scale` shrinks pixel sizes for a downscaled preview.
# ----------------------------------------------------------------------
def apply(key, img, v, scale=1.0):
    h, w = img.shape[:2]
    if key == "flip":
        if not (v["h"] or v["v"]):
            return img, None
        tf = GeoTransform(w, h, v["h"], v["v"])
        return tf.warp(img), tf
    if key == "rot90":
        if not v["k"]:
            return img, None
        tf = GeoTransform(w, h, angle=90.0 * v["k"])
        return tf.warp(img), tf
    if key == "crop":
        cw, ch = max(1, round(w * (1 - v["zoom"]))), max(1, round(h * (1 - v["zoom"])))
        if cw >= w and ch >= h:
            return img, None
        tf = CropTransform(w, h, round(v["ux"] * (w - cw)), round(v["uy"] * (h - ch)), cw, ch)
        return tf.warp(img), tf
    if key == "rotation":
        tf = GeoTransform(w, h, angle=v["angle"])
        return tf.warp(img), tf
    if key == "shear":
        tf = GeoTransform(w, h, shear_x=v["sx"], shear_y=v["sy"])
        return tf.warp(img), tf

    _, F = _tv()
    if key == "noise":
        rng = np.random.default_rng(v["seed"])
        mask = rng.random((h, w)) < v["pct"] / 100.0
        salt = rng.random((h, w)) < 0.5
        out = img.copy()
        out[mask & salt] = 255
        out[mask & ~salt] = 0
        return out, None
    if key == "motion_blur":
        length = max(1, int(round(v["length"] * scale)))
        if length < 2:
            return img, None
        kernel = np.zeros((length, length), np.float32)
        centre = (length - 1) / 2.0
        dx, dy = math.cos(math.radians(v["angle"])), math.sin(math.radians(v["angle"]))
        cv2.line(kernel, (round(centre - dx * centre), round(centre - dy * centre)),
                 (round(centre + dx * centre), round(centre + dy * centre)), 1.0, 1)
        return cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6), borderType=cv2.BORDER_REPLICATE), None

    t = _to_tensor(img)
    if key == "grayscale":
        if v["on"]:
            t = F.rgb_to_grayscale(t, num_output_channels=3)
    elif key == "hue":
        t = F.adjust_hue(t, max(-0.5, min(0.5, v["shift"] / 360.0)))
    elif key == "saturation":
        t = F.adjust_saturation(t, max(0.0, 1.0 + v["pct"] / 100.0))
    elif key == "brightness":
        t = F.adjust_brightness(t, max(0.0, 1.0 + v["pct"] / 100.0))
    elif key == "exposure":
        t = F.adjust_gamma(t, 1.0 / max(0.05, 1.0 + v["pct"] / 100.0))
    elif key == "blur":
        k = _odd(max(3, v["k"] * scale))
        t = F.gaussian_blur(t, [k, k])
    elif key == "cutout":
        torch, _ = _tv()
        for ux, uy, side in v["boxes"]:
            bw, bh = max(1, round(w * side)), max(1, round(h * side))
            t = F.erase(t, round(uy * (h - bh)), round(ux * (w - bw)), bh, bw, torch.zeros(3, 1, 1, dtype=t.dtype))
    else:
        raise KeyError(key)
    return _to_bgr(t), None


# ----------------------------------------------------------------------
#  variants: what the preview shows for an augmentation with these parameters
# ----------------------------------------------------------------------
def variants(key, params):
    """[(caption, values)] - the fixed examples shown next to the original in the preview."""
    p = clean_params(key, params)
    out = []
    if key == "flip":
        if p["horizontal"]:
            out.append(("horizontal", {"h": True, "v": False}))
        if p["vertical"]:
            out.append(("vertical", {"h": False, "v": True}))
    elif key == "rot90":
        for name, k, on in (("clockwise", -1, p["clockwise"]), ("counter-clockwise", 1, p["counter"]),
                            ("upside down", 2, p["upside_down"])):
            if on:
                out.append((name, {"k": k}))
    elif key == "crop":
        for zoom in sorted({p["min_zoom"], p["max_zoom"]} - {0.0}):
            out.append((f"{zoom:g}% zoom", {"zoom": zoom / 100.0, "ux": 0.5, "uy": 0.5}))
    elif key == "rotation":
        out = [(f"-{p['degrees']:g}°", {"angle": -p["degrees"]}), (f"{p['degrees']:g}°", {"angle": p["degrees"]})]
    elif key == "shear":
        if p["horizontal"]:
            out.append((f"horizontal {p['horizontal']:g}°", {"sx": p["horizontal"], "sy": 0.0}))
        if p["vertical"]:
            out.append((f"vertical {p['vertical']:g}°", {"sx": 0.0, "sy": p["vertical"]}))
    elif key == "grayscale":
        out = [("grayscale", {"on": True})]
    elif key == "hue":
        out = [(f"-{p['degrees']:g}°", {"shift": -p["degrees"]}), (f"{p['degrees']:g}°", {"shift": p["degrees"]})]
    elif key in ("saturation", "exposure"):
        out = [(f"-{p['percent']:g}%", {"pct": -p["percent"]}), (f"{p['percent']:g}%", {"pct": p["percent"]})]
    elif key == "brightness":
        if p["darken"]:
            out.append((f"-{p['percent']:g}%", {"pct": -p["percent"]}))
        if p["brighten"]:
            out.append((f"{p['percent']:g}%", {"pct": p["percent"]}))
    elif key == "blur":
        out = [(f"{p['pixels']:g}px", {"k": _odd(max(3, p["pixels"]))})]
    elif key == "noise":
        out = [(f"{p['percent']:g}% of pixels", {"pct": p["percent"], "seed": 1})]
    elif key == "cutout":
        n = int(p["boxes"])
        out = [(f"{n} box" + ("es" if n > 1 else ""),
                {"boxes": [((i * 0.37 + 0.13) % 1.0, (i * 0.61 + 0.29) % 1.0, p["size"] / 100.0) for i in range(n)]})]
    elif key == "motion_blur":
        out = [(f"{p['length']:g}px", {"length": int(p["length"]), "angle": 20.0})]
    return out


def preview(key, params, img, scale=1.0):
    """[(caption, image)] for the preview of one augmentation: the original first, then each variant."""
    shots = [("original", img)]
    for caption, values in variants(key, params):
        shots.append((caption, apply(key, img, values, scale)[0]))
    return shots


# ----------------------------------------------------------------------
#  A whole augmented copy
# ----------------------------------------------------------------------
def augment_image(img, config, rng):
    """Apply every enabled augmentation once, each with its own random values.

    Returns (image, [keys applied], transform or None). The transform maps the labels of the input onto the
    output (None when only pixel values changed). A copy that nothing touched would just be a duplicate, so
    the draw is repeated (up to MAX_REROLLS times); if it still changes nothing, (None, [], None) is returned
    and the caller skips that copy."""
    active = {k: v for k, v in config.ops.items() if k in OPS}
    if not active:
        return None, [], None
    for _ in range(MAX_REROLLS):
        out, tfs, applied = img, [], []
        for key in _PIXEL_ORDER + _GEOMETRY_ORDER:
            if key not in active:
                continue
            before = out
            out, tf = apply(key, out, sample(key, active[key], rng, (out.shape[1], out.shape[0])))
            if tf is not None:
                tfs.append(tf)
                applied.append(key)
            elif out is not before and not np.array_equal(out, before):
                applied.append(key)
        if applied:
            break
    else:
        return None, [], None
    return out, applied, compose(*tfs)


# ----------------------------------------------------------------------
#  Image I/O that survives non-ASCII Windows paths (cv2.imread does not)
# ----------------------------------------------------------------------
def read_image(path):
    """BGR image with the EXIF orientation applied (the same pixels Jelibox annotated), or None."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except OSError:
        return None


def write_image(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Could not encode {path}")
    buf.tofile(path)


class Augmenter:
    """What the dataset exporter talks to: hand it a (prepared) image, get back the augmented copies to write
    next to the original."""

    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.created = 0
        self.skipped = 0

    def copies(self, img):
        """Yield (suffix, image, transform) for each augmented version, e.g. ("_aug1", array, transform-or-None).
        Move the annotations with the transform when it is not None; otherwise they are copied unchanged. With a
        resize step in the config, every copy is brought back to that size."""
        if img is None or not self.config.enabled:
            return
        resize = self.config.resize
        for i in range(1, min(MAX_COPIES, self.config.copies) + 1):
            out, _, tf = augment_image(img, self.config, self.rng)
            if out is None:
                self.skipped += 1
                continue
            if resize is not None and (out.shape[1], out.shape[0]) != (resize.width, resize.height):
                back = ResizeTransform(out.shape[1], out.shape[0], resize.width, resize.height, resize.mode)
                out, tf = back.warp(out), compose(tf, back)
            self.created += 1
            yield f"_aug{i}", out, tf

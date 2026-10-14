"""Preprocessing of the images of an exported dataset (applied to every split): Auto-Orient and Resize.

Jelibox annotates the picture as OpenCV loads it, i.e. with the EXIF orientation already applied. A file that
still carries an orientation tag other than 1 is therefore shown rotated by Jelibox but unrotated by a tool that
ignores EXIF, and the labels then no longer fit. *Auto-Orient* writes such an image with its pixels turned and the
tag gone. Images without a rotating tag are copied untouched (their metadata stays).

Resize changes the pixels, so the labels are moved with the same matrix (augment_geometry.ResizeTransform).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from .augment_geometry import RESIZE_MODES, ResizeTransform
from .augmentation import read_image

ORIENTATION_TAG = 0x0112


@dataclass
class ResizeConfig:
    mode: str = "stretch"
    width: int = 640
    height: int = 640

    def __post_init__(self):
        if self.mode not in RESIZE_MODES:
            raise ValueError(f"unknown resize mode: {self.mode!r}")
        self.width, self.height = max(1, int(self.width)), max(1, int(self.height))


@dataclass
class Preprocess:
    auto_orient: bool = True
    resize: Optional[ResizeConfig] = None


def exif_orientation(path):
    """The EXIF orientation of the file (1 = upright, also when there is none or it cannot be read)."""
    try:
        with Image.open(path) as im:
            return int(im.getexif().get(ORIENTATION_TAG, 1) or 1)
    except Exception:
        return 1


def oriented_size(path):
    """(width, height) of the picture as Jelibox sees it (EXIF orientation applied)."""
    with Image.open(path) as im:
        w, h = im.size
    return (h, w) if exif_orientation(path) in (5, 6, 7, 8) else (w, h)


def prepare(path, pre):
    """(image, transform) when the file has to be rewritten, else (None, None) and the file is copied as it is.
    The transform maps the labels (in the picture Jelibox shows) to the rewritten image; None = unchanged."""
    rotated = pre.auto_orient and exif_orientation(path) != 1
    if not rotated and pre.resize is None:
        return None, None
    img = read_image(path)
    if img is None:
        return None, None
    if pre.resize is None:
        return img, None
    r = pre.resize
    tf = ResizeTransform(img.shape[1], img.shape[0], r.width, r.height, r.mode)
    return tf.warp(img), tf


def resize_preview(img, pre):
    """The picture after the preprocessing (for the previews in the export dialog)."""
    if pre.resize is None:
        return img
    r = pre.resize
    return ResizeTransform(img.shape[1], img.shape[0], r.width, r.height, r.mode).warp(img)

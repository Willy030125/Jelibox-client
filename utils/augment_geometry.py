"""Geometry for the dataset exporter: one affine matrix moves the pixels *and* the
annotations, so image and labels can't drift apart.

Coordinates are image-edge coordinates (0..W, 0..H), the same convention the
annotations use (a box covering the whole image is 0,0,W,H).

Transforms (all share AffineTransform, so the label writers treat them alike and
they can be chained with `compose`):
  GeoTransform    flip, rotation, shear. Nothing is scaled or cropped: the canvas grows just
                  enough to keep every pixel of the original, new corners are black.
  CropTransform   a window of the image scaled back to the original size (zoom in).
  ResizeTransform the export's Resize step (stretch, fill with centre crop, fit with edges).

A bounding box that gets rotated can only be stored as an upright box, so it
becomes the tightest upright box around the rotated one (looser than the
object, an unavoidable limit of box labels). Polygons - and YOLO segmentation
labels, where boxes are already stored as 4-point polygons - move exactly.
A crop or a fill can push an object out of the picture: `keeps_box` says when
too little of it is left, and the writers then drop that object.
"""
import copy
import math
import xml.etree.ElementTree as ET

import cv2
import numpy as np

MIN_VISIBLE = 0.2        # an object is dropped when less than this share of its box stays in the picture


class AffineTransform:
    def __init__(self, src_w, src_h, dst_w, dst_h, matrix):
        self.src_w, self.src_h = int(src_w), int(src_h)
        self.dst_w, self.dst_h = int(dst_w), int(dst_h)
        self.matrix = np.asarray(matrix, dtype=float)

    # ------------------------------------------------------- annotations
    def raw_points(self, pts):
        """Source coordinates -> destination coordinates, not clipped to the picture."""
        if not len(pts):
            return []
        arr = np.array([[x, y, 1.0] for x, y in pts], dtype=float)
        out = arr @ self.matrix.T
        return list(zip(out[:, 0].tolist(), out[:, 1].tolist()))

    def points(self, pts):
        """[(x, y), ...] in source coordinates -> destination coordinates (kept inside the picture)."""
        moved = self.raw_points(pts)
        return [(min(max(x, 0.0), self.dst_w), min(max(y, 0.0), self.dst_h)) for x, y in moved]

    def box(self, x1, y1, x2, y2):
        """Upright box in source coordinates -> tightest upright box around its image."""
        moved = self.points([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        xs = [p[0] for p in moved]
        ys = [p[1] for p in moved]
        return min(xs), min(ys), max(xs), max(ys)

    def keeps_box(self, x1, y1, x2, y2, min_visible=MIN_VISIBLE):
        """False when (almost) none of the box is left inside the destination picture."""
        moved = self.raw_points([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        xs = [p[0] for p in moved]
        ys = [p[1] for p in moved]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        w = min(max(xs), self.dst_w) - max(min(xs), 0.0)
        h = min(max(ys), self.dst_h) - max(min(ys), 0.0)
        if w < 1.0 or h < 1.0:
            return False
        return area <= 0 or (w * h) / area >= min_visible

    def then(self, other):
        """This transform followed by `other` (which starts from this one's output)."""
        return AffineTransform(self.src_w, self.src_h, other.dst_w, other.dst_h, other.matrix @ self.matrix)


def compose(*transforms):
    """The transforms one after the other; None entries are skipped, None when nothing is left."""
    chain = [t for t in transforms if t is not None]
    if not chain:
        return None
    out = chain[0]
    for t in chain[1:]:
        out = out.then(t)
    return out


class GeoTransform(AffineTransform):
    """flip (about the centre), then shear, then rotation (counter-clockwise on screen, degrees)."""

    def __init__(self, src_w, src_h, flip_h=False, flip_v=False, angle=0.0, shear_x=0.0, shear_y=0.0):
        self.flip_h, self.flip_v, self.angle = bool(flip_h), bool(flip_v), float(angle)
        self.shear_x, self.shear_y = float(shear_x), float(shear_y)
        src_w, src_h = int(src_w), int(src_h)
        cx, cy = src_w / 2.0, src_h / 2.0

        flip = np.eye(3)
        if self.flip_h:
            flip[0] = [-1.0, 0.0, src_w]
        if self.flip_v:
            flip[1] = [0.0, -1.0, src_h]

        tx, ty = math.tan(math.radians(self.shear_x)), math.tan(math.radians(self.shear_y))
        shear = np.array([[1.0, tx, -tx * cy],
                          [ty, 1.0, -ty * cx],
                          [0.0, 0.0, 1.0]])

        a = math.radians(self.angle)
        c, s = math.cos(a), math.sin(a)
        # counter-clockwise on screen (y points down)
        rot = np.array([[c, s, cx - c * cx - s * cy],
                        [-s, c, cy + s * cx - c * cy],
                        [0.0, 0.0, 1.0]])

        # The picture is symmetric about its centre, which every step keeps in place, so its new width / height
        # are the extents of the linear part. 1e-6: cos(90deg) is 6e-17, which must not cost an extra pixel row.
        lin = (rot @ shear @ flip)[:2, :2]
        dst_w = max(1, math.ceil(abs(lin[0, 0]) * src_w + abs(lin[0, 1]) * src_h - 1e-6))
        dst_h = max(1, math.ceil(abs(lin[1, 0]) * src_w + abs(lin[1, 1]) * src_h - 1e-6))
        move = np.eye(3)
        move[0, 2] = dst_w / 2.0 - cx
        move[1, 2] = dst_h / 2.0 - cy
        super().__init__(src_w, src_h, dst_w, dst_h, move @ rot @ shear @ flip)

    @property
    def is_rotated(self):
        """True when the pixels need a real warp (rotation or shear), not just a mirror."""
        return abs(self.angle) > 1e-9 or abs(self.shear_x) > 1e-9 or abs(self.shear_y) > 1e-9

    def warp(self, img):
        if not self.is_rotated:
            if self.flip_h and self.flip_v:
                return cv2.flip(img, -1)
            if self.flip_h:
                return cv2.flip(img, 1)
            if self.flip_v:
                return cv2.flip(img, 0)
            return img
        # cv2 puts pixel centres on integers, annotations put pixel edges there
        half_in, half_out = np.eye(3), np.eye(3)
        half_in[0, 2] = half_in[1, 2] = 0.5
        half_out[0, 2] = half_out[1, 2] = -0.5
        m = half_out @ self.matrix @ half_in
        return cv2.warpAffine(img, m[:2], (self.dst_w, self.dst_h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


class CropTransform(AffineTransform):
    """The window (x0, y0, w, h) of the picture, scaled back up to the size of the picture."""

    def __init__(self, src_w, src_h, x0, y0, w, h):
        self.window = (int(x0), int(y0), max(1, int(w)), max(1, int(h)))
        x0, y0, w, h = self.window
        sx, sy = src_w / w, src_h / h
        super().__init__(src_w, src_h, src_w, src_h, [[sx, 0, -x0 * sx], [0, sy, -y0 * sy], [0, 0, 1]])

    def warp(self, img):
        x0, y0, w, h = self.window
        return cv2.resize(img[y0:y0 + h, x0:x0 + w], (self.dst_w, self.dst_h), interpolation=cv2.INTER_LINEAR)


RESIZE_MODES = {
    "stretch": "Stretch to",
    "fill": "Fill (center crop) to",
    "fit_black": "Fit within (black edges) in",
    "fit_white": "Fit within (white edges) in",
    "fit_reflect": "Fit within (reflect edges) in",
}


class ResizeTransform(AffineTransform):
    def __init__(self, src_w, src_h, out_w, out_h, mode="stretch"):
        if mode not in RESIZE_MODES:
            raise ValueError(f"unknown resize mode: {mode!r}")
        self.mode, src_w, src_h = mode, int(src_w), int(src_h)
        out_w, out_h = max(1, int(out_w)), max(1, int(out_h))
        if mode == "stretch":
            nw, nh, left, top = out_w, out_h, 0, 0
        elif mode == "fill":
            s = max(out_w / src_w, out_h / src_h)
            nw, nh = max(out_w, round(src_w * s)), max(out_h, round(src_h * s))
            left, top = -((nw - out_w) // 2), -((nh - out_h) // 2)          # negative: the part that is cut off
        else:
            s = min(out_w / src_w, out_h / src_h)
            nw, nh = max(1, min(out_w, round(src_w * s))), max(1, min(out_h, round(src_h * s)))
            left, top = (out_w - nw) // 2, (out_h - nh) // 2
        self.scaled, self.offset = (nw, nh), (left, top)
        sx, sy = nw / src_w, nh / src_h
        super().__init__(src_w, src_h, out_w, out_h, [[sx, 0, left], [0, sy, top], [0, 0, 1]])

    def warp(self, img):
        nw, nh = self.scaled
        left, top = self.offset
        shrinking = nw < img.shape[1] or nh < img.shape[0]
        out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)
        if self.mode == "stretch":
            return out
        if self.mode == "fill":
            return out[-top:-top + self.dst_h, -left:-left + self.dst_w]
        pad = (top, self.dst_h - nh - top, left, self.dst_w - nw - left)
        if self.mode == "fit_reflect":
            return cv2.copyMakeBorder(out, *pad, cv2.BORDER_REFLECT_101 if min(nw, nh) > max(pad) else cv2.BORDER_REPLICATE)
        value = (255, 255, 255) if self.mode == "fit_white" else (0, 0, 0)
        return cv2.copyMakeBorder(out, *pad, cv2.BORDER_CONSTANT, value=value)


# ----------------------------------------------------------------------
#  YOLO
# ----------------------------------------------------------------------
def _unit(v):
    return max(0.0, min(1.0, v))


def yolo_label_text(text, tf):
    """Transform the contents of a YOLO label file (detection and/or segmentation lines)."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, vals = parts[0], [float(v) for v in parts[1:]]
        if len(vals) == 4:                                    # class cx cy w h
            cx, cy, w, h = vals
            x1, y1 = (cx - w / 2) * tf.src_w, (cy - h / 2) * tf.src_h
            x2, y2 = (cx + w / 2) * tf.src_w, (cy + h / 2) * tf.src_h
            if not tf.keeps_box(x1, y1, x2, y2):
                continue
            nx1, ny1, nx2, ny2 = tf.box(x1, y1, x2, y2)
            out.append(f"{cls} {_unit((nx1 + nx2) / 2 / tf.dst_w):.6f} "
                       f"{_unit((ny1 + ny2) / 2 / tf.dst_h):.6f} "
                       f"{_unit((nx2 - nx1) / tf.dst_w):.6f} {_unit((ny2 - ny1) / tf.dst_h):.6f}")
        else:                                                 # class x1 y1 x2 y2 ...
            pts = [(vals[i] * tf.src_w, vals[i + 1] * tf.src_h) for i in range(0, len(vals) - 1, 2)]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            if not tf.keeps_box(min(xs), min(ys), max(xs), max(ys)):
                continue
            moved = tf.points(pts)
            out.append(cls + " " + " ".join(
                f"{_unit(x / tf.dst_w):.6f} {_unit(y / tf.dst_h):.6f}" for x, y in moved))
    return "\n".join(out)


# ----------------------------------------------------------------------
#  Pascal VOC
# ----------------------------------------------------------------------
def _rounded(v):
    return str(max(0, int(round(v))))


def voc_xml_write(src_xml, dest_xml, new_image_name, tf=None):
    """Copy a VOC annotation for another image file: new file name, and (when the
    geometry changed) new image size and transformed boxes/polygons. Objects that
    the transform pushed out of the picture are left out."""
    tree = ET.parse(src_xml)
    root = tree.getroot()
    for tag in ("filename", "path"):
        elem = root.find(tag)
        if elem is not None:
            elem.text = new_image_name

    if tf is not None:
        size = root.find("size")
        if size is not None:
            if size.find("width") is not None:
                size.find("width").text = str(tf.dst_w)
            if size.find("height") is not None:
                size.find("height").text = str(tf.dst_h)

        for obj in root.findall("object"):
            poly = obj.find("polygon")
            if poly is not None and _voc_polygon_points(poly):
                pts = _voc_polygon_points(poly)
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                if not tf.keeps_box(min(xs), min(ys), max(xs), max(ys)):
                    root.remove(obj)
                    continue
                _move_voc_polygon(poly, tf)
                continue
            bnd = obj.find("bndbox")
            if bnd is not None:
                x1, y1, x2, y2 = (float(bnd.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
                if not tf.keeps_box(x1, y1, x2, y2):
                    root.remove(obj)
                    continue
                nx1, ny1, nx2, ny2 = tf.box(x1, y1, x2, y2)
                for key, val in (("xmin", nx1), ("ymin", ny1), ("xmax", nx2), ("ymax", ny2)):
                    bnd.find(key).text = _rounded(val)
    tree.write(dest_xml, encoding="utf-8")


def _voc_polygon_points(poly):
    point_elems = poly.findall("point")
    if point_elems:
        return [(float(p.findtext("x")), float(p.findtext("y"))) for p in point_elems]
    pts, i = [], 1
    while poly.find(f"x{i}") is not None and poly.find(f"y{i}") is not None:
        pts.append((float(poly.find(f"x{i}").text), float(poly.find(f"y{i}").text)))
        i += 1
    return pts


def _move_voc_polygon(poly, tf):
    """Rewrite a polygon in place (either <point><x/><y/></point> or x1/y1/x2/y2... style)."""
    point_elems = poly.findall("point")
    if point_elems:
        moved = tf.points([(float(p.findtext("x")), float(p.findtext("y"))) for p in point_elems])
        for elem, (x, y) in zip(point_elems, moved):
            elem.find("x").text = _rounded(x)
            elem.find("y").text = _rounded(y)
        return True
    count = len(_voc_polygon_points(poly))
    if count == 0:
        return False
    moved = tf.points(_voc_polygon_points(poly))
    for i, (x, y) in enumerate(moved, start=1):
        poly.find(f"x{i}").text = _rounded(x)
        poly.find(f"y{i}").text = _rounded(y)
    return True


# ----------------------------------------------------------------------
#  COCO
# ----------------------------------------------------------------------
def _shoelace(points):
    n = len(points)
    return 0.5 * abs(sum(points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]
                         for i in range(n)))


def coco_annotation(ann, tf):
    """Transform one COCO annotation dict (bbox, area, segmentation) into the new image; None when the
    transform pushed the object out of the picture."""
    new = copy.deepcopy(ann)
    rings = ann.get("segmentation") or []
    if rings:
        flat = [(ring[i], ring[i + 1]) for ring in rings for i in range(0, len(ring) - 1, 2)]
        xs, ys = [p[0] for p in flat], [p[1] for p in flat]
        if not tf.keeps_box(min(xs), min(ys), max(xs), max(ys)):
            return None
        moved_rings = []
        for ring in rings:
            pts = [(ring[i], ring[i + 1]) for i in range(0, len(ring) - 1, 2)]
            moved_rings.append(tf.points(pts))
        new["segmentation"] = [[c for p in ring for c in p] for ring in moved_rings]
        xs = [p[0] for ring in moved_rings for p in ring]
        ys = [p[1] for ring in moved_rings for p in ring]
        new["bbox"] = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
        new["area"] = sum(_shoelace(r) for r in moved_rings)
    else:
        x, y, w, h = ann["bbox"]
        if not tf.keeps_box(x, y, x + w, y + h):
            return None
        nx1, ny1, nx2, ny2 = tf.box(x, y, x + w, y + h)
        new["bbox"] = [nx1, ny1, nx2 - nx1, ny2 - ny1]
        new["area"] = (nx2 - nx1) * (ny2 - ny1)
    return new

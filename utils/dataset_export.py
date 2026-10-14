"""Export Dataset, without any GUI: pick the images, split them, preprocess, augment (Train only) and write
YOLO, Pascal VOC or COCO files.

  records      [{'source', 'filename', 'base', 'folder'}]  every image of the workspace (datasetsInput/<ws>-<n>)
  by_split     {'train': [record, ...], 'valid': [...], 'test': [...]}
  export_dataset(...) writes the three splits; Train images also get the augmented copies of the Augmenter.

The splits come from the splits assigned in Jelibox (`utils/splits.py`) or from a random split by percentage.
"""
import os
import random
import shutil
import xml.etree.ElementTree as ET
import json

from . import splits as image_splits
from .augment_geometry import coco_annotation, compose, voc_xml_write, yolo_label_text
from .augmentation import read_image, write_image
from .export_preprocess import prepare, oriented_size, Preprocess

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
FORMATS = ("YOLO", "Pascal VOC (XML)", "COCO")
SPLIT_ORDER = (image_splits.TRAIN, image_splits.VALID, image_splits.TEST)


# ----------------------------------------------------------------------
#  Which images, and which split
# ----------------------------------------------------------------------
def find_images(datasets_root, workspace, exts=IMAGE_EXTS):
    """Every image of the workspace: datasetsInput/<workspace>-<n>/ for every n."""
    records = []
    if not os.path.isdir(datasets_root):
        return records
    prefix = workspace + '-'
    for folder in sorted(f for f in os.listdir(datasets_root)
                         if f.startswith(prefix) and os.path.isdir(os.path.join(datasets_root, f))):
        folder_path = os.path.join(datasets_root, folder)
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(exts):
                records.append({'source': os.path.join(folder_path, filename), 'filename': filename,
                                'base': os.path.splitext(filename)[0], 'folder': folder})
    return records


def exportable(fmt, records, yolo_labels_folder):
    """The images this format can export: YOLO needs a label file, VOC and COCO take every image."""
    if fmt != "YOLO":
        return list(records)
    return [r for r in records if os.path.isfile(os.path.join(yolo_labels_folder, r['base'] + '.txt'))]


def random_counts(n, train_pct, valid_pct):
    """(train, valid, test) image counts of a percentage split of n images."""
    n_train = int(round(n * (train_pct / 100.0)))
    n_valid = int(round(n * (valid_pct / 100.0)))
    if n_train + n_valid > n:
        n_valid = max(0, n - n_train)
    return n_train, n_valid, n - n_train - n_valid


def assigned_counts(records, store):
    counts = store.counts([r['filename'] for r in records])
    return counts[image_splits.TRAIN], counts[image_splits.VALID], counts[image_splits.TEST]


def split_records(records, mode, store=None, train_pct=70, valid_pct=20, seed=None):
    """{'train': [...], 'valid': [...], 'test': [...]}. mode 'assigned' uses the splits set in Jelibox,
    'random' shuffles and cuts by percentage (the same seed gives the same split)."""
    if mode == 'assigned':
        out = {s: [] for s in SPLIT_ORDER}
        for r in records:
            out[store.get(r['filename'])].append(r)
        return out
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_train, n_valid, _ = random_counts(len(shuffled), train_pct, valid_pct)
    return {image_splits.TRAIN: shuffled[:n_train], image_splits.VALID: shuffled[n_train:n_train + n_valid],
            image_splits.TEST: shuffled[n_train + n_valid:]}


# ----------------------------------------------------------------------
#  Format writers: the folder layout and the labels of one image (or augmented copy)
# ----------------------------------------------------------------------
class _YoloWriter:
    names = {'train': 'train', 'valid': 'val', 'test': 'test'}

    def __init__(self, target, classes, labels_folder, voc_folder):
        self.target, self.classes, self.labels = target, classes, labels_folder
        for s in self.names.values():
            os.makedirs(os.path.join(target, s, 'images'), exist_ok=True)
            os.makedirs(os.path.join(target, s, 'labels'), exist_ok=True)

    def image_dir(self, split):
        return os.path.join(self.target, self.names[split], 'images')

    def add(self, split, rec, stem, ext, size, src_size, tf):
        src = os.path.join(self.labels, rec['base'] + '.txt')
        dst = os.path.join(self.target, self.names[split], 'labels', stem + '.txt')
        if tf is None:
            shutil.copy2(src, dst)
            return
        with open(src, encoding='utf-8') as f:
            text = yolo_label_text(f.read(), tf)
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(text)

    def finish(self):
        with open(os.path.join(self.target, 'data.yml'), 'w') as f:
            for key, s in (('train', 'train'), ('val', 'val'), ('test', 'test')):
                f.write(f"{key + ':':6s}{os.path.abspath(os.path.join(self.target, s, 'images'))}\n")
            f.write(f"nc:    {len(self.classes)}\nnames:\n")
            for i, name in enumerate(self.classes):
                f.write(f"  {i}: {name}\n")


class _VocWriter:
    names = {'train': 'train', 'valid': 'valid', 'test': 'test'}

    def __init__(self, target, classes, labels_folder, voc_folder):
        self.target, self.voc = target, voc_folder
        for s in self.names.values():
            os.makedirs(os.path.join(target, s), exist_ok=True)

    def image_dir(self, split):
        return os.path.join(self.target, self.names[split])

    def add(self, split, rec, stem, ext, size, src_size, tf):
        src = os.path.join(self.voc, rec['base'] + '.xml')
        if not os.path.isfile(src):
            return
        dst = os.path.join(self.image_dir(split), stem + '.xml')
        if tf is None and stem == rec['base']:
            shutil.copy2(src, dst)
        else:
            voc_xml_write(src, dst, stem + ext, tf)

    def finish(self):
        pass


class _CocoWriter:
    names = {'train': 'train', 'valid': 'val', 'test': 'test'}

    def __init__(self, target, classes, labels_folder, voc_folder):
        self.target, self.voc = target, voc_folder
        self.categories = {name: i + 1 for i, name in enumerate(classes)}
        self.docs = {}
        for split, s in self.names.items():
            os.makedirs(os.path.join(target, 'images', s), exist_ok=True)
            self.docs[split] = {
                'info': {'description': 'Jelibox dataset'}, 'licenses': [], 'images': [], 'annotations': [],
                'categories': [{'id': i, 'name': n, 'supercategory': 'object'} for n, i in self.categories.items()]}
        self._cache = {}

    def image_dir(self, split):
        return os.path.join(self.target, 'images', self.names[split])

    def _annotations(self, rec, src_size):
        key = rec['base']
        if key not in self._cache:
            self._cache = {key: read_coco_annotations(os.path.join(self.voc, rec['base'] + '.xml'),
                                                      src_size[0], src_size[1], self.categories)}
        return self._cache[key]

    def add(self, split, rec, stem, ext, size, src_size, tf):
        doc = self.docs[split]
        image_id = len(doc['images']) + 1
        doc['images'].append({'id': image_id, 'file_name': stem + ext, 'width': size[0], 'height': size[1]})
        for ann in self._annotations(rec, src_size):
            moved = ann if tf is None else coco_annotation(ann, tf)
            if moved is None:
                continue
            moved = dict(moved)
            moved['id'] = len(doc['annotations']) + 1
            moved['image_id'] = image_id
            doc['annotations'].append(moved)

    def finish(self):
        out = os.path.join(self.target, 'annotations')
        os.makedirs(out, exist_ok=True)
        for split, s in self.names.items():
            with open(os.path.join(out, f'instances_{s}.json'), 'w', encoding='utf-8') as f:
                json.dump(self.docs[split], f, indent=2)


_WRITERS = {"YOLO": _YoloWriter, "Pascal VOC (XML)": _VocWriter, "COCO": _CocoWriter}


def read_coco_annotations(xml_path, image_width, image_height, categories):
    """Jelibox VOC objects as COCO annotation dictionaries (without id / image_id)."""
    annotations = []
    if not os.path.exists(xml_path):
        return annotations
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        print(f"[COCO] Could not read {xml_path}: {exc}")
        return annotations

    for obj in root.findall('object'):
        name_elem = obj.find('name')
        class_name = (name_elem.text or '').strip() if name_elem is not None else ''
        if not class_name or class_name not in categories:
            continue

        polygon = obj.find('polygon')
        segmentation, points = [], []
        if polygon is not None:
            point_elements = polygon.findall('point')
            if point_elements:
                for point in point_elements:
                    x_elem, y_elem = point.find('x'), point.find('y')
                    if x_elem is not None and y_elem is not None:
                        points.append((float(x_elem.text), float(y_elem.text)))
            else:
                index = 1
                while polygon.find(f'x{index}') is not None and polygon.find(f'y{index}') is not None:
                    points.append((float(polygon.find(f'x{index}').text), float(polygon.find(f'y{index}').text)))
                    index += 1

        if len(points) >= 3:
            points = [(max(0.0, min(float(image_width), x)), max(0.0, min(float(image_height), y))) for x, y in points]
            segmentation = [[coord for point in points for coord in point]]
            xmin, ymin = min(p[0] for p in points), min(p[1] for p in points)
            xmax, ymax = max(p[0] for p in points), max(p[1] for p in points)
            area = 0.5 * abs(sum(points[i][0] * points[(i + 1) % len(points)][1] -
                                 points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points))))
        else:
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
            xmin, ymin = float(bbox.findtext('xmin', '0')), float(bbox.findtext('ymin', '0'))
            xmax, ymax = float(bbox.findtext('xmax', '0')), float(bbox.findtext('ymax', '0'))
            area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)

        xmin = max(0.0, min(float(image_width), xmin))
        ymin = max(0.0, min(float(image_height), ymin))
        xmax = max(xmin, min(float(image_width), xmax))
        ymax = max(ymin, min(float(image_height), ymax))
        annotations.append({'category_id': categories[class_name], 'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                            'area': area, 'segmentation': segmentation, 'iscrowd': 0})
    return annotations


# ----------------------------------------------------------------------
#  The export
# ----------------------------------------------------------------------
def export_dataset(fmt, target, by_split, classes, voc_folder, yolo_labels_folder, preprocess=None, augmenter=None,
                   progress_cb=None):
    """Write the dataset. Returns {'counts': {split: images written}, 'augmented': n, 'skipped': n}.
    Augmented copies are made for the Train split only."""
    if fmt not in _WRITERS:
        raise ValueError(f'Unknown format: {fmt}')
    preprocess = preprocess or Preprocess(auto_orient=False)
    total = sum(len(v) for v in by_split.values())
    if total == 0:
        raise ValueError('There are no images to export.')
    writer = _WRITERS[fmt](target, classes, yolo_labels_folder, voc_folder)
    counts, done = {s: 0 for s in SPLIT_ORDER}, 0

    for split in SPLIT_ORDER:
        img_dir, taken = writer.image_dir(split), set()
        for rec in by_split.get(split, []):
            stem, ext = rec['base'], os.path.splitext(rec['filename'])[1]
            if stem in taken:                                    # the same name in two instances of the workspace
                stem = f"{rec['folder']}_{stem}"
            taken.add(stem)
            try:
                src_size = oriented_size(rec['source'])
                image, t0 = prepare(rec['source'], preprocess)
                copies = []
                if augmenter is not None and split == image_splits.TRAIN:
                    base = image if image is not None else read_image(rec['source'])
                    copies = list(augmenter.copies(base))
                # Roboflow-style "Nx": the augmented outputs stand in for the image, unless the original is kept
                # (it is also written when no output could be made, so no source image drops out)
                if not copies or augmenter.config.keep_original:
                    if image is None:
                        shutil.copy2(rec['source'], os.path.join(img_dir, stem + ext))
                        size = src_size
                    else:
                        write_image(os.path.join(img_dir, stem + ext), image)
                        size = (image.shape[1], image.shape[0])
                    writer.add(split, rec, stem, ext, size, src_size, t0)
                counts[split] += 1
                for suffix, out, t_aug in copies:
                    write_image(os.path.join(img_dir, stem + suffix + ext), out)
                    writer.add(split, rec, stem + suffix, ext, (out.shape[1], out.shape[0]), src_size,
                               compose(t0, t_aug))
            except Exception as exc:
                print(f"[Export] Skipped {rec['source']}: {exc}")
            done += 1
            if progress_cb:
                progress_cb(done, total, f'Exporting {fmt}... {done}/{total}')
    writer.finish()
    return {'counts': counts, 'augmented': augmenter.created if augmenter else 0,
            'skipped': augmenter.skipped if augmenter else 0}

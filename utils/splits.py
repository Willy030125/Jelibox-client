"""
Train / valid / test split of every image of a workspace, and the filters built on it (no tkinter here).

The split is kept in `vocdataset/<workspace>/splits.json` as {image file name: split}. An image that is not in
the file is a Train image, so a new workspace (or images imported later) starts as all Train without writing
anything. Only valid / test are stored.
"""
import json
import os

TRAIN, VALID, TEST = "train", "valid", "test"
SPLITS = (TRAIN, VALID, TEST)
LABELS = {TRAIN: "Train", VALID: "Valid", TEST: "Test"}
FILE_NAME = "splits.json"

# folder names an exported dataset keeps its splits in (Roboflow: train / valid / test, Ultralytics: train / val)
_FOLDER_SPLITS = {"train": TRAIN, "training": TRAIN, "val": VALID, "valid": VALID, "validation": VALID,
                  "test": TEST, "testing": TEST}


def split_from_path(path, root):
    """The split an image sits in, from the folders between `root` and the file (the innermost one that names a
    split wins); None when none of them does."""
    parts = os.path.relpath(os.path.dirname(path), root).replace("\\", "/").split("/")
    for part in reversed(parts):
        if part.lower() in _FOLDER_SPLITS:
            return _FOLDER_SPLITS[part.lower()]
    return None


class SplitStore:
    def __init__(self, folder):
        self.path = os.path.join(folder, FILE_NAME)
        self.data = {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self.data = {str(k): v for k, v in raw.items() if v in (VALID, TEST)}
        except (OSError, ValueError):
            pass

    def get(self, name):
        return self.data.get(name, TRAIN)

    def set(self, name, split):
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split!r}")
        if split == TRAIN:
            self.data.pop(name, None)
        else:
            self.data[name] = split
        self._save()

    def set_many(self, assignments):
        """{image name: split}, one write."""
        for name, split in assignments.items():
            if split not in SPLITS:
                raise ValueError(f"unknown split: {split!r}")
            if split == TRAIN:
                self.data.pop(name, None)
            else:
                self.data[name] = split
        self._save()

    def forget(self, name):
        """The image was deleted."""
        if self.data.pop(name, None) is not None:
            self._save()

    def counts(self, names):
        out = {s: 0 for s in SPLITS}
        for n in names:
            out[self.get(n)] += 1
        return out

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(self.data.items())), f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)


class ClassIndex:
    """Which class ids each image has, read from its YOLO label file and cached until the file changes."""

    def __init__(self, labels_folder):
        self.folder = labels_folder
        self._cache = {}

    def ids(self, image_name):
        path = os.path.join(self.folder, os.path.splitext(image_name)[0] + ".txt")
        try:
            stamp = os.stat(path)
        except OSError:
            return frozenset()
        key = (stamp.st_mtime_ns, stamp.st_size)
        hit = self._cache.get(path)
        if hit and hit[0] == key:
            return hit[1]
        found = set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    head = line.split(None, 1)
                    if head and head[0].isdigit():
                        found.add(int(head[0]))
        except OSError:
            pass
        found = frozenset(found)
        self._cache[path] = (key, found)
        return found


def matching(images, store, class_index, classes, split=None, class_names=()):
    """The images (in their own order) that are in `split` (None = any) and hold at least one of `class_names`
    (empty = any). `classes` is the workspace class list, whose positions are the ids in the label files."""
    wanted = {classes.index(c) for c in class_names if c in classes}
    if class_names and not wanted:
        return []
    out = []
    for name in images:
        if split and store.get(name) != split:
            continue
        if wanted and not (class_index.ids(name) & wanted):
            continue
        out.append(name)
    return out

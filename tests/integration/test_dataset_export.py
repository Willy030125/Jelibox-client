"""Generate (Export Dataset) with preprocessing and augmentation, without any window."""
import json
import os
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image

from tests.helpers import isolated_workspace

isolated_workspace()

from utils import augmentation as aug                                   # noqa: E402
from utils import dataset_export as de                                  # noqa: E402
from utils import splits                                                # noqa: E402
from utils.augmentation import AugmentConfig, Augmenter                 # noqa: E402
from utils.export_preprocess import Preprocess, ResizeConfig, exif_orientation  # noqa: E402

W, H = 160, 100
BOX = (40, 20, 100, 70)                         # x1 y1 x2 y2 of the white rectangle in every picture
CLASSES = ["cat", "dog"]


class Workspace:
    """A temp workspace: 10 images (a white box on grey), YOLO labels and VOC xml for each, 6 train / 2 valid / 2 test."""

    def __init__(self, count=10):
        self.root = tempfile.mkdtemp(prefix="jelibox_gen_")
        self.images = os.path.join(self.root, "datasetsInput", "ws-1")
        self.labels = os.path.join(self.root, "labels")
        self.voc = os.path.join(self.root, "voc")
        for d in (self.images, self.labels, self.voc):
            os.makedirs(d)
        cx, cy = (BOX[0] + BOX[2]) / 2 / W, (BOX[1] + BOX[3]) / 2 / H
        bw, bh = (BOX[2] - BOX[0]) / W, (BOX[3] - BOX[1]) / H
        for i in range(count):
            img = np.full((H, W, 3), 90, np.uint8)
            img[BOX[1]:BOX[3], BOX[0]:BOX[2]] = 255
            img[0:8, 0:8] = (20 + i * 10, 60, 200)                  # a marker that tells the images apart
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(os.path.join(self.images, f"p{i}.png"))
            with open(os.path.join(self.labels, f"p{i}.txt"), "w") as f:
                f.write(f"{i % 2} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            with open(os.path.join(self.voc, f"p{i}.xml"), "w") as f:
                f.write(f"<annotation><filename>p{i}.png</filename><path>p{i}.png</path>"
                        f"<size><width>{W}</width><height>{H}</height><depth>3</depth></size>"
                        f"<object><name>{CLASSES[i % 2]}</name><bndbox><xmin>{BOX[0]}</xmin><ymin>{BOX[1]}</ymin>"
                        f"<xmax>{BOX[2]}</xmax><ymax>{BOX[3]}</ymax></bndbox></object></annotation>")
        self.records = de.find_images(os.path.join(self.root, "datasetsInput"), "ws")
        self.store = splits.SplitStore(os.path.join(self.root, "store"))
        self.store.set_many({"p6.png": "valid", "p7.png": "valid", "p8.png": "test", "p9.png": "test"})

    def generate(self, fmt="YOLO", mode="assigned", pre=None, ops=None, copies=2, seed=5, keep=False, **split):
        target = tempfile.mkdtemp(prefix="jelibox_out_", dir=self.root)
        by_split = de.split_records(de.exportable(fmt, self.records, self.labels), mode, self.store, seed=seed, **split)
        pre = pre or Preprocess(auto_orient=False)
        augmenter = None
        if ops:
            augmenter = Augmenter(AugmentConfig(ops={k: aug.clean_params(k, v) for k, v in ops.items()},
                                                copies=copies, keep_original=keep, seed=seed, resize=pre.resize))
        result = de.export_dataset(fmt, target, by_split, CLASSES, self.voc, self.labels, pre, augmenter)
        return target, result

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def names(folder):
    return sorted(os.listdir(folder))


class GenerateAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.addCleanup(self.ws.close)

    def test_yolo_generate_augments_only_the_train_split(self):
        ops = {"brightness": {"percent": 30}, "flip": {"horizontal": True, "vertical": False}}
        target, result = self.ws.generate("YOLO", ops=ops, copies=2)
        train = names(os.path.join(target, "train", "images"))
        self.assertEqual(len(train), 6 * 2)                                   # "2x": two outputs per Train image
        self.assertTrue(all("_aug" in n for n in train))                      # the originals are not in the Train set
        self.assertEqual(names(os.path.join(target, "val", "images")), ["p6.png", "p7.png"])
        self.assertEqual(names(os.path.join(target, "test", "images")), ["p8.png", "p9.png"])
        for split in ("train", "val", "test"):
            imgs = [os.path.splitext(n)[0] for n in names(os.path.join(target, split, "images"))]
            lbls = [os.path.splitext(n)[0] for n in names(os.path.join(target, split, "labels"))]
            self.assertEqual(imgs, lbls, split)
        self.assertEqual(result["augmented"], 12)
        self.assertEqual(result["counts"], {"train": 6, "valid": 2, "test": 2})
        with open(os.path.join(target, "data.yml")) as f:
            text = f.read()
        self.assertIn("nc:    2", text)
        self.assertIn("0: cat", text)

    def test_generated_copies_really_differ_and_keep_the_labelled_object_on_the_box(self):
        ops = {"brightness": {"percent": 30}, "rotation": {"degrees": 20}, "shear": {"horizontal": 10, "vertical": 10}}
        target, _ = self.ws.generate("YOLO", ops=ops, copies=3, keep=True)
        original = cv2.imread(os.path.join(target, "train", "images", "p0.png"))
        for i in (1, 2, 3):
            copy_ = cv2.imread(os.path.join(target, "train", "images", f"p0_aug{i}.png"))
            self.assertFalse(copy_.shape == original.shape and np.array_equal(copy_, original))
            with open(os.path.join(target, "train", "labels", f"p0_aug{i}.txt")) as f:
                cls, cx, cy, bw, bh = f.read().split()
            h, w = copy_.shape[:2]
            cx, cy = float(cx) * w, float(cy) * h
            # the label centre must sit on a bright pixel of the white box (brightness 30 % keeps it light)
            patch = copy_[int(cy) - 1:int(cy) + 2, int(cx) - 1:int(cx) + 2]
            self.assertGreater(patch.mean(), 150, f"copy {i}: label centre is not on the object")

    def test_every_augmentation_can_be_generated_at_once(self):
        ops = {key: aug.default_params(key) for key in aug.OPS}
        target, result = self.ws.generate("YOLO", ops=ops, copies=2)
        self.assertGreater(result["augmented"], 0)
        for n in names(os.path.join(target, "train", "labels")):
            with open(os.path.join(target, "train", "labels", n)) as f:
                for line in f.read().splitlines():
                    values = [float(v) for v in line.split()[1:]]
                    self.assertTrue(all(0.0 <= v <= 1.0 for v in values), (n, line))
        for n in names(os.path.join(target, "train", "images")):
            self.assertIsNotNone(cv2.imread(os.path.join(target, "train", "images", n)), n)

    def test_voc_generate_augments_only_train_and_updates_the_xml(self):
        ops = {"rotation": {"degrees": 30}}
        target, result = self.ws.generate("Pascal VOC (XML)", ops=ops, copies=2)
        self.assertEqual(sum("_aug" in n and n.endswith(".png") for n in names(os.path.join(target, "train"))), 12)
        self.assertNotIn("p0.png", names(os.path.join(target, "train")))
        self.assertEqual(names(os.path.join(target, "valid")), ["p6.png", "p6.xml", "p7.png", "p7.xml"])
        root = ET.parse(os.path.join(target, "train", "p0_aug1.xml")).getroot()
        img = cv2.imread(os.path.join(target, "train", "p0_aug1.png"))
        self.assertEqual(root.findtext("filename"), "p0_aug1.png")
        self.assertEqual((int(root.findtext("size/width")), int(root.findtext("size/height"))), (img.shape[1], img.shape[0]))
        self.assertTrue(0 < int(root.findtext("object/bndbox/xmax")) <= img.shape[1])

    def test_coco_generate_augments_only_train_and_ids_are_unique(self):
        ops = {"flip": {"horizontal": True, "vertical": True}, "brightness": {"percent": 20}}
        target, _ = self.ws.generate("COCO", ops=ops, copies=2)
        docs = {}
        for split in ("train", "val", "test"):
            with open(os.path.join(target, "annotations", f"instances_{split}.json"), encoding="utf-8") as f:
                docs[split] = json.load(f)
        self.assertEqual((len(docs["train"]["images"]), len(docs["val"]["images"]), len(docs["test"]["images"])), (12, 2, 2))
        for doc in docs.values():
            ids = [i["id"] for i in doc["images"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual({a["image_id"] for a in doc["annotations"]}, set(ids))
        by_id = {i["id"]: i for i in docs["train"]["images"]}
        for ann in docs["train"]["annotations"]:
            x, y, w, h = ann["bbox"]
            self.assertLessEqual(x + w, by_id[ann["image_id"]]["width"] + 1e-6)
        for entry in docs["train"]["images"]:
            self.assertTrue(os.path.exists(os.path.join(target, "images", "train", entry["file_name"])))

    def test_nx_means_n_outputs_per_train_image_and_keep_original_makes_one_of_them_the_original(self):
        ops = {"brightness": {"percent": 30}}
        for n in (2, 3, 4):
            target, _ = self.ws.generate("YOLO", ops=ops, copies=n)
            self.assertEqual(len(names(os.path.join(target, "train", "images"))), 6 * n, n)
        target, _ = self.ws.generate("YOLO", ops=ops, copies=2, keep=True)                # 3x = original + 2 outputs
        train = names(os.path.join(target, "train", "images"))
        self.assertEqual(len(train), 6 * 3)
        self.assertIn("p0.png", train)
        self.assertIn("p0_aug2.png", train)

    def test_an_image_whose_outputs_all_failed_is_written_as_it_is(self):
        real = Augmenter.copies
        Augmenter.copies = lambda self, img: iter(())
        try:
            target, _ = self.ws.generate("YOLO", ops={"brightness": {"percent": 30}}, copies=3)
        finally:
            Augmenter.copies = real
        self.assertEqual(names(os.path.join(target, "train", "images")), [f"p{i}.png" for i in range(6)])

    def test_without_augmentation_nothing_extra_is_written(self):
        target, result = self.ws.generate("YOLO")
        self.assertEqual(len(names(os.path.join(target, "train", "images"))), 6)
        self.assertEqual(result["augmented"], 0)

    def test_same_seed_generates_the_same_files(self):
        ops = {"brightness": {"percent": 30}, "noise": {"percent": 5}}
        a, _ = self.ws.generate("YOLO", ops=ops, seed=9)
        b, _ = self.ws.generate("YOLO", ops=ops, seed=9)
        for n in names(os.path.join(a, "train", "images")):
            self.assertTrue(np.array_equal(cv2.imread(os.path.join(a, "train", "images", n)),
                                           cv2.imread(os.path.join(b, "train", "images", n))), n)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.addCleanup(self.ws.close)

    def test_assigned_split_follows_the_buttons(self):
        by = de.split_records(self.ws.records, "assigned", self.ws.store)
        self.assertEqual({k: len(v) for k, v in by.items()}, {"train": 6, "valid": 2, "test": 2})
        self.assertEqual(de.assigned_counts(self.ws.records, self.ws.store), (6, 2, 2))

    def test_random_split_cuts_by_percentage_and_repeats_with_a_seed(self):
        self.assertEqual(de.random_counts(10, 70, 20), (7, 2, 1))
        self.assertEqual(de.random_counts(10, 100, 50), (10, 0, 0))
        a = de.split_records(self.ws.records, "random", train_pct=60, valid_pct=30, seed=4)
        b = de.split_records(self.ws.records, "random", train_pct=60, valid_pct=30, seed=4)
        self.assertEqual({k: len(v) for k, v in a.items()}, {"train": 6, "valid": 3, "test": 1})
        self.assertEqual({k: [r["filename"] for r in v] for k, v in a.items()},
                         {k: [r["filename"] for r in v] for k, v in b.items()})
        every = [r["filename"] for v in a.values() for r in v]
        self.assertEqual(sorted(every), sorted(r["filename"] for r in self.ws.records))

    def test_yolo_only_exports_labelled_images(self):
        os.remove(os.path.join(self.ws.labels, "p3.txt"))
        self.assertEqual(len(de.exportable("YOLO", self.ws.records, self.ws.labels)), 9)
        self.assertEqual(len(de.exportable("COCO", self.ws.records, self.ws.labels)), 10)

    def test_nothing_to_export_is_an_error(self):
        with self.assertRaises(ValueError):
            de.export_dataset("YOLO", tempfile.mkdtemp(dir=self.ws.root), {}, CLASSES, self.ws.voc, self.ws.labels)


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.addCleanup(self.ws.close)

    def test_stretch_resize_sets_the_size_and_keeps_normalised_labels(self):
        pre = Preprocess(False, ResizeConfig("stretch", 64, 64))
        target, _ = self.ws.generate("YOLO", pre=pre)
        for n in names(os.path.join(target, "train", "images")):
            self.assertEqual(cv2.imread(os.path.join(target, "train", "images", n)).shape[:2], (64, 64))
        with open(os.path.join(target, "train", "labels", "p0.txt")) as f:
            cls, cx, cy, bw, bh = (float(v) for v in f.read().split())
        self.assertAlmostEqual(cx, (BOX[0] + BOX[2]) / 2 / W, places=4)
        self.assertAlmostEqual(bw, (BOX[2] - BOX[0]) / W, places=4)

    def test_fit_resize_moves_the_labels_with_the_padding(self):
        pre = Preprocess(False, ResizeConfig("fit_black", 100, 100))
        target, _ = self.ws.generate("YOLO", pre=pre)
        img = cv2.imread(os.path.join(target, "train", "images", "p0.png"))
        self.assertEqual(img.shape[:2], (100, 100))
        with open(os.path.join(target, "train", "labels", "p0.txt")) as f:
            cls, cx, cy, bw, bh = (float(v) for v in f.read().split())
        # 160x100 fitted into 100x100: scale 0.625, 18 px of black above and below
        self.assertAlmostEqual(cy, (((BOX[1] + BOX[3]) / 2) * 0.625 + 18.5) / 100, delta=0.02)
        self.assertAlmostEqual(bh, (BOX[3] - BOX[1]) * 0.625 / 100, delta=0.02)
        ys, xs = np.where(img[..., 0] > 240)
        self.assertAlmostEqual((xs.min() + xs.max() + 1) / 2 / 100, cx, delta=0.03)
        self.assertAlmostEqual((ys.min() + ys.max() + 1) / 2 / 100, cy, delta=0.03)

    def test_resize_updates_voc_and_coco_pixel_coordinates(self):
        pre = Preprocess(False, ResizeConfig("stretch", 80, 50))               # exactly half
        target, _ = self.ws.generate("Pascal VOC (XML)", pre=pre)
        root = ET.parse(os.path.join(target, "train", "p0.xml")).getroot()
        self.assertEqual((root.findtext("size/width"), root.findtext("size/height")), ("80", "50"))
        self.assertEqual([root.findtext(f"object/bndbox/{k}") for k in ("xmin", "ymin", "xmax", "ymax")],
                         ["20", "10", "50", "35"])
        target, _ = self.ws.generate("COCO", pre=pre)
        with open(os.path.join(target, "annotations", "instances_train.json"), encoding="utf-8") as f:
            doc = json.load(f)
        entry = next(i for i in doc["images"] if i["file_name"] == "p0.png")
        self.assertEqual((entry["width"], entry["height"]), (80, 50))
        ann = next(a for a in doc["annotations"] if a["image_id"] == entry["id"])
        self.assertEqual([round(v) for v in ann["bbox"]], [20, 10, 30, 25])

    def test_augmented_copies_are_resized_too(self):
        pre = Preprocess(False, ResizeConfig("stretch", 64, 64))
        target, _ = self.ws.generate("YOLO", pre=pre, ops={"rotation": {"degrees": 40}}, copies=2)
        for n in names(os.path.join(target, "train", "images")):
            self.assertEqual(cv2.imread(os.path.join(target, "train", "images", n)).shape[:2], (64, 64), n)

    def test_auto_orient_turns_exif_rotated_pictures_and_drops_the_tag(self):
        path = os.path.join(self.ws.images, "p0.png")
        jpg = os.path.join(self.ws.images, "p0.jpg")
        os.remove(path)
        exif = Image.Exif()
        exif[0x0112] = 6                                                       # stored sideways, shown rotated
        Image.new("RGB", (100, 160), (10, 200, 10)).save(jpg, exif=exif)
        self.assertEqual(exif_orientation(jpg), 6)
        self.ws.records = de.find_images(os.path.join(self.ws.root, "datasetsInput"), "ws")
        target, _ = self.ws.generate("YOLO", pre=Preprocess(True, None))
        out = os.path.join(target, "train", "images", "p0.jpg")
        self.assertEqual(exif_orientation(out), 1)
        with Image.open(out) as im:
            self.assertEqual(im.size, (160, 100))                               # the pixels are turned, not just tagged
        target, _ = self.ws.generate("YOLO", pre=Preprocess(False, None))
        self.assertEqual(exif_orientation(os.path.join(target, "train", "images", "p0.jpg")), 6)   # copied as it is


if __name__ == "__main__":
    unittest.main()

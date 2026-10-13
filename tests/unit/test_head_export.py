"""Export of a workspace's labels for the command-line head trainer, and the settings of the detector-only Annotate all."""
import os
import tempfile
import unittest

from utils import head_data, head_export
from utils import workspace_config as wcfg


def write(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


class HeadExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.images = os.path.join(self.tmp.name, "images")
        self.labels = os.path.join(self.tmp.name, "labels")
        for i in range(10):
            write(os.path.join(self.images, f"f{i:02d}.jpg"), "x")
            write(os.path.join(self.labels, f"f{i:02d}.txt"),
                  "" if i == 9 else "1 0.500000 0.500000 0.200000 0.400000\n")
        # a segmentation row becomes its bounding box
        write(os.path.join(self.labels, "f00.txt"), "0 0.1 0.1 0.3 0.1 0.3 0.5 0.1 0.5\n")

    def export(self, kind, allow_empty, **kw):
        pairs = head_data.labelled_pairs([self.images], self.labels, allow_empty=allow_empty)
        out = os.path.join(self.tmp.name, "out")
        info = head_export.export_for_trainer(pairs, out, head_name="my head", head_type=kind,
                                              classes=["a", "b"], anchors=[0], weights="models/yolov8m.pt",
                                              taps=(15, 18, 21), imgsz=(384, 640), **kw)
        return out, info

    def test_classify_layout_split_and_box_rows(self):
        out, info = self.export("classify", False, val_fraction=0.2)
        self.assertEqual((info["train"], info["val"]), (7, 2))               # 9 images with boxes, last 2 validate
        self.assertTrue(os.path.isfile(os.path.join(out, "images", "val", "f08.jpg")))
        rows = open(os.path.join(out, "labels_my_head", "train", "f00.txt")).read().split()
        self.assertEqual(rows[0], "0")
        self.assertEqual(len(rows), 5)                                       # class + 4 numbers, not a polygon
        self.assertAlmostEqual(float(rows[3]), 0.2)                          # width
        self.assertFalse(os.path.exists(os.path.join(out, "images", "val", "f09.jpg")))   # empty label: not for classify

    def test_detect_keeps_empty_labels_as_negatives(self):
        out, info = self.export("detect", True, val_fraction=0.2)
        self.assertEqual(info["train"] + info["val"], 10)
        self.assertEqual(open(os.path.join(out, "labels_my_head", "val", "f09.txt")).read(), "")

    def test_config_text(self):
        out, info = self.export("detect", True, epochs=30, batch=8, lr=0.002, flip=False)
        text = open(info["config"]).read()
        for part in ('weights: "yolov8m.pt"', "imgsz: [384, 640]", "type: detect", "attach_to: [0]",
                     'labels_my_head/{split}', "flip: false", "epochs: 30"):
            self.assertIn(part, text)
        self.assertEqual(info["command"], "python train.py --config my_head.yaml --head my_head")


class DetectorVariantSettingsTests(unittest.TestCase):
    def test_defaults_and_cleaning(self):
        h = wcfg._normalize({})["label_assistant"]["custom_head"]
        self.assertEqual((h["annotate_with"], h["pre_detector"], h["pre_label"]), ("head", "yolov8m.pt", ""))
        self.assertEqual(len(wcfg.PRE_DETECTORS), 18)
        for name in ("yolov5mu.pt", "yolov5x6u.pt", "yolov8m.pt", "yolov9e.pt", "yolo11l.pt", "yolo26x.pt"):
            self.assertIn(name, wcfg.PRE_DETECTORS)
        self.assertIn(h["pre_detector"], wcfg.PRE_DETECTORS)
        good26 = wcfg._normalize({"label_assistant": {"custom_head": {"pre_detector": "yolo26x.pt"}}})
        self.assertEqual(good26["label_assistant"]["custom_head"]["pre_detector"], "yolo26x.pt")
        bad = wcfg._normalize({"label_assistant": {"custom_head": {"annotate_with": "x", "pre_detector": "nope.pt"}}})
        h = bad["label_assistant"]["custom_head"]
        self.assertEqual((h["annotate_with"], h["pre_detector"]), ("head", "yolov8m.pt"))
        good = wcfg._normalize({"label_assistant": {"custom_head": {
            "annotate_with": "detector", "pre_detector": "yolov9c.pt", "pre_label": "standing"}}})
        h = good["label_assistant"]["custom_head"]
        self.assertEqual((h["annotate_with"], h["pre_detector"], h["pre_label"]), ("detector", "yolov9c.pt", "standing"))


if __name__ == "__main__":
    unittest.main()

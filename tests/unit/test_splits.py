import json
import os
import tempfile
import unittest

from tests.helpers import isolated_workspace

isolated_workspace()

from utils import splits as sp                                   # noqa: E402


class SplitStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = os.path.join(self.tmp.name, "voc")

    def test_a_new_workspace_is_all_train_and_writes_nothing(self):
        store = sp.SplitStore(self.folder)
        self.assertEqual({store.get(n) for n in ("a.png", "b.png")}, {sp.TRAIN})
        self.assertFalse(os.path.exists(store.path))

    def test_valid_and_test_are_stored_and_train_removes_the_entry(self):
        store = sp.SplitStore(self.folder)
        store.set("a.png", sp.VALID)
        store.set("b.png", sp.TEST)
        again = sp.SplitStore(self.folder)
        self.assertEqual((again.get("a.png"), again.get("b.png"), again.get("c.png")), (sp.VALID, sp.TEST, sp.TRAIN))
        again.set("a.png", sp.TRAIN)
        with open(again.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"b.png": "test"})

    def test_unknown_split_is_refused_and_a_broken_file_reads_as_all_train(self):
        store = sp.SplitStore(self.folder)
        with self.assertRaises(ValueError):
            store.set("a.png", "dev")
        os.makedirs(self.folder, exist_ok=True)
        with open(store.path, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(sp.SplitStore(self.folder).get("a.png"), sp.TRAIN)

    def test_counts_and_forget(self):
        store = sp.SplitStore(self.folder)
        store.set("a.png", sp.VALID)
        store.set("b.png", sp.VALID)
        self.assertEqual(store.counts(["a.png", "b.png", "c.png"]), {sp.TRAIN: 1, sp.VALID: 2, sp.TEST: 0})
        store.forget("a.png")
        self.assertEqual(store.get("a.png"), sp.TRAIN)


class FolderSplitTests(unittest.TestCase):
    def test_split_from_path(self):
        root = os.path.join("x", "ds")
        at = lambda *parts: sp.split_from_path(os.path.join(root, *parts), root)
        self.assertEqual(at("train", "images", "a.jpg"), sp.TRAIN)
        self.assertEqual(at("valid", "images", "a.jpg"), sp.VALID)
        self.assertEqual(at("images", "VAL", "a.jpg"), sp.VALID)         # Ultralytics spelling, any case
        self.assertEqual(at("test", "a.jpg"), sp.TEST)
        self.assertIsNone(at("a.jpg"))
        self.assertIsNone(at("images", "a.jpg"))
        self.assertEqual(at("train", "valid", "a.jpg"), sp.VALID)        # the innermost folder wins
        self.assertIsNone(sp.split_from_path(os.path.join("test", "ds", "a.jpg"), os.path.join("test", "ds")))


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.labels = os.path.join(self.tmp.name, "labels")
        os.makedirs(self.labels)
        for name, text in (("a", "0 0.5 0.5 0.2 0.2\n"), ("b", "1 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.1 0.1\n"), ("c", "")):
            with open(os.path.join(self.labels, name + ".txt"), "w", encoding="utf-8") as f:
                f.write(text)
        self.store = sp.SplitStore(os.path.join(self.tmp.name, "voc"))
        self.store.set("b.png", sp.VALID)
        self.store.set("c.png", sp.TEST)
        self.index = sp.ClassIndex(self.labels)
        self.images = ["a.png", "b.png", "c.png", "d.png"]          # d has no label file at all
        self.classes = ["cat", "dog"]

    def run_filter(self, split=None, names=()):
        return sp.matching(self.images, self.store, self.index, self.classes, split, names)

    def test_no_filter_keeps_everything_in_order(self):
        self.assertEqual(self.run_filter(), self.images)

    def test_by_split(self):
        self.assertEqual(self.run_filter(sp.TRAIN), ["a.png", "d.png"])
        self.assertEqual(self.run_filter(sp.VALID), ["b.png"])
        self.assertEqual(self.run_filter(sp.TEST), ["c.png"])

    def test_by_class_means_any_of_the_chosen_classes(self):
        self.assertEqual(self.run_filter(names=["dog"]), ["b.png"])
        self.assertEqual(self.run_filter(names=["cat"]), ["a.png", "b.png"])
        self.assertEqual(self.run_filter(names=["cat", "dog"]), ["a.png", "b.png"])

    def test_split_and_class_together_and_an_unknown_class_matches_nothing(self):
        self.assertEqual(self.run_filter(sp.TRAIN, ["cat"]), ["a.png"])
        self.assertEqual(self.run_filter(sp.TEST, ["cat"]), [])
        self.assertEqual(self.run_filter(names=["bird"]), [])

    def test_the_class_index_follows_a_changed_label_file(self):
        self.assertEqual(self.run_filter(names=["dog"]), ["b.png"])
        with open(os.path.join(self.labels, "a.txt"), "w", encoding="utf-8") as f:
            f.write("1 0.5 0.5 0.2 0.2 0.3 0.3\n")                  # a segmentation row counts too
        self.assertEqual(self.run_filter(names=["dog"]), ["a.png", "b.png"])


if __name__ == "__main__":
    unittest.main()

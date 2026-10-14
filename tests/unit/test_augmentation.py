import os
import sys
import tempfile
import unittest

import numpy as np

from utils import augmentation as aug
from utils.augmentation import AugmentConfig, Augmenter, augment_image
from utils.export_preprocess import ResizeConfig


def photo(h=48, w=64):
    """A deterministic, colourful, non-flat test image (BGR uint8)."""
    rng = np.random.default_rng(1)
    base = rng.integers(40, 215, (h, w, 3), dtype=np.uint8)
    base[:, : w // 2, 2] = 220                       # strong red half so hue/saturation matter
    return base


def cfg(**ops):
    """cfg(brightness={'percent': 25}) -> only brightness; missing parameters take their defaults."""
    return AugmentConfig(ops={k: aug.clean_params(k, v) for k, v in ops.items()})


class TorchvisionOnOneImageTests(unittest.TestCase):
    """The colour augmentations run on torchvision.transforms.v2.functional: prove it on a single image."""

    TORCHVISION_OPS = ("brightness", "exposure", "hue", "saturation", "grayscale", "blur", "cutout")

    def test_each_torchvision_augmentation_changes_one_image_and_keeps_its_format(self):
        img = photo()
        extreme = {"brightness": {"pct": 40.0}, "exposure": {"pct": 40.0}, "hue": {"shift": 60.0},
                   "saturation": {"pct": -60.0}, "grayscale": {"on": True}, "blur": {"k": 7},
                   "cutout": {"boxes": [(0.2, 0.3, 0.3)]}}
        for key in self.TORCHVISION_OPS:
            out, tf = aug.apply(key, img, extreme[key])
            self.assertIsNone(tf, key)
            self.assertEqual(out.shape, img.shape, key)
            self.assertEqual(out.dtype, np.uint8, key)
            self.assertFalse(np.array_equal(out, img), key)
        import torchvision
        self.assertIn("torchvision.transforms.v2.functional", sys.modules)
        self.assertTrue(torchvision.__version__)

    def test_brightness_and_exposure_go_the_right_way(self):
        img = photo()
        self.assertGreater(aug.apply("brightness", img, {"pct": 30.0})[0].mean(), img.mean())
        self.assertLess(aug.apply("brightness", img, {"pct": -30.0})[0].mean(), img.mean())
        self.assertGreater(aug.apply("exposure", img, {"pct": 30.0})[0].mean(), img.mean())
        self.assertLess(aug.apply("exposure", img, {"pct": -30.0})[0].mean(), img.mean())

    def test_grayscale_removes_colour_and_off_keeps_it(self):
        out, _ = aug.apply("grayscale", photo(), {"on": True})
        self.assertTrue((out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all())
        self.assertTrue(np.array_equal(aug.apply("grayscale", photo(), {"on": False})[0], photo()))

    def test_saturation_down_makes_the_picture_greyer(self):
        spread = lambda im: float(im.max(axis=2).astype(int).__sub__(im.min(axis=2)).mean())
        self.assertLess(spread(aug.apply("saturation", photo(), {"pct": -80.0})[0]), spread(photo()))

    def test_cutout_blacks_out_boxes_only(self):
        white = np.full((40, 40, 3), 255, np.uint8)
        out, _ = aug.apply("cutout", white, {"boxes": [(0.0, 0.0, 0.25)]})
        self.assertTrue((out[:10, :10] == 0).all())
        self.assertTrue((out[20:, 20:] == 255).all())

    def test_noise_and_motion_blur_work_without_torch_ops(self):
        out, _ = aug.apply("noise", photo(), {"pct": 20.0, "seed": 3})
        self.assertGreater(np.count_nonzero((out != photo()).any(axis=2)), 0)
        edge = np.zeros((30, 30, 3), np.uint8)
        edge[:, 15:] = 255
        blurred, _ = aug.apply("motion_blur", edge, {"length": 9, "angle": 0.0})
        self.assertTrue(((blurred[:, 12:18] > 0) & (blurred[:, 12:18] < 255)).any())


class CatalogueTests(unittest.TestCase):
    def setUp(self):
        self.img = photo()
        self.rng = np.random.default_rng(7)

    def test_every_operation_samples_applies_and_keeps_dtype_and_channels(self):
        for key in aug.OPS:
            for seed in range(6):
                values = aug.sample(key, aug.default_params(key), np.random.default_rng(seed), (64, 48))
                out, tf = aug.apply(key, self.img, values)
                self.assertEqual(out.dtype, np.uint8, key)
                self.assertEqual(out.shape[2], 3, key)
                if tf is not None:
                    self.assertEqual(out.shape[:2], (tf.dst_h, tf.dst_w), key)

    def test_every_operation_has_a_preview_with_the_original_first(self):
        for key in aug.OPS:
            shots = aug.preview(key, aug.default_params(key), self.img)
            self.assertGreaterEqual(len(shots), 2, key)
            self.assertEqual(shots[0][0], "original")
            self.assertTrue(np.array_equal(shots[0][1], self.img))
            self.assertFalse(np.array_equal(shots[-1][1], self.img) and shots[-1][1].shape == self.img.shape, key)

    def test_previews_follow_the_checkboxes(self):
        flip = aug.preview("flip", {"horizontal": True, "vertical": False}, self.img)
        self.assertEqual([c for c, _ in flip], ["original", "horizontal"])
        bright = aug.preview("brightness", {"percent": 15, "brighten": True, "darken": False}, self.img)
        self.assertEqual([c for c, _ in bright], ["original", "15%"])
        rot = aug.preview("rot90", {"clockwise": True, "counter": False, "upside_down": True}, self.img)
        self.assertEqual([c for c, _ in rot], ["original", "clockwise", "upside down"])

    def test_clean_params_clamps_and_fills_defaults(self):
        self.assertEqual(aug.clean_params("brightness", {"percent": 500})["percent"], 99)
        self.assertEqual(aug.clean_params("blur", {"pixels": "x"})["pixels"], aug.default_params("blur")["pixels"])
        crop = aug.clean_params("crop", {"min_zoom": 50, "max_zoom": 10})
        self.assertLessEqual(crop["min_zoom"], crop["max_zoom"])
        self.assertTrue(aug.clean_params("flip", {})["horizontal"])

    def test_summary_names_the_setting(self):
        self.assertEqual(aug.summary("brightness", {"percent": 15}), "Brightness ±15%")
        self.assertIn("Flip", aug.summary("flip", aug.default_params("flip")))

    def test_sampled_values_stay_inside_the_range(self):
        for seed in range(40):
            rng = np.random.default_rng(seed)
            self.assertLessEqual(abs(aug.sample("rotation", {"degrees": 10}, rng, (64, 48))["angle"]), 10)
            self.assertLessEqual(abs(aug.sample("brightness", {"percent": 10, "brighten": True, "darken": True},
                                                rng, (64, 48))["pct"]), 10)
            self.assertGreaterEqual(aug.sample("brightness", {"percent": 10, "brighten": True, "darken": False},
                                               rng, (64, 48))["pct"], 0)
            k = aug.sample("blur", {"pixels": 9}, rng, (64, 48))["k"]
            self.assertTrue(k % 2 == 1 and 3 <= k <= 9)


class AugmentImageTests(unittest.TestCase):
    def test_a_copy_is_never_an_untouched_duplicate(self):
        img = photo()
        rng = np.random.default_rng(0)
        for _ in range(60):
            out, applied, _ = augment_image(img, cfg(flip={"horizontal": True, "vertical": False}), rng)
            self.assertIsNotNone(out)
            self.assertTrue(applied)
            self.assertFalse(np.array_equal(out, img))

    def test_nothing_enabled_means_no_copy(self):
        self.assertFalse(AugmentConfig(ops={}).enabled)
        self.assertEqual(augment_image(photo(), AugmentConfig(ops={}), np.random.default_rng(0)), (None, [], None))

    def test_geometry_gives_a_transform_and_colour_does_not(self):
        rng = np.random.default_rng(2)
        self.assertIsNone(augment_image(photo(), cfg(brightness={"percent": 30}), rng)[2])
        self.assertIsNotNone(augment_image(photo(), cfg(rotation={"degrees": 30}), rng)[2])

    def test_colour_runs_before_geometry_and_both_apply(self):
        out, applied, tf = augment_image(photo(), cfg(grayscale={"percent": 100}, flip={"horizontal": True,
                                                                                         "vertical": False}),
                                         np.random.default_rng(1))
        self.assertEqual(applied, ["grayscale", "flip"])
        self.assertTrue((out[..., 0] == out[..., 2]).all())
        self.assertIsNotNone(tf)


class SeedTests(unittest.TestCase):
    def run_once(self, seed):
        config = cfg(brightness={"percent": 25}, blur={"pixels": 5}, noise={"percent": 5}, rotation={"degrees": 20})
        config.seed, config.copies = seed, 3
        return [img for _, img, _ in Augmenter(config).copies(photo())]

    def test_same_seed_gives_identical_copies(self):
        a, b = self.run_once(42), self.run_once(42)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(x, y))

    def test_different_seed_gives_different_copies(self):
        a, b = self.run_once(1), self.run_once(2)
        self.assertFalse(all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b)))


class AugmenterTests(unittest.TestCase):
    def make(self, copies=2, **ops):
        config = cfg(**(ops or {"brightness": {"percent": 25}}))
        config.copies, config.seed = copies, 5
        return Augmenter(config)

    def test_makes_the_requested_number_of_named_copies(self):
        a = self.make(copies=3)
        copies = list(a.copies(photo()))
        self.assertEqual([c[0] for c in copies], ["_aug1", "_aug2", "_aug3"])
        self.assertEqual(a.created, 3)

    def test_keep_original_defaults_to_off(self):
        self.assertFalse(AugmentConfig(ops={}).keep_original)

    def test_copies_are_capped(self):
        self.assertEqual(len(list(self.make(copies=99).copies(photo()))), aug.MAX_COPIES)

    def test_no_image_or_nothing_enabled_yields_nothing(self):
        self.assertEqual(list(self.make().copies(None)), [])
        self.assertEqual(list(Augmenter(AugmentConfig(ops={})).copies(photo())), [])

    def test_copies_are_brought_back_to_the_resize_size(self):
        config = cfg(rotation={"degrees": 45})
        config.copies, config.seed = 3, 1
        config.resize = ResizeConfig("stretch", 64, 48)
        for _, img, tf in Augmenter(config).copies(photo()):
            self.assertEqual(img.shape[:2], (48, 64))
            self.assertEqual((tf.dst_w, tf.dst_h), (64, 48))

    def test_non_ascii_paths_round_trip(self):
        path = os.path.join(tempfile.mkdtemp(), "kucing_üñí_猫.png")
        aug.write_image(path, photo())
        self.assertTrue(np.array_equal(aug.read_image(path), photo()))


if __name__ == "__main__":
    unittest.main()

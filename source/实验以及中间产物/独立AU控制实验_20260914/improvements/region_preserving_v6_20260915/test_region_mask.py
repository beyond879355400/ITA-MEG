"""Synthetic, non-model tests of the frozen source-only geometric mask contract."""

import copy
import unittest

import numpy as np

from region_mask import AUS, _allowed_path, build_mask


def synthetic():
    # Entirely synthetic geometry, not extracted from any train/dev/test subject.
    points = np.zeros((68, 2), dtype=float)
    for i, angle in enumerate(np.linspace(np.pi, 0, 17)):
        points[i] = [256 + 180 * np.cos(angle), 230 + 230 * np.sin(angle)]
    points[17:22] = [[115, 190], [140, 180], [165, 178], [190, 185], [215, 195]]
    points[22:27] = [[297, 195], [322, 185], [347, 178], [372, 180], [397, 190]]
    points[27:36] = [[256, 230], [256, 260], [256, 290], [256, 315],
                     [220, 330], [238, 340], [256, 345], [274, 340], [292, 330]]
    points[36:42] = [[130, 230], [150, 220], [178, 220], [200, 230], [178, 242], [150, 242]]
    points[42:48] = [[312, 230], [334, 220], [362, 220], [382, 230], [362, 242], [334, 242]]
    for i, angle in enumerate(np.linspace(np.pi, -np.pi, 12, endpoint=False)):
        points[48 + i] = [256 + 66 * np.cos(angle), 380 - 18 * np.sin(angle)]
    for i, angle in enumerate(np.linspace(np.pi, -np.pi, 8, endpoint=False)):
        points[60 + i] = [256 + 45 * np.cos(angle), 380 - 6 * np.sin(angle)]
    raw = {f"{axis}_{i}": str(points[i, dim]) for i in range(68)
           for dim, axis in enumerate(("x", "y"))}
    source = {"path": "/synthetic/train/source.png", "sha256": "a" * 64}
    case = {"case_id": "synthetic_train_case", "split": "train", "source_real": source,
            "source_aus_0_5": [0] * 12, "target_aus_0_5": [0] * 12}
    record = {"method": "source_real", "split": "train", **source,
              "image_size": [512, 512], "parse_status": "primary_face_finite",
              "confidence": .98, "raw_rows": [raw], "selected_row": 0}
    return case, record


def edit(case, *aus):
    case = copy.deepcopy(case)
    for au in aus:
        case["target_aus_0_5"][AUS.index(au)] = 1
    return case


class RegionMaskTests(unittest.TestCase):
    def setUp(self):
        self.case, self.record = synthetic()

    def test_no_edit_is_exact_zero(self):
        mask, receipt = build_mask(self.case, self.record)
        self.assertEqual(mask.dtype, np.float32)
        self.assertEqual(mask.shape, (64, 64))
        self.assertEqual(np.count_nonzero(mask), 0)
        self.assertEqual(receipt["changed_aus"], [])
        self.assertEqual(receipt["mask_area"]["alpha_sum_64"], 0)

    def test_each_au_is_bounded_finite_soft_and_not_whole_face(self):
        for au in AUS:
            mask, receipt = build_mask(edit(self.case, au), self.record)
            self.assertTrue(np.isfinite(mask).all())
            self.assertGreater(mask.max(), .5)
            self.assertLessEqual(mask.max(), 1)
            self.assertGreaterEqual(mask.min(), 0)
            self.assertTrue(np.any((mask > 0) & (mask < 1)))
            self.assertLess(np.mean(mask > 0), .25)
            self.assertEqual(receipt["per_au"][0]["au"], au)

    def test_mouth_and_chin_never_open_eye_or_nose(self):
        for au in (12, 15, 17, 20, 25, 26):
            mask, receipt = build_mask(edit(self.case, au), self.record)
            self.assertEqual(np.count_nonzero(mask[:43, :]), 0)
            self.assertTrue(all("mouth" in name for name in receipt["active_regions"]))
        mouth, _ = build_mask(edit(self.case, 12), self.record)
        chin, _ = build_mask(edit(self.case, 17), self.record)
        self.assertGreater(chin.sum(), mouth.sum())

    def test_source_only_ignores_target_images_geometry_and_results(self):
        case = edit(self.case, 1, 12)
        expected, receipt = build_mask(case, self.record)
        case.update({"target_real": {"path": "/should/not/be/read.png"},
                     "target_alignment": {"landmarks": "malformed"},
                     "target_landmarks": [[float("nan"), 99999]],
                     "generated_image": "unavailable", "metrics": {"score": 999}})
        record = copy.deepcopy(self.record)
        record.update({"vector": [99] * 12, "selected_au17": {"1": 5},
                       "target_labels_12": [5] * 12, "target_landmarks": "unused"})
        actual, actual_receipt = build_mask(case, record)
        np.testing.assert_array_equal(expected, actual)
        self.assertEqual(receipt, actual_receipt)

    def test_actual_source_landmarks_control_the_mask(self):
        case = edit(self.case, 12)
        expected, _ = build_mask(case, self.record)
        record = copy.deepcopy(self.record)
        for index in range(48, 68):
            record["raw_rows"][0][f"x_{index}"] = str(float(record["raw_rows"][0][f"x_{index}"]) + 20)
        changed, _ = build_mask(case, record)
        self.assertFalse(np.array_equal(expected, changed))

    def test_au_sign_and_magnitude_do_not_tune_geometric_extent(self):
        increasing = edit(self.case, 12)
        decreasing = copy.deepcopy(self.case)
        decreasing["source_aus_0_5"][6] = 5
        expected, _ = build_mask(increasing, self.record)
        actual, _ = build_mask(decreasing, self.record)
        np.testing.assert_array_equal(expected, actual)

    def test_unknown_au_is_rejected(self):
        case = copy.deepcopy(self.case)
        case["au_order"] = list(AUS[:-1]) + [99]
        with self.assertRaisesRegex(ValueError, "Unknown AU"):
            build_mask(case, self.record)
        case = copy.deepcopy(self.case)
        case["changed_aus"] = [99]
        with self.assertRaisesRegex(ValueError, "Unknown AU"):
            build_mask(case, self.record)

    def test_test_split_and_path_are_rejected(self):
        case = copy.deepcopy(self.case)
        case["split"] = "test"
        with self.assertRaises(ValueError):
            build_mask(case, self.record)
        with self.assertRaises(ValueError):
            _allowed_path("/synthetic/final_test/source.png")
        with self.assertRaises(ValueError):
            _allowed_path("relative/train/source.png")

    def test_target_role_and_source_sha_mismatch_are_rejected(self):
        for mutation in ({"method": "target_real"}, {"sha256": "b" * 64},
                         {"path": "/synthetic/dev/other.png"}):
            with self.assertRaises(ValueError):
                build_mask(self.case, dict(self.record, **mutation))

    def test_non512_and_invalid_geometry_are_rejected(self):
        for mutation in ({"image_size": [256, 256]}, {"selected_row": -1},
                         {"confidence": 0.1}, {"parse_status": "no_face"}):
            with self.assertRaises(ValueError):
                build_mask(self.case, dict(self.record, **mutation))
        for invalid in ("nan", "10000", "missing"):
            record = copy.deepcopy(self.record)
            record["raw_rows"][0]["x_0"] = invalid
            with self.assertRaises(ValueError):
                build_mask(self.case, record)

    def test_limited_crop_edge_extrapolation_is_recorded_not_rescaled(self):
        record = copy.deepcopy(self.record)
        record["raw_rows"][0]["y_8"] = "520"
        mask, receipt = build_mask(edit(self.case, 26), record)
        self.assertIn(8, receipt["landmarks_outside_crop_indices"])
        self.assertEqual(receipt["source_landmarks_68_512"][8][1], 520)
        self.assertEqual(mask.shape, (64, 64))

    def test_conflicting_no_edit_group_rejected(self):
        case = edit(self.case, 12)
        case["group"] = "no_edit"
        with self.assertRaises(ValueError):
            build_mask(case, self.record)


if __name__ == "__main__":
    unittest.main()

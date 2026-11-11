"""Synthetic-only v7 tests; CPU NumPy/PIL, no real faces or model execution."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

import numpy as np
from PIL import Image

import region_mask as v7


# Reuse only the existing entirely hand-constructed 68-point fixture, read-only.
fixture_path = v7.V6_PATH.parent / "test_region_mask.py"
if v7.file_sha(fixture_path) != "8f865c928ece279a7a4dfa9138b9ea07b87d1467fab8ae4a76089b46e7cacab0":
    raise ValueError("Frozen synthetic fixture changed")
spec = importlib.util.spec_from_file_location("readonly_v6_synthetic_fixture", fixture_path)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)

try:
    from scipy.ndimage import gaussian_filter, maximum_filter
except ImportError:
    gaussian_filter = maximum_filter = None


class SmoothMaskTests(unittest.TestCase):
    def setUp(self):
        self.case, self.record = fixture.synthetic()

    def test_no_edit_is_strictly_zero_float32(self):
        mask, receipt = v7.build_mask(self.case, self.record)
        self.assertEqual(mask.dtype, np.float32)
        self.assertEqual(mask.shape, (64, 64))
        self.assertEqual(np.count_nonzero(mask), 0)
        self.assertEqual(receipt["mask_area"]["alpha_sum_64"], 0)
        self.assertIsNone(receipt["smoothed_support_bbox_64"])
        self.assertEqual(receipt["base_v6"]["mask_bytes_sha256"], v7._mask_sha(mask))

    def test_receipt_binds_exact_v6_mask_and_v7_parameters(self):
        case = fixture.edit(self.case, 12)
        base, base_receipt = v7.BASE_V6.build_mask(case, self.record)
        mask, receipt = v7.build_mask(case, self.record)
        self.assertEqual(receipt["base_v6"]["mask_bytes_sha256"], v7._mask_sha(base))
        self.assertEqual(receipt["base_v6"]["mask_area"], base_receipt["mask_area"])
        self.assertEqual(receipt["base_v6"]["policy_sha256"], base_receipt["policy_sha256"])
        self.assertEqual(receipt["base_v6"]["code"]["sha256"], v7.V6_CODE_SHA256)
        self.assertEqual(receipt["policy_sha256"], v7.canonical_sha(v7.POLICY))
        self.assertEqual(receipt["mask_bytes_sha256"], v7._mask_sha(mask))
        self.assertEqual(receipt["smoothing"]["maxpool"]["kernel_size"], [7, 7])
        self.assertEqual(receipt["smoothing"]["gaussian"]["sigma_latent_px"], 1.5)
        self.assertEqual(receipt["smoothing"]["gaussian"]["truncate_sigma"], 3.0)
        self.assertGreater(receipt["mask_area"]["alpha_sum_64"], base_receipt["mask_area"]["alpha_sum_64"])
        self.assertTrue(np.isfinite(mask).all())
        self.assertGreaterEqual(mask.min(), 0)
        self.assertLessEqual(mask.max(), 1)

    def test_single_impulse_has_fixed_context_and_truncated_tail(self):
        impulse = np.zeros((64, 64), dtype=np.float32)
        impulse[32, 32] = 1
        mask = v7.smooth_mask(impulse)
        self.assertEqual(v7._support_bbox(mask), [24, 24, 41, 41])
        self.assertEqual(np.count_nonzero(mask), 17 * 17)
        self.assertAlmostEqual(float(mask.sum(dtype=np.float64)), 49, places=6)
        self.assertGreater(mask[32, 32], .9)
        self.assertEqual(mask[23, 32], 0)

    def test_constant_extremes_and_input_not_modified(self):
        for value in (0, 1):
            original = np.full((64, 64), value, dtype=np.float32)
            expected = original.copy()
            result = v7.smooth_mask(original)
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_array_equal(original, expected)

    def test_kernel_is_fixed_normalized_symmetric(self):
        weights = v7._kernel()
        self.assertEqual(len(weights), 11)
        np.testing.assert_array_equal(weights, weights[::-1])
        self.assertAlmostEqual(weights.sum(), 1, places=15)
        self.assertTrue(np.all(weights > 0))

    @unittest.skipUnless(gaussian_filter is not None, "scipy optional independent reference")
    def test_numpy_implementation_matches_scipy_fixed_parameters(self):
        for base in (np.random.default_rng(7).random((64, 64)).astype(np.float32),
                     np.eye(64, dtype=np.float32)):
            dilated = maximum_filter(base.astype(np.float64), size=7, mode="constant", cval=0)
            expected = gaussian_filter(dilated, sigma=1.5, truncate=3.0, mode="reflect").astype(np.float32)
            actual = v7.smooth_mask(base)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=6e-8)

    def test_target_geometry_scores_and_generated_results_do_not_affect_mask(self):
        case = fixture.edit(self.case, 1, 12)
        expected, original_receipt = v7.build_mask(case, self.record)
        case.update({"target_alignment": {"landmarks": "invalid"}, "target_real": {"path": "/not/read.png"},
                     "target_landmarks": [[float("nan"), 9999]], "generation_result": "ignored", "metric": 999})
        record = dict(self.record, target_landmarks="unused", vector=[99] * 12)
        actual, receipt = v7.build_mask(case, record)
        np.testing.assert_array_equal(expected, actual)
        self.assertEqual(original_receipt, receipt)

    def test_invalid_masks_rejected(self):
        for value in (np.zeros((32, 32), np.float32), np.zeros((64, 64), np.float64),
                      np.full((64, 64), np.nan, np.float32), np.full((64, 64), 2, np.float32)):
            with self.assertRaises(ValueError):
                v7.smooth_mask(value)

    def test_v6_source_guardrails_preserved(self):
        for case, record in ((dict(self.case, split="test"), self.record),
                             (dict(self.case, changed_aus=[99]), self.record),
                             (self.case, dict(self.record, method="target_real")),
                             (self.case, dict(self.record, image_size=[256, 256]))):
            with self.assertRaises(ValueError):
                v7.build_mask(case, record)

    def test_synthetic_prepare_8_cases_4_sources_schema_and_hashes(self):
        # These are solid-color synthetic PNGs only, not real source images.
        parent = v7.HERE / "synthetic_checks"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fixture_", dir=parent) as tmp:
            root = Path(tmp)
            cases, records = [], []
            for subject in range(4):
                path = root / f"source_{subject}.png"
                Image.new("RGB", (512, 512), (20 + 30 * subject, 70, 100)).save(path)
                source = {"path": str(path), "sha256": v7.file_sha(path)}
                for is_edit in (False, True):
                    case = fixture.edit(self.case, 12) if is_edit else copy.deepcopy(self.case)
                    case.update(case_id=f"synthetic_train_s{subject}_{'edit' if is_edit else 'noedit'}",
                                subject=f"S{subject}", group="lowest_available" if is_edit else "no_edit", source_real=source)
                    record = dict(self.record, case_id=case["case_id"], subject=case["subject"], **source)
                    cases.append(case)
                    records.append(record)
            case_path, record_path = root / "cases.json", root / "measurements.json"
            case_path.write_text(json.dumps({"case_count": 8, "test_data_used": False, "cases": cases}))
            record_path.write_text(json.dumps({"test_used": False, "records": records}))
            base_before, receipt_before = v7.BASE_V6.build_mask(cases[1], records[1])
            output = root / "prepared"
            manifest = v7.prepare(str(case_path), str(record_path), str(output))
            self.assertEqual(json.loads(json.dumps(manifest)), json.loads((output / "manifest.json").read_text()))
            self.assertEqual(manifest["inputs"]["code_sha256"], v7.file_sha(v7.__file__))
            self.assertNotEqual(manifest["inputs"]["code_sha256"], v7.V6_CODE_SHA256)
            self.assertEqual(manifest["policy_sha256"], v7.canonical_sha(v7.POLICY))
            self.assertEqual(manifest["unique_source_count"], 4)
            self.assertEqual(manifest["case_count"], 8)
            self.assertTrue(manifest["source_unchanged"])
            self.assertTrue(manifest["source_only"])
            self.assertFalse(manifest["test_used"])
            for index, entry in enumerate(manifest["cases"]):
                self.assertEqual(v7.file_sha(entry["receipt_path"]), entry["receipt_sha256"])
                self.assertEqual(v7.file_sha(entry["mask64"]["path"]), entry["mask64"]["sha256"])
                receipt = json.loads(Path(entry["receipt_path"]).read_text())
                self.assertEqual(receipt["source_sha256"], cases[index]["source_real"]["sha256"])
                np.testing.assert_array_equal(np.load(entry["mask64"]["path"], allow_pickle=False),
                                              v7.build_mask(cases[index], records[index])[0])
            base_after, receipt_after = v7.BASE_V6.build_mask(cases[1], records[1])
            np.testing.assert_array_equal(base_before, base_after)
            self.assertEqual(receipt_before, receipt_after)
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                v7.prepare(str(case_path), str(record_path), str(output))
        self.assertEqual(v7.file_sha(v7.V6_PATH), v7.V6_CODE_SHA256)


if __name__ == "__main__":
    unittest.main()

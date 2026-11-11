"""Metadata/CLI checks only. No real image loading or render execution."""

import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
import render_v7 as v7


class RenderV7Tests(unittest.TestCase):
    def test_configs_exactly_match_fixed_v7_matrix(self):
        self.assertEqual(v7.RENDERER.CONFIGS, v7.CONFIGS)
        new = [c for c in v7.RENDERER.columns() if c["kind"] == "v7"]
        self.assertEqual(len(new), 6)
        self.assertEqual([c["expected_steps"] for c in new], [15] * 6)
        self.assertEqual([c["strength"] for c in new], [.75] * 6)
        self.assertEqual([c["au_scale"] for c in new], [1, 1, 1.5, 1.5, 2, 2])
        self.assertEqual([c["key"] for c in new], [f"{method}_{config}" for config, _, _, _ in v7.CONFIGS
                                                  for method in ("baseline", "bidirectional")])

    def test_headers_do_not_mislabel_v7_as_v6(self):
        cols = v7.RENDERER.columns()
        self.assertEqual(len(cols), 10)
        self.assertNotIn("v6", json.dumps(cols, ensure_ascii=False))
        self.assertEqual([c["key"] for c in cols[:4]], ["source_real", "target_real",
                         "bidirectional_img2img_s055", "bidirectional_img2img_s075"])
        self.assertEqual([c["expected_steps"] for c in cols[2:4]], [11, 15])
        debug = v7.RENDERER.columns(True)
        self.assertEqual(len(debug), 11)
        self.assertEqual(debug[-1]["kind"], "debug")
        self.assertIn("不是生成图", " ".join(debug[-1]["lines"]))

    def test_paths_and_metadata_patch_bind_v7_entry_point(self):
        self.assertEqual(v7.RENDERER.HERE, v7.HERE)
        self.assertEqual(Path(v7.RENDERER.__file__), Path(v7.__file__))
        self.assertEqual(v7.RENDERER.metadata_adaptation_counts, v7.EXPECTED_COUNTS)
        self.assertEqual(v7.RENDERER.sha(v7.TEMPLATE), v7.TEMPLATE_SHA256)

    def test_missing_generation_record_keeps_explicit_cell(self):
        cell = v7.RENDERER.cell_source({"case_id": "synthetic"}, v7.RENDERER.columns()[4], {}, {})
        self.assertEqual(cell["generation_status"], "missing_record")
        self.assertIsNone(cell["path"])

    def test_rendering_refused_without_terminal_result(self):
        path = v7.HERE / "runs" / "synthetic_missing_terminal"
        self.assertFalse(path.exists())
        with self.assertRaisesRegex(ValueError, "No terminal result.json"):
            v7.RENDERER.load_inputs(path)


if __name__ == "__main__":
    unittest.main()

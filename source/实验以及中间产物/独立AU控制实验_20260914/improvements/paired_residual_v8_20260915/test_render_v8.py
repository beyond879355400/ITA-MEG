"""V8 rendering metadata tests only; never load real images or run a model."""

import copy
import json
import sys
import unittest

sys.dont_write_bytecode = True
import render_v8 as r


def config():
    return {"kind": "v8_paired_residual_diagnostic", "gains": [{"name": n, "gain": g} for n, g in r.GAINS],
            "methods": ["baseline", "bidirectional"], "seed": 101, "nominal_steps": 20,
            "strength": .75, "actual_steps": 15, "guidance_scale": 7.5, "au_scale": 1., "ip_scale": .9,
            "solver": r.SOLVER, "scheduler_step_calls": 0, "final_rho": 0.,
            "test_data_used": False, "post_generation_pixel_blend": False}


class RenderV8Tests(unittest.TestCase):
    def test_ten_columns_and_exact_three_gains(self):
        cols = r.columns()
        self.assertEqual(len(cols), 10)
        self.assertEqual([c["key"] for c in cols[:4]], ["source_real", "target_real", "bidirectional_img2img_s055", r.V7_METHOD])
        self.assertEqual([c["gain"] for c in cols[4:]], [.25, .25, .5, .5, 1, 1])
        self.assertEqual([c["expected_steps"] for c in cols[4:]], [15] * 6)
        self.assertEqual([c["key"] for c in cols[4:]], [f"{m}_{n}" for n, _ in r.GAINS for m in ("baseline", "bidirectional")])
        self.assertTrue(all(c["kind"] == "v8" for c in cols[4:]))

    def test_first_order_labels_and_old_comparator_labels_are_accurate(self):
        cols = r.columns()
        for col in cols[4:]:
            text = " ".join(col["lines"])
            self.assertIn("AU×1", text)
            self.assertIn("CFG7.5", text)
            self.assertIn("15次一阶rho更新", text)
            self.assertNotIn("DEIS", text)
        self.assertIn("11步", " ".join(cols[2]["lines"]))
        self.assertIn("AU×1.5", " ".join(cols[3]["lines"]))
        self.assertIn("15步", " ".join(cols[3]["lines"]))

    def test_important_headers_fit_without_ellipsis(self):
        fonts, _ = r.layout.make_fonts()
        for col in r.columns():
            for text in col["lines"]:
                self.assertLessEqual(fonts[20].getlength(text), r.CELL - 16, text)

    def test_configuration_validates_gain_not_au_scale(self):
        r.validate_config(config())
        for mutation in ({"au_scale": .25}, {"actual_steps": 20}, {"scheduler_step_calls": 15},
                         {"gains": [{"name": "paired_g025", "gain": 1.}]}, {"guidance_scale": 1.}):
            with self.assertRaises(ValueError):
                r.validate_config(dict(config(), **mutation))

    def test_v7_reference_is_read_from_separate_records_without_changing_case(self):
        case = {"case_id": "synthetic"}
        before = copy.deepcopy(case)
        old = {(case["case_id"], r.V7_METHOD): {"output": "/synthetic/v7.png", "sha256": "a" * 64,
                                              "status": "complete", "actual_steps": 15}}
        actual = r.cell_source(case, r.columns()[3], {}, old)
        self.assertEqual(actual["path"], "/synthetic/v7.png")
        self.assertEqual(actual["expected_sha256"], "a" * 64)
        self.assertEqual(case, before)

    def test_missing_new_or_old_results_remain_explicit(self):
        for col in r.columns()[2:]:
            source = r.cell_source({"case_id": "synthetic"}, col, {}, {})
            self.assertEqual(source["generation_status"], "missing_record")
            self.assertIsNone(source["path"])

    def test_nonterminal_render_is_refused_and_template_is_unchanged(self):
        path = r.HERE / "runs" / "synthetic_missing_terminal"
        self.assertFalse(path.exists())
        with self.assertRaisesRegex(ValueError, "No terminal result.json"):
            r.load_inputs(path)
        self.assertEqual(r.sha(r.TEMPLATE), r.TEMPLATE_SHA)


if __name__ == "__main__":
    unittest.main()

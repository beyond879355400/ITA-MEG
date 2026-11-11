"""Terminal-only v8 paired-rho contact sheets; all cases/configs, no retouching."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "region_preserving_v6_20260915" / "render_v6.py"
TEMPLATE_SHA = "107143fe267144d9c8ac5b500faa5395f9179008038855ccdc0367133ab690d0"
V5RUN = HERE.parent / "quality_diagnosis_v5_20260915" / "runs" / "source_init_202609151525"
V5_RESULT_SHA = "4e92651f47abadfefdf0be2a91600544c1142cbbd99ebab411b2bea0b22177ce"
V7RUN = HERE.parent / "smooth_regions_v7_20260915" / "runs" / "smooth_202609151710"
V7_RESULT_SHA = "e59ce11ad87ff6bba91c8131019997833fee0f482a3df364e2d6f28e80aa079d"
V7_METHOD = "bidirectional_smooth_s075_a150"
GAINS = (("paired_g025", .25), ("paired_g050", .5), ("paired_g100", 1.))
SOLVER = "explicit_first_order_rho_on_DEIS_query_grid"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if sha(TEMPLATE) != TEMPLATE_SHA:
    raise ValueError("Frozen scientific rendering template changed")
spec = importlib.util.spec_from_file_location("v8_readonly_v6_scientific_layout", TEMPLATE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
layout = base.layout
need, read, ref = base.need, base.read, base.ref
SUBJECTS, GROUPS, CELL, GAP, MARGIN, AUS = base.SUBJECTS, base.GROUPS, base.CELL, base.GAP, base.MARGIN, base.AUS


def columns():
    result = [
        {"key": "source_real", "kind": "reference", "lines": ["源图", "原始对齐图", "仅参照 · 无模型更新"]},
        {"key": "target_real", "kind": "reference", "lines": ["真实目标", "同一固定案例", "仅参照 · 不用于生成输入"]},
        {"key": "bidirectional_img2img_s055", "kind": "v5", "expected_steps": 11,
         "lines": ["v5 双向 · 无区域保留", "强度 .55 / AU×1", "实际扩散去噪11步"]},
        {"key": V7_METHOD, "kind": "v7", "expected_steps": 15,
         "lines": ["v7 双向 · 平滑区域", "强度 .75 / AU×1.5", "实际扩散去噪15步"]},
    ]
    for config, gain in GAINS:
        for method, label in (("baseline", "基底"), ("bidirectional", "双向")):
            result.append({"key": f"{method}_{config}", "kind": "v8", "config": config, "gain": gain,
                           "expected_steps": 15, "solver": SOLVER,
                           "lines": [f"v8 配对残差 · {label}", f"gain {gain:g} / AU×1 / CFG7.5", "实际15次一阶rho更新"]})
    need(len(result) == 10, "Exactly ten scientific columns required")
    return result


def _bound_doc(parent, result, key, filename):
    binding = result[key]
    path = parent / filename
    need(Path(binding["path"]).resolve() == path and sha(path) == binding["sha256"], f"{key} binding changed")
    return read(path)


def validate_config(config):
    need(config.get("kind") == "v8_paired_residual_diagnostic", "Wrong v8 config kind")
    need(config.get("gains") == [{"name": n, "gain": g} for n, g in GAINS], "Paired gain matrix changed")
    need(config.get("methods") == ["baseline", "bidirectional"], "Method matrix changed")
    expected = {"seed": 101, "nominal_steps": 20, "strength": .75, "actual_steps": 15,
                "guidance_scale": 7.5, "au_scale": 1., "ip_scale": .9,
                "solver": SOLVER, "scheduler_step_calls": 0, "final_rho": 0.}
    need(all(config.get(key) == value for key, value in expected.items()), "Paired solver/scales/step metadata changed")
    need(config.get("test_data_used") is False and config.get("post_generation_pixel_blend") is False,
         "Unexpected test use or post-generation retouch")


def load_inputs(run):
    need(run.parent == HERE / "runs", "Input must be a direct v8 runs child")
    need((run / "result.json").is_file(), "No terminal result.json; active generation cannot be rendered")
    result = read(run / "result.json")
    need(result.get("kind") == "v8_paired_residual_diagnostic", "Wrong v8 result kind")
    need(result.get("status") in ("complete", "completed_with_failures", "stopped_with_failure"), "Nonterminal run")
    need(result.get("test_data_used") is False and result.get("planned") == 48, "Expected fixed dev-only 48 outputs")
    document = _bound_doc(run, result, "cases", "cases.json")
    config = _bound_doc(run, result, "configuration", "configuration.json")
    validate_config(config)
    need(document.get("kind") == "v8_development_paired_cases" and document.get("test_data_used") is False, "Wrong cases protocol")
    cases = document["cases"]
    need(document.get("case_count") == len(cases) == 8, "All eight fixed cases required")
    need(len({c["case_id"] for c in cases}) == 8 and all(c.get("split") == "dev" for c in cases), "Duplicate/non-dev cases")
    need(Counter((c["subject"], c["group"]) for c in cases) == Counter((s, g) for s in SUBJECTS for g in GROUPS),
         "All four subjects and both groups are required")
    known = {c["case_id"] for c in cases}
    allowed = {c["key"]: c for c in columns() if c["kind"] == "v8"}
    records = {}
    for row in result["records"]:
        key = row["case_id"], row["method"]
        need(key[0] in known and key[1] in allowed and key not in records, "Unexpected/duplicate generated row")
        col = allowed[key[1]]
        expected = {"config": col["config"], "gain": col["gain"], "seed": 101, "split": "dev", "strength": .75,
                    "au_scale": 1., "guidance_scale": 7.5, "nominal_steps": 20, "scheduler_step_calls": 0}
        need(all(row.get(k) == v for k, v in expected.items()), "Record does not match labelled paired protocol")
        if row.get("status") == "complete":
            need(row.get("actual_steps") == 15 and len(row.get("step_trace", [])) == 15, "Unexpected rho update count")
        records[key] = row
    need(sha(V5RUN / "result.json") == V5_RESULT_SHA, "Frozen v5 result changed")
    need(Path(config["v5_cases"]["path"]).resolve() == V5RUN / "cases.json"
         and sha(V5RUN / "cases.json") == config["v5_cases"]["sha256"], "Frozen v5 case binding changed")
    old = {(r["case_id"], r["method"]): r for r in read(V5RUN / "result.json")["records"]}
    need(sha(V7RUN / "result.json") == V7_RESULT_SHA, "Frozen v7 result changed")
    v7 = read(V7RUN / "result.json")
    need(v7.get("status") == "complete" and v7.get("test_data_used") is False, "Invalid frozen v7 comparator")
    v7cases = {c["case_id"]: c for c in _bound_doc(V7RUN, v7, "cases", "cases.json")["cases"]}
    for case in cases:
        need(case["case_id"] in v7cases, "Missing paired v7 comparator case")
        previous = v7cases[case["case_id"]]
        keys = ("subject", "split", "group", "source_aus_0_5", "target_aus_0_5", "source_real", "target_real")
        need(all(previous[key] == case[key] for key in keys), "V7 comparator source/target identity differs")
    for row in v7["records"]:
        if row["method"] == V7_METHOD:
            need(row.get("config") == "smooth_s075_a150" and row.get("strength") == .75
                 and row.get("au_scale") == 1.5 and row.get("actual_steps") == 15 and row.get("seed") == 101,
                 "V7 comparator label/scales mismatch")
            old[row["case_id"], row["method"]] = row
    # No references are added to cases, and no cases file is ever rewritten.
    return cases, records, old, result


def cell_source(case, col, records, old):
    if col["kind"] == "reference":
        source = case.get(col["key"], {})
        return {"path": source.get("path"), "expected_sha256": source.get("sha256"),
                "generation_status": "reference", "actual_steps": None}
    row = (records if col["kind"] == "v8" else old).get((case["case_id"], col["key"]))
    if row is None:
        return {"path": None, "expected_sha256": None, "generation_status": "missing_record", "actual_steps": None}
    if col["kind"] == "v5" and row.get("status") == "complete":
        need(case[col["key"]] == {"path": row["output"], "sha256": row["sha256"]}, "V5 retained reference changed")
        need(row.get("actual_steps") == 11, "V5 comparator step count changed")
    return {"path": row.get("output"), "expected_sha256": row.get("sha256"),
            "generation_status": row.get("status"), "actual_steps": row.get("actual_steps"),
            "actual_timesteps": row.get("actual_timesteps"), "error": row.get("error"),
            "update_semantics": "first_order_rho" if col["kind"] == "v8" else "diffusion_denoising"}


def render_page(name, selected, records, old, output, fonts):
    cols = columns()
    width, height = 2 * MARGIN + 10 * CELL + 9 * GAP, 256 + len(selected) * 398 + 130
    page = Image.new("RGB", (width, height), layout.BACKGROUND)
    draw = ImageDraw.Draw(page)
    title = "全部4主体 × 两类固定案例" if len(selected) == 8 else f"{selected[0]['subject']} × 两类固定案例"
    draw.text((MARGIN, 18), f"v8 配对残差诊断｜{title}", font=fonts[42], fill=layout.TEXT)
    draw.text((MARGIN, 74), "每主体先零编辑、后编辑；全部48张新图与所有gain保留，无质量筛选。seed101；开发集，不是独立最终测试。", font=fonts[25], fill=layout.MUTED)
    draw.text((MARGIN, 112), "原图仅整图等比排版，不裁脸、不修图。v8固定AU×1、CFG7.5，执行一阶rho残差更新，不是DEIS采样。", font=fonts[22], fill=layout.MUTED)
    for index, col in enumerate(cols):
        x = MARGIN + index * (CELL + GAP)
        draw.rounded_rectangle((x, 157, x + CELL, 252), radius=8, fill="#e0e9f5" if col["kind"] == "v8" else "#e6e9ef")
        for offset, text in enumerate(col["lines"]):
            layout.centered(draw, text, x, 164 + offset * 28, fonts[20])
    cells = []
    for row_index, case in enumerate(selected):
        top = 267 + row_index * 398
        changed = [f"AU{au}:{s:g}→{t:g}" for au, s, t in zip(AUS, case["source_aus_0_5"], case["target_aus_0_5"]) if s != t]
        group = "零编辑" if case["group"] == "no_edit" else "编辑"
        note = " / ".join(changed) or "AU控制无变化；配对场相同，源VAE重建为代数保证"
        draw.text((MARGIN, top), f"{case['subject']} · {group}｜{case['case_id']}｜{note}", font=fonts[22], fill=layout.TEXT)
        for column_index, col in enumerate(cols):
            x, y = MARGIN + column_index * (CELL + GAP), top + 37
            source = cell_source(case, col, records, old)
            info = layout.paste_cell(page, draw, x, y, source, fonts)
            if col["kind"] == "v8":
                # Replace only the contact-sheet caption, never source pixels.
                draw.rectangle((x, y + CELL + 10, x + CELL, y + CELL + 39), fill=layout.BACKGROUND)
                steps = source.get("actual_steps")
                caption = f"实际一阶rho更新 {steps} 次" if type(steps) is int else "一阶rho更新未完成/未记录"
                layout.centered(draw, caption, x, y + CELL + 14, fonts[18], layout.MUTED)
                info["display_step_caption"] = caption
            info.update(case_id=case["case_id"], subject=case["subject"], group=case["group"], row=row_index,
                        column=column_index, column_key=col["key"], column_kind=col["kind"], column_label=col["lines"])
            cells.append(info)
    footer = height - 110
    draw.line((MARGIN, footer, width - MARGIN, footer), fill="#cbd3df", width=2)
    draw.text((MARGIN, footer + 15), "顺序：源 / 目标 / v5双向.55 / v7双向.75-AU×1.5 / v8 gain.25、.5、1（每组基底、双向；各15次rho更新）。", font=fonts[20], fill=layout.MUTED)
    draw.text((MARGIN, footer + 48), "gain是残差步进增益，不是AUscale；v8全部AUscale=1。旧v5/v7为实际留存的不同算法对照，不等同预算或同求解器实验。", font=fonts[20], fill=layout.MUTED)
    draw.text((MARGIN, footer + 81), "缺失、失败或SHA不符保留占位格；全部case、原图SHA、显示边界和页面SHA见index。没有生成图后处理。", font=fonts[18], fill=layout.MUTED)
    destination = output / f"{name}.png"
    with destination.open("xb") as handle:
        page.save(handle, format="PNG")
    return {**ref(destination), "size": [width, height], "case_ids": [c["case_id"] for c in selected], "columns": cols, "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Terminal v8 run; no models are loaded")
    args = parser.parse_args()
    run = args.run.resolve()
    cases, records, old, result = load_inputs(run)
    fonts, font_path = layout.make_fonts()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = HERE / "rendered" / f"{run.name}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    pages = [render_page("all_four_subjects_all_cases", base.ordered_cases(cases, SUBJECTS), records, old, output, fonts)]
    pages += [render_page(f"{s}_all_cases", base.ordered_cases(cases, [s]), records, old, output, fonts) for s in SUBJECTS]
    observed = {c["path"]: c["actual_sha256"] for page in pages for c in page["cells"] if c.get("actual_sha256")}
    need(all(sha(path) == digest for path, digest in observed.items()), "Input image changed during rendering")
    need(sha(run / "cases.json") == result["cases"]["sha256"], "V8 cases changed during rendering")
    overview = pages[0]
    manifest = {"kind": "v8_paired_rho_scientific_contact_sheets", "status": "rendered_pending_visual_review",
                "created_utc": datetime.now(timezone.utc).isoformat(), "test_data_used": False, "models_loaded": False,
                "source_files_modified": False, "cases_modified": False, "source_hashes_rechecked_after_render": True,
                "selection": "all 8 fixed dev cases, all 4 subjects, both groups, all 3 gains, both methods; no quality filtering",
                "layout": {"columns": 10, "overview_rows": 8, "subject_rows": 2, "crop": False, "retouch": False,
                           "whole_image_letterbox": True}, "run": str(run), "generation_status": result["status"],
                "solver": SOLVER, "scheduler_step_calls": 0, "gain_not_au_scale": True,
                "v8_scales": {"au_scale": 1, "guidance_scale": 7.5, "actual_rho_updates": 15},
                "planned_generated_images": 48, "generation_record_count": len(records), "case_ids": overview["case_ids"],
                "shown_overview_cells": sum(c["render_status"] == "shown" for c in overview["cells"]),
                "placeholder_overview_cells": sum(c["render_status"] == "placeholder" for c in overview["cells"]),
                "inputs": {"cases": ref(run / "cases.json"), "result": ref(run / "result.json"),
                           "configuration": ref(run / "configuration.json"), "retained_v5_result": ref(V5RUN / "result.json"),
                           "retained_v7_result": ref(V7RUN / "result.json")},
                "script": ref(__file__), "readonly_render_template": ref(TEMPLATE),
                "readonly_image_layout_helper": ref(base.HELPER), "font": ref(font_path),
                "overview": {key: overview[key] for key in ("path", "sha256", "size")}, "pages": pages}
    index = output / "index.json"
    with index.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": manifest["status"], "page_count": len(pages), "columns": 10,
                      "index": ref(index), "overview": manifest["overview"],
                      "placeholder_overview_cells": manifest["placeholder_overview_cells"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

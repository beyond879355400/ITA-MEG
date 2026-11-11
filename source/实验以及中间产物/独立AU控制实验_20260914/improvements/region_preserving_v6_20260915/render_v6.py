"""Terminal-only complete v6 contact sheets, all cases/configs, no retouching."""

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
V5 = HERE.parent / "quality_diagnosis_v5_20260915"
V5RUN = V5 / "runs" / "source_init_202609151525"
HELPER = V5 / "render_diagnostic.py"
HELPER_SHA = "000412933c22a6b6bbf4b36523c97bb4c2138a58903a237064637834fc833f09"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if sha(HELPER) != HELPER_SHA:
    raise ValueError("Original v5 scientific rendering helper changed")
spec = importlib.util.spec_from_file_location("v5_readonly_scientific_layout", HELPER)
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)
need, read, ref = layout.need, layout.read_json, layout.file_ref
SUBJECTS, GROUPS, CELL, GAP, MARGIN = layout.SUBJECTS, layout.GROUPS, layout.CELL, layout.GAP, layout.MARGIN
AUS = (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)
CONFIGS = (("masked_s055_a100", .55, 1., 11),
           ("masked_s075_a100", .75, 1., 15),
           ("masked_s075_a150", .75, 1.5, 15))


def columns(include_mask_debug=False):
    result = [
        {"key": "source_real", "kind": "reference", "lines": ["源图", "原始对齐图", "仅参照 · 无去噪"]},
        {"key": "target_real", "kind": "reference", "lines": ["真实目标", "同一固定案例", "仅参照 · 未用于区域几何"]},
    ]
    for suffix, strength, steps in (("055", .55, 11), ("075", .75, 15)):
        result.append({"key": f"bidirectional_img2img_s{suffix}", "kind": "v5",
                       "strength": strength, "au_scale": 1., "expected_steps": steps,
                       "lines": ["v5 双向 · 无区域保留", f"强度 {strength:.2f} / AU×1", f"名义20 / 实际{steps}步"]})
    for name, strength, au_scale, steps in CONFIGS:
        for method, label in (("baseline", "基底"), ("bidirectional", "双向")):
            result.append({"key": f"{method}_{name}", "kind": "v6", "config": name,
                           "strength": strength, "au_scale": au_scale, "expected_steps": steps,
                           "lines": [f"v6 区域保留 · {label}", f"强度 {strength:.2f} / AU×{au_scale:g}",
                                     f"名义20 / 实际{steps}步"]})
    if include_mask_debug:
        result.append({"key": "source_overlay_debug", "kind": "debug",
                       "lines": ["源区域 overlay · DEBUG", "红色=可编辑支持域", "调试示意，不是生成图"]})
    need(len(result) == (11 if include_mask_debug else 10), "Unexpected column count")
    return result


def load_inputs(run):
    need(run.parent == HERE / "runs", "Input must be a direct v6 runs child")
    need((run / "result.json").is_file(), "No terminal result.json: do not render an active generation")
    result, document = read(run / "result.json"), read(run / "cases.json")
    need(result.get("status") in ("complete", "completed_with_failures"), "Generation is not terminal")
    need(result.get("test_data_used") is False and document.get("test_data_used") is False, "Development-only inputs required")
    need(result.get("planned") == 48, "Expected the fixed 48-image v6 diagnostic")
    for key, filename in (("cases", "cases.json"), ("configuration", "configuration.json")):
        bound = result[key]
        need(Path(bound["path"]).resolve() == run / filename and sha(run / filename) == bound["sha256"],
             f"Terminal {key} binding mismatch")
    config = read(run / "configuration.json")
    expected_config = [{"name": n, "strength": s, "au_scale": a} for n, s, a, _ in CONFIGS]
    need(config.get("configs") == expected_config, "Configuration matrix changed")
    need(config.get("seed") == 101 and config.get("nominal_steps") == 20, "Seed/step protocol changed")
    need(config.get("post_generation_pixel_blend") is False, "Unexpected post-generation retouch declaration")
    need(Path(config["v5_cases"]["path"]).resolve() == V5RUN / "cases.json", "Different v5 comparison cases")
    need(sha(V5RUN / "cases.json") == config["v5_cases"]["sha256"], "Original v5 cases changed")
    need(sha(V5RUN / "result.json") == "4e92651f47abadfefdf0be2a91600544c1142cbbd99ebab411b2bea0b22177ce", "Original v5 result changed")
    cases = document["cases"]
    need(document.get("case_count") == len(cases) == 8, "Exactly eight cases required")
    need(len({c["case_id"] for c in cases}) == 8 and all(c.get("split") == "dev" for c in cases), "Duplicate/non-dev cases")
    need(Counter((c["subject"], c["group"]) for c in cases) == Counter((s, g) for s in SUBJECTS for g in GROUPS),
         "Must contain all four subjects and both case groups")
    known_cases = {c["case_id"] for c in cases}
    allowed = {col["key"]: col for col in columns() if col["kind"] == "v6"}
    records = {}
    for record in result["records"]:
        key = record["case_id"], record["method"]
        need(key[0] in known_cases and key[1] in allowed and key not in records, "Unexpected/duplicate generation record")
        col = allowed[key[1]]
        need(record.get("split") == "dev" and record.get("seed") == 101, "Record split/seed changed")
        need(record.get("config") == col["config"] and record.get("strength") == col["strength"]
             and record.get("au_scale") == col["au_scale"], "Generated record does not match the labelled configuration")
        if record.get("status") == "complete":
            need(record.get("actual_steps") == col["expected_steps"], "Successful record has unexpected actual step count")
        records[key] = record
    v5_records = {(r["case_id"], r["method"]): r for r in read(V5RUN / "result.json")["records"]}
    return cases, records, v5_records, result


def cell_source(case, column, records, v5_records):
    kind, key = column["kind"], column["key"]
    if kind == "reference":
        source = case.get(key, {})
        return {"path": source.get("path"), "expected_sha256": source.get("sha256"),
                "generation_status": "reference", "actual_steps": None}
    if kind == "debug":
        receipt = case.get("region_mask", {}).get("receipt", {})
        path = receipt.get("path")
        if not path or not Path(path).is_file() or sha(path) != receipt.get("sha256"):
            return {"path": None, "expected_sha256": None, "generation_status": "missing_record",
                    "actual_steps": None, "error": "Missing or SHA-mismatched debug receipt"}
        doc = read(path)
        need(doc["case_id"] == case["case_id"] and doc["source_sha256"] == case["source_real"]["sha256"],
             "Debug receipt belongs to another case/source")
        source = doc["artifacts"]["source_overlay_debug"]
        return {"path": source["path"], "expected_sha256": source["sha256"], "generation_status": "reference",
                "actual_steps": None, "debug_only_not_generated": True}
    record = (records if kind == "v6" else v5_records).get((case["case_id"], key))
    if record is None:
        return {"path": None, "expected_sha256": None, "generation_status": "missing_record", "actual_steps": None}
    if kind == "v5" and record.get("status") == "complete":
        source = case.get(key, {})
        need(source.get("path") == record.get("output") and source.get("sha256") == record.get("sha256"),
             "Retained v5 image differs from v6 comparison binding")
        need(record.get("actual_steps") == column["expected_steps"], "V5 actual step count differs from header")
    return {"path": record.get("output"), "expected_sha256": record.get("sha256"),
            "generation_status": record.get("status"), "actual_steps": record.get("actual_steps"),
            "actual_timesteps": record.get("actual_timesteps"), "error": record.get("error")}


def ordered_cases(cases, subjects):
    return [next(c for c in cases if c["subject"] == subject and c["group"] == group)
            for subject in subjects for group in GROUPS]


def render_page(name, selected, records, old_records, cols, output, fonts):
    width = 2 * MARGIN + len(cols) * CELL + (len(cols) - 1) * GAP
    height = 256 + len(selected) * 398 + 120
    page = Image.new("RGB", (width, height), layout.BACKGROUND)
    draw = ImageDraw.Draw(page)
    title = "全部4主体 × 两类固定案例" if len(selected) == 8 else f"{selected[0]['subject']} × 两类固定案例"
    draw.text((MARGIN, 18), f"v6 区域保留诊断｜{title}", font=fonts[42], fill=layout.TEXT)
    draw.text((MARGIN, 74), "每主体先零编辑、后编辑；全部配置保留，无质量筛选。seed 101；开发集，不是独立最终测试。", font=fonts[25], fill=layout.MUTED)
    draw.text((MARGIN, 112), "整张原图仅等比缩放排版，不裁脸、不修图；无区域 v5 与区域 v6 共用原双向检查点。缺失/失败保留占位。", font=fonts[22], fill=layout.MUTED)
    for index, col in enumerate(cols):
        x = MARGIN + index * (CELL + GAP)
        color = "#e0e9f5" if col["kind"] == "v6" else "#f4e3cf" if col["kind"] == "debug" else "#e6e9ef"
        draw.rounded_rectangle((x, 157, x + CELL, 252), radius=8, fill=color)
        for offset, line in enumerate(col["lines"]):
            layout.centered(draw, line, x, 164 + offset * 28, fonts[20])
    cells = []
    for row, case in enumerate(selected):
        top = 267 + row * 398
        changed = [f"AU{au}:{before:g}→{after:g}" for au, before, after
                   in zip(AUS, case["source_aus_0_5"], case["target_aus_0_5"]) if before != after]
        group = "零编辑" if case["group"] == "no_edit" else "编辑"
        control = " / ".join(changed) or "AU控制无变化；v6源VAE重建为构造保证"
        draw.text((MARGIN, top), f"{case['subject']} · {group}｜{case['case_id']}｜{control}", font=fonts[22], fill=layout.TEXT)
        for index, col in enumerate(cols):
            x = MARGIN + index * (CELL + GAP)
            info = layout.paste_cell(page, draw, x, top + 37, cell_source(case, col, records, old_records), fonts)
            info.update(case_id=case["case_id"], subject=case["subject"], group=case["group"], row=row,
                        column=index, column_key=col["key"], column_kind=col["kind"], column_label=col["lines"])
            cells.append(info)
    footer = height - 100
    draw.line((MARGIN, footer, width - MARGIN, footer), fill="#cbd3df", width=2)
    draw.text((MARGIN, footer + 15), "顺序：源 / 目标 / v5双向 .55、.75 / v6 .55-AU×1、.75-AU×1、.75-AU×1.5（每组基底、双向）。", font=fonts[20], fill=layout.MUTED)
    draw.text((MARGIN, footer + 48), "区域来自源关键点与AU控制差值，非严格因果分区；零编辑保护不应归因为模块学到身份。步骤以生成记录为准。", font=fonts[20], fill=layout.MUTED)
    if any(c["kind"] == "debug" for c in cols):
        draw.text((MARGIN, footer + 79), "最后一列为源区域 debug overlay，不是生成图，不纳入生成质量比较。", font=fonts[18], fill="#8b5b1d")
    path = output / f"{name}.png"
    with path.open("xb") as handle:
        page.save(handle, format="PNG")
    return {**ref(path), "size": [width, height], "case_ids": [c["case_id"] for c in selected],
            "columns": cols, "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Terminal v6 runs child; no models loaded")
    parser.add_argument("--include-mask-debug", action="store_true", help="Add clearly labelled debug-only 11th column")
    args = parser.parse_args()
    run = args.run.resolve()
    cases, records, old_records, result = load_inputs(run)
    fonts, font_path = layout.make_fonts()
    cols = columns(args.include_mask_debug)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = HERE / "rendered" / f"{run.name}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    pages = [render_page("all_four_subjects_all_cases", ordered_cases(cases, SUBJECTS), records, old_records, cols, output, fonts)]
    pages += [render_page(f"{subject}_all_cases", ordered_cases(cases, [subject]), records, old_records, cols, output, fonts)
              for subject in SUBJECTS]
    observed = {cell["path"]: cell["actual_sha256"] for p in pages for cell in p["cells"] if cell.get("actual_sha256")}
    need(all(sha(path) == digest for path, digest in observed.items()), "An input image changed during rendering")
    overview = pages[0]
    manifest = {"kind": "v6_complete_scientific_contact_sheets", "status": "rendered_pending_visual_review",
                "created_utc": datetime.now(timezone.utc).isoformat(), "test_data_used": False,
                "models_loaded": False, "source_files_modified": False, "source_hashes_rechecked_after_render": True,
                "selection": "all 8 fixed dev cases, all 4 subjects, both groups, all 3 configs, both methods; no quality filtering",
                "layout": {"columns": len(cols), "overview_rows": 8, "subject_rows": 2,
                           "whole_image_letterbox": True, "crop": False, "retouch": False,
                           "includes_explicit_debug_column": args.include_mask_debug},
                "run": str(run), "generation_status": result["status"], "planned_generated_images": 48,
                "generation_record_count": len(records), "case_ids": overview["case_ids"],
                "shown_overview_cells": sum(c["render_status"] == "shown" for c in overview["cells"]),
                "placeholder_overview_cells": sum(c["render_status"] == "placeholder" for c in overview["cells"]),
                "inputs": {"cases": ref(run / "cases.json"), "result": ref(run / "result.json"),
                           "configuration": ref(run / "configuration.json"), "retained_v5_result": ref(V5RUN / "result.json")},
                "script": ref(__file__), "readonly_layout_helper": ref(HELPER), "font": ref(font_path),
                "overview": {k: overview[k] for k in ("path", "sha256", "size")}, "pages": pages}
    index = output / "index.json"
    with index.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": manifest["status"], "page_count": len(pages), "columns": len(cols),
                      "index": ref(index), "overview": manifest["overview"],
                      "placeholder_overview_cells": manifest["placeholder_overview_cells"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

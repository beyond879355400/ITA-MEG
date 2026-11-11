"""Render every fixed v5 development case without retouching or selecting images."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SUBJECTS = ("SN004", "SN012", "SN017", "SN018")
GROUPS = ("no_edit", "lowest_available")
STRENGTHS = (("035", ".35"), ("055", ".55"), ("075", ".75"))
BASE_COLUMNS = (
    ("source_real", "源图", "原始对齐图", "仅作参照"),
    ("target_real", "真实目标", "同一固定案例", "仅作参照"),
    ("vae_reconstruction", "VAE 源图重建", "确定性均值编码", "无扩散去噪"),
    ("baseline_text2img", "旧文本生图", "基底", "原始留存 · 名义20步"),
    ("bidirectional_text2img", "旧文本生图", "双向模块", "原始留存 · 名义20步"),
)
CELL = 280
GAP = 14
MARGIN = 32
WIDTH = CELL * 11 + GAP * 10 + MARGIN * 2
HEIGHT = 1220
BACKGROUND = "#f3f5f8"
TEXT = "#172336"
MUTED = "#526176"
ERROR = "#a12b34"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def make_fonts():
    # Use installed Chinese fonts only; never download or modify font assets.
    candidates = (
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    )
    font_path = next((path for path in candidates if path.is_file()), None)
    need(font_path is not None, "No installed Chinese font found; no fallback with missing glyphs")
    fonts = {size: ImageFont.truetype(str(font_path), size) for size in (18, 20, 22, 25, 30, 42)}
    return fonts, font_path


def fit_text(draw, text, font, width):
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


def centered(draw, text, x, y, font, fill=TEXT):
    text = fit_text(draw, text, font, CELL - 16)
    offset = (CELL - draw.textlength(text, font=font)) / 2
    draw.text((x + offset, y), text, font=font, fill=fill)


def load_inputs(run):
    need(run.parent == HERE / "runs", "Input must be a v5 runs child directory")
    result_path, cases_path = run / "result.json", run / "cases.json"
    need(result_path.is_file(), "Generation has no terminal result.json; rendering is not allowed yet")
    result, document = read_json(result_path), read_json(cases_path)
    need(result.get("status") in ("complete", "completed_with_failures"), "Generation is not terminal")
    need(result.get("test_data_used") is False and document.get("test_data_used") is False, "Development-only inputs required")
    need(result.get("planned") == 48, "Expected the fixed 48-image diagnostic")
    need(result["cases"]["sha256"] == sha256(cases_path), "cases.json differs from terminal result binding")
    need(Path(result["cases"]["path"]).resolve() == cases_path, "Terminal result belongs to another cases file")
    cases = document["cases"]
    need(document.get("case_count") == len(cases) == 8, "Exactly eight fixed development cases required")
    need(len({case["case_id"] for case in cases}) == 8, "Duplicate case id")
    need(all(case.get("split") == "dev" for case in cases), "Non-development case")
    pairs = Counter((case["subject"], case["group"]) for case in cases)
    need(pairs == Counter((subject, group) for subject in SUBJECTS for group in GROUPS), "Expected all four subjects with no_edit and lowest_available")
    allowed_methods = {f"{method}_img2img_s{suffix}" for suffix, _ in STRENGTHS for method in ("baseline", "bidirectional")}
    known_cases = {case["case_id"] for case in cases}
    records = {}
    for record in result["records"]:
        key = (record["case_id"], record["method"])
        need(key[0] in known_cases and key[1] in allowed_methods, "Unexpected case or method in terminal results")
        need(key not in records, "Duplicate generation record")
        need(record.get("seed") == 101, "Expected the unchanged seed 101")
        records[key] = record
    return cases, records, result


def columns_for(cases, records):
    columns = [dict(key=key, lines=[title, subtitle, detail], kind="reference")
               for key, title, subtitle, detail in BASE_COLUMNS]
    for suffix, strength in STRENGTHS:
        for method, title in (("baseline", "基底"), ("bidirectional", "双向模块")):
            key = f"{method}_img2img_s{suffix}"
            selected = [records.get((case["case_id"], key)) for case in cases]
            steps = [row.get("actual_steps") if row else None for row in selected]
            all_complete = all(row and row.get("status") == "complete" for row in selected)
            if all_complete and len(set(steps)) == 1 and type(steps[0]) is int:
                detail = f"名义20 / 实际{steps[0]}步"
            else:
                detail = "名义20 / 实际见格下"
            columns.append({"key": key, "kind": "generated", "strength": strength,
                            "lines": ["新源图初始化", f"{title} · 强度 {strength}", detail]})
    need(len(columns) == 11, "Expected eleven columns")
    return columns


def cell_input(case, column, records):
    if column["kind"] == "reference":
        reference = case.get(column["key"], {})
        return {"path": reference.get("path"), "expected_sha256": reference.get("sha256"),
                "generation_status": "reference", "actual_steps": None}
    record = records.get((case["case_id"], column["key"]))
    if record is None:
        return {"path": None, "expected_sha256": None, "generation_status": "missing_record", "actual_steps": None}
    return {"path": record.get("output"), "expected_sha256": record.get("sha256"),
            "generation_status": record.get("status"), "actual_steps": record.get("actual_steps"),
            "actual_timesteps": record.get("actual_timesteps"), "error": record.get("error")}


def paste_cell(page, draw, x, y, source, fonts):
    """Letterbox the complete image; never crop, rotate, enhance or alter a source file."""
    info = dict(source)
    draw.rectangle((x, y, x + CELL, y + CELL), fill="white", outline="#cbd3df", width=2)
    label = None
    if source["generation_status"] not in ("reference", "complete"):
        label = "缺失记录" if source["generation_status"] == "missing_record" else "生成失败"
    elif not source.get("path") or not Path(source["path"]).is_file():
        label = "图像文件缺失"
    elif not source.get("expected_sha256"):
        label = "缺少原图 SHA 绑定"
    else:
        path = Path(source["path"])
        info["actual_sha256"] = sha256(path)
        if info["actual_sha256"] != source["expected_sha256"]:
            label = "原图 SHA 校验失败"
        else:
            try:
                with Image.open(path) as original:
                    original.load()
                    info["source_size"] = list(original.size)
                    info["source_mode"] = original.mode
                    scale = min(CELL / original.width, CELL / original.height)
                    size = (max(1, round(original.width * scale)), max(1, round(original.height * scale)))
                    # RGB conversion is only for the new contact sheet. The source is read-only.
                    preview = original.convert("RGBA").resize(size, Image.Resampling.LANCZOS)
                    px, py = x + (CELL - size[0]) // 2, y + (CELL - size[1]) // 2
                    page.paste(preview, (px, py), preview)
                    info["display_box"] = [px, py, size[0], size[1]]
                    info["display_scale"] = scale
                    info["render_status"] = "shown"
            except (OSError, ValueError) as error:
                label = "图像读取失败"
                info["render_error"] = str(error)
    if label:
        centered(draw, label, x, y + CELL // 2 - 28, fonts[22], ERROR)
        centered(draw, "保留此案例与列，不补图", x, y + CELL // 2 + 10, fonts[18], MUTED)
        info["render_status"] = "placeholder"
        info["placeholder_reason"] = label
    if source["generation_status"] == "reference":
        detail = "原始文件 · 仅等比排版"
    else:
        actual = source.get("actual_steps")
        detail = f"实际去噪 {actual} 步" if type(actual) is int else "实际去噪未记录"
        detail += " · 成功" if source["generation_status"] == "complete" else " · 非成功"
    centered(draw, detail, x, y + CELL + 14, fonts[18], MUTED)
    digest = info.get("actual_sha256")
    tail = f"SHA {digest[:12]}" if digest else (label or "无可验证文件")
    centered(draw, tail, x, y + CELL + 41, fonts[18], MUTED if not label else ERROR)
    return info


def render_subject(subject, cases, records, output, fonts):
    subject_cases = [next(case for case in cases if case["subject"] == subject and case["group"] == group)
                     for group in GROUPS]
    columns = columns_for(subject_cases, records)
    page = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(page)
    draw.text((MARGIN, 24), f"{subject}｜源图初始化诊断：全部固定开发案例", font=fonts[42], fill=TEXT)
    draw.text((MARGIN, 88), "第一行：零编辑；第二行：最低可用编辑。固定 seed 101；开发集，非新最终测试。", font=fonts[25], fill=MUTED)
    draw.text((MARGIN, 128), "只排版原始文件：不修图、不裁脸、不按质量选择；整图等比缩放并留白。新强度同时改变初始噪声与去噪深度。", font=fonts[22], fill=MUTED)
    for index, column in enumerate(columns):
        x = MARGIN + index * (CELL + GAP)
        draw.rounded_rectangle((x, 179, x + CELL, 281), radius=8,
                               fill="#e0e9f5" if column["kind"] == "generated" else "#e6e9ef")
        for offset, line in enumerate(column["lines"]):
            centered(draw, line, x, 188 + offset * 28, fonts[20], TEXT)
    cells = []
    for row_index, case in enumerate(subject_cases):
        row_top = 308 + row_index * 404
        label = "零编辑（no_edit）" if case["group"] == "no_edit" else "编辑（lowest_available）"
        draw.text((MARGIN, row_top), f"{label}｜{case['case_id']}", font=fonts[25], fill=TEXT)
        for column_index, column in enumerate(columns):
            x = MARGIN + column_index * (CELL + GAP)
            info = paste_cell(page, draw, x, row_top + 46, cell_input(case, column, records), fonts)
            info.update(case_id=case["case_id"], subject=subject, group=case["group"],
                        row=row_index, column=column_index, column_key=column["key"], column_label=column["lines"])
            cells.append(info)
    draw.line((MARGIN, 1128, WIDTH - MARGIN, 1128), fill="#cbd3df", width=2)
    draw.text((MARGIN, 1145), "阅读顺序：源 / 真实目标 / VAE / 旧文本生图基底与双向 / 新 .35 基底与双向 / 新 .55 基底与双向 / 新 .75 基底与双向", font=fonts[20], fill=MUTED)
    draw.text((MARGIN, 1180), "步数以 generation record 为准；缺失或校验失败显示占位格。全部路径、原图 SHA、页面 SHA 及显示边界见 index.json。", font=fonts[18], fill=MUTED)
    destination = output / f"{subject}_all_fixed_development_cases.png"
    with destination.open("xb") as handle:
        page.save(handle, format="PNG")
    return {**file_ref(destination), "subject": subject, "size": [WIDTH, HEIGHT],
            "case_ids": [case["case_id"] for case in subject_cases], "columns": columns, "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed v5 run; no models are loaded")
    args = parser.parse_args()
    run = args.run.resolve()
    cases, records, result = load_inputs(run)
    fonts, font_path = make_fonts()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = HERE / "rendered" / f"{run.name}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    pages = [render_subject(subject, cases, records, output, fonts) for subject in SUBJECTS]
    manifest = {
        "kind": "v5_fixed_development_contact_sheets", "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_used": False, "models_loaded": False, "source_files_modified": False,
        "selection": "all 8 fixed cases; 4 subjects; no quality filtering; explicit missing/failed cells",
        "layout": "11 columns; no_edit then lowest_available; aspect-preserving whole-image letterbox; no crop or retouch",
        "run": str(run), "generation_status": result["status"], "planned_generated_images": 48,
        "generation_record_count": len(records), "case_ids": [case["case_id"] for case in cases],
        "inputs": {"cases": file_ref(run / "cases.json"), "result": file_ref(run / "result.json")},
        "script": file_ref(__file__), "font": file_ref(font_path), "pages": pages,
        "shown_cells": sum(cell["render_status"] == "shown" for page in pages for cell in page["cells"]),
        "placeholder_cells": sum(cell["render_status"] == "placeholder" for page in pages for cell in page["cells"]),
    }
    index = output / "index.json"
    with index.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "rendered", "page_count": len(pages), "case_count": len(cases),
                      "index": file_ref(index), "pages": [{key: page[key] for key in ("subject", "path", "sha256", "size")} for page in pages],
                      "placeholder_cells": manifest["placeholder_cells"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

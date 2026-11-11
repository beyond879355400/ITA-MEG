"""v7 terminal contact-sheet CLI: pinned v6 scientific layout, explicit metadata-only adaptation."""

import ast
import copy
import hashlib
from pathlib import Path
import sys
from types import ModuleType

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "region_preserving_v6_20260915" / "render_v6.py"
TEMPLATE_SHA256 = "107143fe267144d9c8ac5b500faa5395f9179008038855ccdc0367133ab690d0"
OLD_CONFIGS = (("masked_s055_a100", .55, 1., 11),
               ("masked_s075_a100", .75, 1., 15),
               ("masked_s075_a150", .75, 1.5, 15))
CONFIGS = (("smooth_s075_a100", .75, 1., 15),
           ("smooth_s075_a150", .75, 1.5, 15),
           ("smooth_s075_a200", .75, 2., 15))

# Only the protocol matrix, schema kind and displayed prose change. Each exact
# source constant is occurrence-checked against the SHA-frozen v6 template.
TEXT_CHANGES = {
    "Terminal-only complete v6 contact sheets, all cases/configs, no retouching.":
        "Terminal-only complete v7 contact sheets, all cases/configs, no retouching.",
    "v6": "v7",
    "v6 区域保留 · ": "v7 平滑区域 · ",
    "Input must be a direct v6 runs child": "Input must be a direct v7 runs child",
    "Expected the fixed 48-image v6 diagnostic": "Expected the fixed 48-image v7 diagnostic",
    "Retained v5 image differs from v6 comparison binding": "Retained v5 image differs from v7 comparison binding",
    "v6 区域保留诊断｜": "v7 平滑区域保留诊断｜",
    "整张原图仅等比缩放排版，不裁脸、不修图；无区域 v5 与区域 v6 共用原双向检查点。缺失/失败保留占位。":
        "整张原图仅等比缩放排版，不裁脸、不修图；无区域 v5 与平滑区域 v7 共用原双向检查点。缺失/失败保留占位。",
    "AU控制无变化；v6源VAE重建为构造保证": "AU控制无变化；v7源VAE重建为构造保证",
    "顺序：源 / 目标 / v5双向 .55、.75 / v6 .55-AU×1、.75-AU×1、.75-AU×1.5（每组基底、双向）。":
        "顺序：源 / 目标 / v5双向 .55、.75 / v7强度均.75，AU×1、AU×1.5、AU×2（每组基底、双向；均15步）。",
    "区域来自源关键点与AU控制差值，非严格因果分区；零编辑保护不应归因为模块学到身份。步骤以生成记录为准。":
        "区域由源关键点与AU控制差值确定，再固定maxpool7+Gaussian平滑；非严格因果分区。零编辑保护是构造保证。",
    "Terminal v6 runs child; no models loaded": "Terminal v7 runs child; no models loaded",
    "v6_complete_scientific_contact_sheets": "v7_complete_scientific_contact_sheets",
}
EXPECTED_COUNTS = {text: 1 for text in TEXT_CHANGES}
EXPECTED_COUNTS["v6"] = 4


class MetadataAdapter(ast.NodeTransformer):
    def __init__(self):
        self.counts = {key: 0 for key in TEXT_CHANGES}
        self.config_count = 0
        self.manifest_count = 0

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in TEXT_CHANGES:
            self.counts[node.value] += 1
            return ast.copy_location(ast.Constant(TEXT_CHANGES[node.value]), node)
        return node

    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id == "CONFIGS" for target in node.targets):
            if ast.literal_eval(node.value) != OLD_CONFIGS:
                raise ValueError("Unexpected template configuration matrix")
            self.config_count += 1
            node.value = ast.parse(repr(CONFIGS), mode="eval").body
        node = self.generic_visit(node)
        if any(isinstance(target, ast.Name) and target.id == "manifest" for target in node.targets):
            if not isinstance(node.value, ast.Dict):
                raise ValueError("Unexpected template manifest structure")
            self.manifest_count += 1
            additions = {
                "render_template": {"path": str(TEMPLATE), "sha256": TEMPLATE_SHA256},
                "metadata_adaptation": {
                    "version": "v7_explicit_metadata_only",
                    "changes": ["three smooth_s075 configs at AUscale1/1.5/2, all15steps",
                                "v7 schema kind and accurate visible labels",
                                "provenance metadata for the immutable rendering template"],
                    "image_loading_resizing_pasting_changed": False,
                    "v6_intermediate_column_included": False,
                },
            }
            for key, value in additions.items():
                node.value.keys.append(ast.Constant(key))
                node.value.values.append(ast.parse(repr(value), mode="eval").body)
        return node


def load_renderer():
    source = TEMPLATE.read_bytes()
    if hashlib.sha256(source).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("Original v6 scientific renderer changed")
    adapter = MetadataAdapter()
    tree = adapter.visit(ast.parse(source, filename=str(TEMPLATE)))
    if adapter.counts != EXPECTED_COUNTS or adapter.config_count != 1 or adapter.manifest_count != 1:
        raise ValueError(f"Metadata-only template adaptation mismatch: {adapter.counts}")
    module = ModuleType("v7_scientific_layout_from_pinned_v6")
    # __file__ intentionally identifies this v7 entry point: HERE, output folder
    # and manifest.script SHA belong to v7, not the immutable v6 template.
    module.__file__ = str(Path(__file__).resolve())
    exec(compile(ast.fix_missing_locations(tree), str(TEMPLATE), "exec"), module.__dict__)
    module.metadata_adaptation_counts = copy.deepcopy(adapter.counts)
    return module


RENDERER = load_renderer()


if __name__ == "__main__":
    RENDERER.main()

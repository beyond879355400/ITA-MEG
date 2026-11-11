"""Versioned metadata adapter over the unchanged v6 inference kernel."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
KERNEL=HERE.parent/'region_preserving_v6_20260915/run_region_diagnostic.py'
EXPECTED='85236dfc8803fb66d8f2440a240475a10c59f189ab3df059a44389f13db1fcec'
raw=KERNEL.read_bytes()
if hashlib.sha256(raw).hexdigest()!=EXPECTED: raise RuntimeError('v6 kernel changed')
source=raw.decode()
changes={
"CONFIGS = [('masked_s055_a100', .55, 1.), ('masked_s075_a100', .75, 1.), ('masked_s075_a150', .75, 1.5)]":
"CONFIGS = [('smooth_s075_a100', .75, 1.), ('smooth_s075_a150', .75, 1.5), ('smooth_s075_a200', .75, 2.)]",
"'v6_development_region_preserving_cases'":"'v7_development_region_preserving_cases'",
"'v6_region_preserving_diagnostic'":"'v7_region_preserving_diagnostic'",
"'v6_complete'":"'v7_complete'"}
for old,new in changes.items():
    expected_count = 2 if old == "'v6_region_preserving_diagnostic'" else 1
    if source.count(old)!=expected_count: raise RuntimeError('Ambiguous metadata adaptation: '+old)
    source=source.replace(old,new)
# __file__ pins this version's output scope and source-only region_mask import.
exec(compile(source,str(KERNEL),'exec'),globals())

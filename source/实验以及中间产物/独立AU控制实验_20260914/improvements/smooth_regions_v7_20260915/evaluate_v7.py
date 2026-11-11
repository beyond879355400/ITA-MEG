"""Same frozen metric calculations; v7 method/config metadata only."""
import hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
KERNEL=HERE.parent/'region_preserving_v6_20260915/evaluate_v6.py'
raw=KERNEL.read_bytes()
if hashlib.sha256(raw).hexdigest()!='e72173eb31f4101eb7e5b586e59cea5c3aae68ece44455c77fa211814e65ad67': raise RuntimeError('v6 evaluator changed')
source=raw.decode()
changes={
"'masked_s055_a100': {'strength': .55, 'au_scale': 1.0}":"'smooth_s075_a100': {'strength': .75, 'au_scale': 1.0}",
"'masked_s075_a100': {'strength': .75, 'au_scale': 1.0}":"'smooth_s075_a150': {'strength': .75, 'au_scale': 1.5}",
"'masked_s075_a150': {'strength': .75, 'au_scale': 1.5}":"'smooth_s075_a200': {'strength': .75, 'au_scale': 2.0}",
"'v6_development_region_preserving_cases'":"'v7_development_region_preserving_cases'",
"'v6_region_preserving_diagnostic'":"'v7_region_preserving_diagnostic'"}
for old,new in changes.items():
    if source.count(old)!=1: raise RuntimeError('Ambiguous evaluation metadata change')
    source=source.replace(old,new)
exec(compile(source,str(KERNEL),'exec'),globals())

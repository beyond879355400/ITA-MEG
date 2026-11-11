"""Finite read-back of frozen tiny manifest; never accepts transient EOF."""
import hashlib,json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
source=HERE.parent.parent/'preservation/bundle_v2_20260916/inputs/smoke_dev4.json'
expected='e4aecce6e6e0206f0788bfa24a6eaa90bd3dc2e2f8c36a2245575a35f5db1d09'
attempts=[];data=None
for attempt in range(3):
    raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    attempts.append({'attempt':attempt+1,'bytes':len(raw),'sha256':digest,'flags_after':source.stat().st_flags})
    if len(raw)==6680 and digest==expected:data=raw;break
    time.sleep(attempt+1)
if data is None:raise RuntimeError('Frozen manifest remains absent/mismatched: '+json.dumps(attempts))
folder=HERE/'verified_inputs';folder.mkdir(exist_ok=True)
out=folder/'smoke_dev4.json'
with out.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
assert out.read_bytes()==data
receipt={'source':str(source),'copy':str(out),'bytes':len(data),'sha256':expected,'attempts':attempts,'original_modified':False}
with (folder/'manifest_receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps({**receipt,'case_ids':[r['case_id'] for r in json.loads(data)['cases']]},indent=2))

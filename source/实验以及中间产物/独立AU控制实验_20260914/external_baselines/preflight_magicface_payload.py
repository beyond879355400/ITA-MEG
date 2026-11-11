#!/usr/bin/env python3
"""Verify actual CPU shapes and the manual delta; never load or call a model."""
import argparse
import json
from pathlib import Path

from magicface_contract import cpu_source_payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    document = json.loads(args.manifest.read_text())
    assert not document['test_used'] and document['input_group'] == 'source_plus_manual_au'
    rows = []
    for case in document['cases']:
        payload = cpu_source_payload(case)
        assert tuple(payload['source'].shape) == (1, 3, 512, 512)
        assert tuple(payload['au'].shape) == (1, 12)
        assert payload['au'][0].tolist() == case['au_delta_native']
        rows.append({'case_id': case['case_id'], 'source_shape': list(payload['source'].shape),
                     'au_shape': list(payload['au'].shape), 'au_native': payload['au'][0].tolist(),
                     'device': str(payload['source'].device), 'source_min': float(payload['source'].min()),
                     'source_max': float(payload['source'].max()), 'background_prepared': False, 'model_called': False})
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump({'cases': rows, 'test_used': False, 'generation_run': False}, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'cpu_payloads_valid': len(rows), 'generation_run': False, 'test_used': False}))


if __name__ == '__main__':
    main()

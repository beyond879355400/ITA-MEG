#!/usr/bin/env python3
"""Record current exact resident resources after restoration; no model execution."""
import argparse
from pathlib import Path

from prepare_dev_contracts import magic_resources, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = magic_resources()
    result.update(all_required_files_resident=not result['missing_resident_components'],
                  model_loading_or_generation_verified=False,
                  note='Component paths are explicit; no same-shape model substitution. Source preprocessing and generation remain a later step.')
    save(args.output, result)
    print({'all_required_files_resident': result['all_required_files_resident'], 'missing': result['missing_resident_components']})


if __name__ == '__main__':
    main()

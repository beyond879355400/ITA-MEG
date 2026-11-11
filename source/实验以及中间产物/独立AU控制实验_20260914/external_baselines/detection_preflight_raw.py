#!/usr/bin/env python3
"""Exact detector-preflight logic, but restricted to the authorized raw-v2 inputs."""
import hashlib
from pathlib import Path


def main():
    source_path = Path(__file__).with_name('detection_preflight.py')
    source = source_path.read_text()
    replacements = {
        "assert not document['test_used']": "assert not document['test_used'] and document['schema_version'] == 2 and document['source_space'] == 'original_raw_1024x768'",
        "assert bgr.shape == (512, 512, 3)": "assert bgr.shape == (768, 1024, 3)",
        "'image_preprocessing': 'original canonical512, no padding'": "'image_preprocessing': 'original raw1024x768, before author-native crop, no padding'",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError('Original detector preflight changed; do not apply an uncontrolled rewrite')
        source = source.replace(old, new)
    namespace = {'__name__': '__raw_preflight__', '__file__': str(Path(__file__).resolve())}
    exec(compile(source, str(source_path), 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()

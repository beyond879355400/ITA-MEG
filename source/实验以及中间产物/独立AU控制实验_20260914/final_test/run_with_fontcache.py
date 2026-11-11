"""Prime the already verified matplotlib font cache; run the unchanged entry."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'development_evaluation/evaluation_policy_v2/buffalo_dev19/cache/matplotlib/fontlist-v390.json'
SHA = 'e00ac7e456e6c76782f9b334fdd347d7a50d8d8ce85e929c68657d5310b12352'

if __name__ == '__main__':
    output, entry, *args = sys.argv[1:]
    output, entry = Path(output).resolve(), Path(entry).resolve()
    assert output.is_relative_to(HERE) and entry.is_relative_to(HERE)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(SOURCE, output / SOURCE.name)
    os.environ.update(MPLCONFIGDIR=str(output), PYTHONDONTWRITEBYTECODE='1')
    import matplotlib.font_manager as fm
    assert fm.fontManager.__version__ == 390
    with (output / 'receipt.json').open('x') as f:
        json.dump({'source': str(SOURCE), 'sha256': SHA, 'entry': str(entry),
                   'entry_sha256': hashlib.sha256(entry.read_bytes()).hexdigest(),
                   'algorithm_changed': False}, f, indent=2)
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry), *args]
    runpy.run_path(str(entry), run_name='__main__')

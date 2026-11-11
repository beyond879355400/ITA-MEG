"""Retry interrupted CPU evaluation using a same-version font metadata cache."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from step2000_gate import HERE,require,sha,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);args=p.parse_args();job=HERE/args.group;old=job/'buffalo'
    require(job.is_dir() and not(job/'buffalo/results.json').exists(),'Only incomplete evaluation retry allowed')
    reached_model_initialization=(old/'runtime.log').exists()
    archived=job/'buffalo_fontscan_interrupted';require(not archived.exists(),'Import attempt already archived');old.rename(archived)
    source=HERE/'buffalo_dev19/cache/matplotlib/fontlist-v390.json';cache=job/'font_cache_reuse';cache.mkdir(exist_ok=False)
    target=cache/source.name
    with source.open('rb') as inp,target.open('xb') as out:shutil.copyfileobj(inp,out)
    require(sha(source)==sha(target),'Font metadata copy mismatch')
    save(job/'fontcache_retry.json',{'cause':'process sample caught ft2font read; between sampling and SIGINT it advanced into detector inference, then stopped before any complete results file',
         'source_fontcache':str(source),'source_sha256':sha(source),'copied_fontcache':str(target),'original_attempt_preserved':str(archived),
         'original_attempt_reached_model_initialization':reached_model_initialization,'rerun_scope':'all95 fixed logical inputs; no partial result or selected case reused',
         'model_parameters_changed':False,'model_core_changed':False,'script_sha256':sha(__file__)})
    os.environ['MPLCONFIGDIR']=str(cache);os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
    import matplotlib.font_manager
    require(matplotlib.font_manager.FontManager.__version__==390,'Font metadata version mismatch')
    print(json.dumps({'font_cache_preloaded':True,'version':390,'no_font_rendering_used_for_evaluation':True}),flush=True)
    import step2000_cpu_eval
    sys.argv=[str(Path(step2000_cpu_eval.__file__)),'buffalo','--group',args.group];step2000_cpu_eval.main()

if __name__=='__main__':main()

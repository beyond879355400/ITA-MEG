"""Small authorized isolated build from the exact public PyPI0.7.3 source."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request
import prepare_overlay as p

HERE=p.HERE
TOOLS=HERE/'build_tools_v1'
TARGET=HERE/'insightface_built_v1'
URL='https://files.pythonhosted.org/packages/0b/8d/0f4af90999ca96cf8cb846eb5ae27c5ef5b390f9c090dd19e4fa76364c13/insightface-0.7.3.tar.gz'
SHA='f191f719612ebb37018f41936814500544cd0f86e6fcd676c023f354c668ddf7'
SIZE=439490
UV=str(Path.home()/'.local/bin/uv')


def main():
    p.guard();p.need(not TOOLS.exists()and not TARGET.exists(),'Append-only build directories required')
    downloads=HERE/'overlay_source_downloads';downloads.mkdir(exist_ok=False)
    archive=downloads/'insightface-0.7.3.tar.gz'
    with urllib.request.urlopen(URL,timeout=30)as r:body=r.read(SIZE+1)
    p.need(len(body)==SIZE and hashlib.sha256(body).hexdigest()==SHA,'Pinned PyPI source mismatch')
    with archive.open('xb')as f:f.write(body)
    with tarfile.open(archive,'r:gz')as tar:
        setup=tar.extractfile('insightface-0.7.3/setup.py').read().decode()
    p.save(HERE/'insightface_build_source.json',{'source':URL,'sha256':SHA,'bytes':SIZE,'setup_py':setup,
           'runtime_algorithm_source':'official PyPI insightface0.7.3, no patches','core_unchanged':True})
    env={**os.environ,'TMPDIR':str(p.TMP),'PYTHONDONTWRITEBYTECODE':'1','NO_ALBUMENTATIONS_UPDATE':'1',
         'MAX_JOBS':'2','CMAKE_BUILD_PARALLEL_LEVEL':'2'}
    cmd=[UV,'pip','install','--python',str(p.CORE/'bin/python'),'--target',str(TOOLS),'--no-deps',
         '--only-binary',':all:','--default-index','https://pypi.org/simple','--no-config','--no-cache','--link-mode','copy',
         'setuptools==70.3.0','wheel==0.45.1','Cython==3.2.4']
    with (HERE/'insightface_build_tools.log').open('x')as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    p.need(r.returncode==0,'Build tools failed');p.guard()
    env.update(PYTHONPATH=os.pathsep.join([str(TOOLS),str(p.SITE)]),UV_OFFLINE='1')
    cmd2=[UV,'pip','install','--python',str(p.CORE/'bin/python'),'--target',str(TARGET),'--no-deps',
          '--no-build-isolation','--no-config','--no-cache','--link-mode','copy',str(archive)]
    with (HERE/'insightface_build.log').open('x')as log:r2=subprocess.run(cmd2,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    p.save(HERE/'insightface_build_receipt.json',{'status':'complete'if r2.returncode==0 else 'failed',
           'source_sha256':SHA,'build_tools_command':cmd,'build_command':cmd2,'returncode':r2.returncode,
           'no_runtime_source_patch':True,'core_unchanged':True,'no_models_run':True,
           'build_tools_bytes':p.total(TOOLS),'target_bytes':p.total(TARGET)if TARGET.exists()else 0,
           'overlay_bytes':p.total(p.SITE),'environment_pythonpath':env['PYTHONPATH']})
    p.need(r2.returncode==0,'InsightFace build failed; see preserved log')
    p.need(p.total(TOOLS)+p.total(TARGET)+p.total(p.SITE)<p.CAP,'Combined overlay/build cost exceeded1GiB')
    print(json.dumps({'status':'built_import_pending','core_unchanged':True,'bytes':p.total(TOOLS)+p.total(TARGET)+p.total(p.SITE)}))


if __name__=='__main__':main()

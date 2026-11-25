"""Record the running hardware and package versions of the two actual numerical environments."""
import importlib.metadata as md
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def env():
    deps={}
    for p in ['torch','torchvision','transformers','peft','numpy','scipy','pyarrow','Pillow']:
        try:deps[p]=md.version(p)
        except md.PackageNotFoundError:pass
    return dict(python=platform.python_version(),interpreter=sys.executable,packages=deps)

def main():
    if len(sys.argv)>1:
        print(json.dumps(env()));return
    legacy='/private/AGSE_E1_20261110/legacy/env/bin/python'
    query=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    cpu=json.loads(subprocess.check_output([legacy,str(Path(__file__).resolve()),'cpu'],text=True))
    info=dict(gpu=query,agent=env(),frozen_reader=cpu,working_directory=str(ROOT),private_mount=subprocess.check_output(['findmnt','-T',str(ROOT),'-n','-o','TARGET,FSTYPE'],text=True).strip())
    (ROOT/'inputs/environment.json').write_text(json.dumps(info,indent=2)+'\n')
if __name__=='__main__':main()

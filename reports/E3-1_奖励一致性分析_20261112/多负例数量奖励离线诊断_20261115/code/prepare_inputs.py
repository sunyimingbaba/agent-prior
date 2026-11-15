import argparse
from pathlib import Path
import shutil
from common import ROOT,read,dump,sha
p=argparse.ArgumentParser();p.add_argument('--parent',type=Path);a=p.parse_args();plan=read(ROOT/'plan.json');parent=a.parent or ROOT.parent/plan['parent_directory'];copied=[]
for n,z in plan['input_files'].items():
    src=parent/n;assert src.stat().st_size==z['bytes'] and sha(src)==z['sha256'],n
    if n.endswith('.tar.gz'):continue
    dst=ROOT/'source_data'/n
    if dst.exists():assert sha(dst)==z['sha256'],n
    else:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    copied.append(n)
assert read(parent/'本机独立核验.json')['ok'] and read(parent/'备份校验.json')['local_copy_verification']=='passed'
dump(ROOT/'input_verification.json',{'ok':True,'source_files_hashed':len(plan['input_files']),'source_files_copied':len(copied),'copied_names':copied,'parent_archive_sha256':plan['parent_evidence_sha256'],'plan_sha256':sha(ROOT/'plan.json')})
print('FROZEN_INPUTS_VERIFIED',len(copied))

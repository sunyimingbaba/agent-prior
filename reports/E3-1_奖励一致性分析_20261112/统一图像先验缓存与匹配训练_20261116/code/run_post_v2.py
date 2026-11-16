"""CPU postprocessing queue attached to the fixed GPU pipeline."""
from pathlib import Path
import subprocess
import sys
import time
from stage_io import read, dump, sha
r=Path(__file__).resolve().parent.parent
for n,h in read(r/'post_code_hashes_v2.json').items():
    assert sha(r/n)==h,n
while not (r/'matched_pipeline_result.json').exists():
    time.sleep(10)
assert read(r/'matched_pipeline_result.json')['status']=='complete'
steps=[]
commands=[['analyze_results.py'],['verify_results_v2.py','--root',str(r),'--out',str(r/'server_verification.json')],['audit_originals.py']]
for command in commands:
    p=subprocess.run([sys.executable,'-u',str(r/'code'/command[0]),*command[1:]])
    steps.append({'script':command[0],'exit_code':p.returncode})
    if p.returncode:
        dump(r/'post_result.json',{'status':'failed','steps':steps})
        sys.exit(p.returncode)
dump(r/'post_result.json',{'status':'complete','steps':steps})
print('MATCHED_READOUT_POSTPROCESSING_COMPLETE',flush=True)

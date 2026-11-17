"""Package the completed chain, with logs/status outside the archived root."""
from pathlib import Path
import subprocess,sys,time,json
r=Path(__file__).resolve().parent.parent
while not (r/'resume_job_result_20261117.json').exists():time.sleep(5)
assert json.loads((r/'resume_job_result_20261117.json').read_text())['status']=='complete'
name='protocol_state_preliminary_20261117.tar.gz'
with (r.parent/(name+'.pack.log')).open('xb') as f:p=subprocess.run([sys.executable,'-u',str(r/'code/package_evidence_20261117.py'),'--name',name],cwd=r,stdout=f,stderr=subprocess.STDOUT)
(r.parent/(name+'.pack_status.json')).write_text(json.dumps({'exit_code':p.returncode})+'\n')
sys.exit(p.returncode)

import datetime,json,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent.parent;start=time.monotonic();phases=[]
commands=[['test_contract.py'],['preflight.py'],['build_quantity_features.py'],['verify_historical_anchors.py'],['generate_quantity.py','--mode','smoke']]
for args in commands:
 tick=time.monotonic();p=subprocess.run([sys.executable,str(r/'code'/args[0]),*args[1:]],cwd=r)
 phases.append({'args':args,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-tick})
 if p.returncode:break
j={'status':'complete' if len(phases)==len(commands) and phases[-1]['exit_code']==0 else 'failed','phases':phases,'elapsed_seconds':time.monotonic()-start,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'gate_result.json').write_text(json.dumps(j,indent=2)+'\n')
if j['status']!='complete':sys.exit(1)

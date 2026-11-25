"""Use recorded training candidates (no optimizer) to verify the online policy worker before any promotion."""
from pathlib import Path
import sys,subprocess,base64,json,os
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'source_transfer_2h_20261125/code'))
from remote_common import read,dump,sha
from runtime import BASE as LEGACY, OLD
from orders import calibrate,score
def main():
    roster=read(BASE/'source_transfer_2h_20261125/inputs/training_roster.json')['rows'];samples=[]
    for r in roster:
        for prefix in ('attribute_questions_20261125','attribute_questions_repeat_20261125'):
            q=BASE/prefix/'records/a'/(r['image']+'.pt')
            if q.is_file():samples.append((r,q));break
    p=subprocess.Popen([str(LEGACY/'env/bin/python'),'-u',str(ROOT/'code/policy_worker.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    out=[]
    try:
        assert p.stdout.readline().strip()=='READY'
        for r,f in samples[:3]:
            y=read(f.with_suffix('.json'));assert y['states']==sha(f) and y['rgb']==r['rgb'];hs=torch.load(f,weights_only=True)['scene']
            req=dict(image=r['image'],data=base64.b64encode(hs.float().numpy().tobytes()).decode())
            p.stdin.write(json.dumps(req)+'\n');p.stdin.flush();z=json.loads(p.stdout.readline())
            old=y['groups']['scene']['scores'];errors={k:max(abs(x[k]-v) for x,v in zip(old,z[k])) for k in ('rank','native','A','keep')};assert errors['rank']==0 and errors['keep']<1e-12 and max(errors['native'],errors['A'])<2e-5,errors
            kap=read(OLD/'inputs/continuous_lock.json')['kappa'];reward,_=score([{k:z[k][i] for k in ('rank','native','A','keep')} for i in range(16)],kap)
            for a in ('R2','R3'):assert np.array_equal(reward[a],y['groups']['scene']['retrieval'][a])
            assert len(z['order'])==16 and all(np.isfinite(z['order']))
            out.append(dict(image=r['image'],record=sha(f),max_errors=errors,distinct_new=len(set(z['order']))))
    finally:p.stdin.close();p.wait()
    assert len(out)==3 and p.returncode==0;dump(ROOT/'results/worker_check.json',dict(inputs=out,worker=sha(ROOT/'code/policy_worker.py'),model=sha(ROOT/'inputs/multi_nomination.json')))
if __name__=='__main__':main()

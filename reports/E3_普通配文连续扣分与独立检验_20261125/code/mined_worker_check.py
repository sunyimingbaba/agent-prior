"""Verify the isolated warm-start training bank against stored first-caption feature projections."""
import os,sys,subprocess,base64,json,hashlib
from fractions import Fraction
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'source_transfer_2h_20261125/code'))
from remote_common import read,dump,sha
from runtime import BASE as LEGACY

def main():
    check=[];b=torch.load(ROOT/'inputs/mined_training_bank.pt',weights_only=True)
    odd=np.array([int(hashlib.sha256(','.join(map(str,x)).encode()).hexdigest()[-1],16)%2 for x in b['native_tokens'].tolist()],bool)
    for s,group in [('a','selected'),('b','uniform')]:
        rows=read(ROOT/'inputs/mined_roster.json')[s+'_'+group][:2]
        worker=subprocess.Popen([str(LEGACY/'env/bin/python'),'-u',str(ROOT/'code/mined_worker.py'),s,group],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        try:
            assert worker.stdout.readline().strip()=='READY'
            for r in rows:
                n=r['image'];base=ROOT/'records/real'/s;raw=base/'raw'/(n+'.pt')
                h=torch.load(raw,weights_only=True);feat=torch.load(base/'features'/(n+'.pt'),weights_only=True);assert feat['source']==sha(raw)
                worker.stdin.write(json.dumps(dict(image=n,data=base64.b64encode(h['hidden'].float().numpy().tobytes()).decode()))+'\n');worker.stdin.flush()
                out=json.loads(worker.stdout.readline())
                v=feat['visual'].double().numpy();ns=v@b['native'].double().numpy().T;hs=v@b['hard'].double().numpy().T;ix=b['maps'][n];ps=ix['native']
                ok=odd.copy();ok[ps]=False;alt=sorted(set(ps)-{ps[0]});rr=[]
                for scores in ns:
                    vals=[]
                    for x in scores[alt]:
                        first=1+int((scores[ok]>x).sum());count=1+int((scores[ok]==x).sum())
                        vals.append(sum(Fraction(1,i) for i in range(first,first+count))/count)
                    rr.append(vals)
                assert rr==[[Fraction(*x) for x in row] for row in out['rr']]
                assert sum((hs[:,ix[k][0]]>hs[:,ix[k][1]]).astype(int) for k in 'AB').tolist()==out['detail']
                check.append(dict(image=n,seed=s,groups=17,captions=len(alt),raw=sha(raw)))
        finally:worker.stdin.close();worker.wait()
        assert worker.returncode==0
    dump(ROOT/'results/mined_worker_check.json',dict(groups=check,worker=sha(ROOT/'code/mined_worker.py')))
if __name__=='__main__':main()

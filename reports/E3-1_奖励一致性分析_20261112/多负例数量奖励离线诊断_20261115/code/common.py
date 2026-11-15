import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,j):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def local_seed(label,base=2026111556):return (base+int(hashlib.sha256(label.encode()).hexdigest()[:8],16))%2**32
def interval(vals,counts,label,macro=False):
    vals=np.asarray(vals,dtype=np.float64);counts=np.asarray(counts);rng=np.random.default_rng(local_seed(label));boot=np.zeros(10000)
    if macro:
        point=0.
        for c in [1,2,3,4]:
            v=vals[counts==c];assert len(v)
            point+=float(v.mean())/4;draw=rng.integers(0,len(v),size=(10000,len(v)));boot+=v[draw].mean(axis=1)/4
    else:
        point=float(vals.mean());draw=rng.integers(0,len(vals),size=(10000,len(vals)));boot=vals[draw].mean(axis=1)
    return {'mean':point,'interval95':np.percentile(boot,[2.5,97.5]).tolist(),'images':int(sum(np.isin(counts,[1,2,3,4]))) if macro else len(vals)}

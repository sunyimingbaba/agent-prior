import hashlib
import numpy as np

def stat(values,counts,label,macro=True):
    v=np.asarray(values,dtype=np.float64);cs=np.asarray(counts);rng=np.random.default_rng((2026111601+int(hashlib.sha256(label.encode()).hexdigest()[:8],16))%2**32)
    if macro:
        point=0.;boot=np.zeros(10000)
        for c in [1,2,3,4]:
            z=v[cs==c];assert len(z);point+=float(z.mean())/4;boot+=z[rng.integers(len(z),size=(10000,len(z)))].mean(axis=1)/4
    else:point=float(v.mean());boot=v[rng.integers(len(v),size=(10000,len(v)))].mean(axis=1)
    return {'mean':point,'interval95':np.percentile(boot,[2.5,97.5]).tolist(),'images':int(np.isin(cs,[1,2,3,4]).sum()) if macro else len(v),'input_images':len(v)}

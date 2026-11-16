import hashlib
import math
from pathlib import Path
import numpy as np
from stage_io import read,dump,sha
ROOT=Path(__file__).resolve().parent.parent

def stat(values,counts,label,macro=True):
    vs=np.asarray(values,dtype=np.float64);counts=np.asarray(counts);rng=np.random.default_rng((2026111601+int(hashlib.sha256(label.encode()).hexdigest()[:8],16))%2**32)
    if macro:
        point=0.;boot=np.zeros(10000)
        for c in [1,2,3,4]:
            v=vs[counts==c];assert len(v);point+=float(v.mean())/4;boot+=v[rng.integers(0,len(v),size=(10000,len(v)))].mean(axis=1)/4
    else:point=float(vs.mean());boot=vs[rng.integers(0,len(vs),size=(10000,len(vs)))].mean(axis=1)
    return {'mean':point,'interval95':np.percentile(boot,[2.5,97.5]).tolist(),'images':len(vs)}

def quantity_metrics(q,gallery,case):
    words={w:i+1 for i,w in enumerate('one two three four five six seven'.split())};out={}
    for w in ['A','B']:
        positive,negative=gallery['quantity_columns'][case['image']][w];prefix=gallery['quantity_text_order'][positive].rsplit(' ',1)[0]
        columns={words[text.rsplit(' ',1)[1][:-1]]:j for j,text in enumerate(gallery['quantity_text_order']) if text.rsplit(' ',1)[0]==prefix};assert columns[case['actual_count']]==positive
        ns=[float(q[j]) for c,j in sorted(columns.items()) if c!=case['actual_count']];p=float(q[positive]);pred=max(sorted(columns),key=lambda c:float(q[columns[c]]))
        out.update({w+'_pair_margin':p-float(q[negative]),w+'_pair_correct':float(p>float(q[negative])),w+'_hard_margin':p-max(ns),w+'_hard_correct':float(p>max(ns)),w+'_mean_margin':p-float(np.mean(ns)),w+'_argmax_count':pred})
    return out

def select(xs,key):return max(xs,key=lambda x:(x[key],-x['seed']))
def concordance(xs):
    import itertools
    signs=[]
    for a,b in itertools.combinations(xs,2):
        x=np.sign(a['A_hard_margin']-b['A_hard_margin']);y=np.sign(a['B_hard_margin']-b['B_hard_margin'])
        if x and y:signs.append(x*y)
    return None if not signs else float(np.mean(signs))

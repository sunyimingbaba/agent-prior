"""Bootstrap pairs; alternative-order corrections use exact recorded next-token logits."""
import hashlib
import json
from pathlib import Path
import numpy as np
DIR=Path(__file__).resolve().parent

def hit(diff):return float(diff>0)+.5*(diff==0)

def main():
    p=DIR/'逐图原始logits.json';stats=json.loads((DIR/'回执.json').read_bytes())
    assert hashlib.sha256(p.read_bytes()).hexdigest()==stats['raw']
    pairs=json.loads(p.read_bytes());values=[]
    for _,records in sorted(pairs.items()):
        z=[]
        for i,r in enumerate(records):
            a,b=r['observations']
            assert r['side']==i and a['order']==0 and b['order']==1
            d=(a['logit_A']-a['logit_B']+b['logit_B']-b['logit_A'])/2
            assert d==r['margin']
            z.append(d)
        for i in (0,1):
            x,y=[r['observations'][i] for r in records]
            assert (x['prompt'],x['prompt_tokens_sha256'])==(y['prompt'],y['prompt_tokens_sha256'])
        values.append([(hit(z[0])+hit(-z[1]))/2,hit(z[0]-z[1]),float(z[0]>0 and z[1]<0),
                       (int(z[0]==0)+int(z[1]==0))/2,float(z[0]==z[1])])
    x=np.array(values)
    n=len(x); assert n==57
    w=np.random.default_rng(271127).multinomial(n,np.full(n,1/n),size=20000)
    labels=('side_accuracy','pair_ranking','both_sides_strict','tied_images','tied_pairs')
    boots=w@x/n
    out={k:dict(mean=float(x[:,i].mean()),interval=np.quantile(boots[:,i],[.025,.975]).tolist(),
                numerator=float(x[:,i].sum()*(2 if i in (0,3) else 1)),denominator=n*(2 if i in (0,3) else 1)) for i,k in enumerate(labels)}
    (DIR/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out)

if __name__=='__main__':main()

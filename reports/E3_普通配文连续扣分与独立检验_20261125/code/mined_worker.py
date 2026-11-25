"""Direct A/B supervision on only the approved warm-start training photos."""
import os,sys,json,base64,hashlib
from pathlib import Path
pipe=os.fdopen(os.dup(1),'w',buffering=1);os.dup2(2,1)
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'multicap_rfpo_20261124/code'))
from runtime import table,rgb,read,sha,setup
from readouts import load_model
from fractions import Fraction
def reciprocal(value,neg):
    above=int((neg>value).sum());tied=1+int((neg==value).sum())
    v=sum(Fraction(1,k) for k in range(above+1,above+tied+1))/tied
    return [v.numerator,v.denominator]
@torch.inference_mode()
def main(seed,group):
    setup();torch.set_num_threads(3)
    lock=read(ROOT/'inputs/mined_lock.json');path=ROOT/'inputs/mined_training_bank.pt';assert sha(path)==lock['bank']
    q=ROOT/'inputs/mined_roster.json';assert sha(q)==lock['training']
    cases={r['image']:r for r in read(q)[seed+'_'+group]};m=load_model(False);t=table();z=torch.load(path,weights_only=True)
    native=z['native'].double().numpy();hard=z['hard'].double().numpy()
    odd=np.array([int(hashlib.sha256(','.join(map(str,v)).encode()).hexdigest()[-1],16)%2==1 for v in z['native_tokens'].tolist()])
    print('READY',file=pipe)
    for line in sys.stdin:
        data=json.loads(line);n=data['image'];c=cases[n];pix=m.preprocess(rgb(t,c))[None]
        ix=z['maps'][n];ps=ix['native'];alt=sorted(set(ps)-{ps[0]});ok=odd.copy();ok[ps]=False
        rows=np.frombuffer(base64.b64decode(data['data']),dtype=np.float32).reshape(17,3584);cache={};ranks=[];ds=[]
        for h in rows:
            key=h.tobytes()
            if key not in cache:
                v=m.img_embeds(pix,agent_hidden=torch.tensor(h.copy())[None])['image_feats'][0].double().numpy()
                a=native@v;d=hard@v;cache[key]=([reciprocal(a[k],a[ok]) for k in alt],sum(int(d[ix[k][0]]>d[ix[k][1]]) for k in 'AB'))
            a,b=cache[key];ranks.append(a);ds.append(b)
        print(json.dumps(dict(rr=ranks,detail=ds,positive=alt)),file=pipe)
if __name__=='__main__':main(*sys.argv[1:])

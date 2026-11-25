"""Direct supervision on the 24 optimization images only."""
import os,sys,json,base64,hashlib
from fractions import Fraction
from pathlib import Path
pipe=os.fdopen(os.dup(1),'w',buffering=1);os.dup2(2,1)
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'multicap_rfpo_20261124/code'))
from runtime import OLD,table,rgb,pool,read,sha,setup
from readouts import load_model
def reciprocal(value,neg):
    n=int((neg>value).sum());t=int((neg==value).sum())+1
    x=sum(Fraction(1,k) for k in range(n+1,n+t+1))/t
    return [x.numerator,x.denominator]
@torch.inference_mode()
def main():
    setup();torch.set_num_threads(3);m=load_model(False);t=table();allc=pool();valid={r['image'] for r in read(BASE/'source_transfer_2h_20261125/inputs/training_roster.json')['rows']}
    banks={}
    for sub in ('','confirm'):
        path=OLD/sub/'records/text.pt';assert sha(path)==read(path.parent/'read_complete.json')['text'];b=torch.load(path,weights_only=True)
        odd=np.array([int(hashlib.sha256(','.join(map(str,v)).encode()).hexdigest()[-1],16)%2==1 for v in b['native_tokens'].tolist()])
        banks[sub]=(b['native'].double().numpy(),b['hard'].double().numpy(),odd,b['maps'])
    print('READY',file=pipe)
    for line in sys.stdin:
        q=json.loads(line);n=q['image'];assert n in valid;c=allc[n];pix=m.preprocess(rgb(t,c))[None]
        native,hard,odd,maps=banks[c['sub']];ix=maps[n];ps=ix['native'];alt=sorted(set(ps)-{ps[0]});ok=odd.copy();ok[ps]=False
        rows=np.frombuffer(base64.b64decode(q['data']),dtype=np.float32).reshape(17,3584);cache={};ranks=[];ds=[]
        for h in rows:
            key=h.tobytes()
            if key not in cache:
                v=m.img_embeds(pix,agent_hidden=torch.tensor(h.copy())[None])['image_feats'][0].double().numpy()
                s=native@v;d=hard@v;cache[key]=([reciprocal(s[k],s[ok]) for k in alt],sum(int(d[ix[k][0]]>d[ix[k][1]]) for k in 'AB'))
            a,b=cache[key];ranks.append(a);ds.append(b)
        print(json.dumps(dict(rr=ranks,detail=ds,positive=alt)),file=pipe)
if __name__=='__main__':main()

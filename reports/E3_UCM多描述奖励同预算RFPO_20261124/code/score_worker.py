import os,sys,json,base64,hashlib
pipe=os.fdopen(os.dup(1),'w',buffering=1);os.dup2(2,1)
import numpy as np,torch
from runtime import ROOT,OLD,REBUILD,pool,table,rgb,setup,sha,read
from readouts import load_model,scorer_for

def reciprocal(p,neg):
    above=int((neg>p).sum());ties=int((neg==p).sum())
    return float(np.mean(1/np.arange(above+1,above+ties+2)))

@torch.inference_mode()
def main():
    setup();m=load_model(False);proxy=scorer_for(load_model(True),REBUILD/'relation_candidates_20261120/gallery/train.pt')
    t=table();rows=pool();banks={}
    for sub in ('','confirm'):
        dr=OLD/sub/'records';assert sha(dr/'text.pt')==read(dr/'read_complete.json')['text']
        b=torch.load(dr/'text.pt',weights_only=True)
        hashes=[hashlib.sha256(','.join(map(str,r)).encode()).hexdigest() for r in b['native_tokens'].tolist()]
        banks[sub]=(b['maps'],b['hard'].double().numpy(),b['native'].double().numpy(),hashes,np.array([int(h[-1],16)%2==0 for h in hashes]))
    print('READY',file=pipe)
    for line in sys.stdin:
        req=json.loads(line);nm=req['image'];c=rows[nm];pix=m.preprocess(rgb(t,c))[None]
        maps,hard,native,hashes,even=banks[c['sub']];ix=maps[nm];p=ix['native']
        alt=sorted(set(p)-{p[0]},key=hashes.__getitem__);trained=[p[0]]+alt[:1] if len(alt)>=2 else [p[0]];held=alt[1:] if len(alt)>=2 else alt
        assert held and not set(trained)&set(held)
        mask=np.ones(len(even),dtype=bool);mask[p]=False
        states=np.frombuffer(base64.b64decode(req['data']),dtype=np.float32).reshape(-1,3584);assert len(states)==17
        cache={};out=[]
        for h in states:
            ident=h.tobytes()
            if ident not in cache:
                hidden=torch.from_numpy(h.copy());v=m.img_embeds(pix,agent_hidden=hidden[None])['image_feats'][0].double().numpy()
                hs=hard@v;s=native@v;x=s[mask&even]
                cache[ident]=dict(rank=proxy.score(hidden.tolist(),c['row'])[0],native=float(s[p[0]]-x.max()),keep=float(np.mean([reciprocal(s[i],x) for i in trained])),
                    rr=reciprocal(s[held].max(),s[mask&~even]),**{k:float(hs[ix[k][0]]-hs[ix[k][1]]) for k in 'ABCD' if k in ix})
            out.append(cache[ident])
        print(json.dumps(out,allow_nan=False),file=pipe)
if __name__=='__main__':main()

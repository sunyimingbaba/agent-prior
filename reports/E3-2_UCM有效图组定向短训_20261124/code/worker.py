import os,sys,json,base64,hashlib
pipe=os.fdopen(os.dup(1),'w',buffering=1)
os.dup2(2,1)
import torch,numpy as np
from common import OLD,REBUILD,pool,table,rgb,setup
from readouts import load_model,scorer_for
@torch.inference_mode()
def main():
    setup();reader=load_model(False);proxy=scorer_for(load_model(True),REBUILD/'relation_candidates_20261120/gallery/train.pt')
    banks={}
    for sub in ('','confirm'):
        b=torch.load(OLD/sub/'records/text.pt',weights_only=True)
        even=np.array([int(hashlib.sha256(','.join(map(str,t)).encode()).hexdigest()[-1],16)%2==0 for t in b['native_tokens'].tolist()])
        banks[sub]=(b['maps'],b['hard'].double(),b['native'].double(),even)
    rows=pool();t=table();print('READY',file=pipe)
    for line in sys.stdin:
        req=json.loads(line);nm=req['image'];c=rows[nm];maps,hard,native,even=banks[c['source']];ix=maps[nm]
        pix=reader.preprocess(rgb(t,c))[None];states=np.frombuffer(base64.b64decode(req['hidden']),np.float32).reshape(-1,3584)
        assert len(states)==24
        neg=even.copy();pos=ix['native'];neg[pos]=False;cache={};out=[]
        for h in states:
            k=h.tobytes()
            if k not in cache:
                z=torch.from_numpy(h.copy());v=reader.img_embeds(pix,agent_hidden=z[None])['image_feats'][0].double()
                ns=native@v;hs=hard@v
                cache[k]=dict(rank=proxy.score(z.tolist(),c['row'])[0],native=float(ns[pos[0]]-ns[neg].max()),
                    **{l:float(hs[ix[l][0]]-hs[ix[l][1]]) for l in 'AB'})
            out.append(cache[k])
        print(json.dumps(out),file=pipe)
if __name__=='__main__':main()

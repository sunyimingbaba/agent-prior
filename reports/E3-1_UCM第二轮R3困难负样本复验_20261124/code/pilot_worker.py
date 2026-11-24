import sys,os,json,base64,hashlib
import torch
import numpy as np
from base import RUN,REBUILD,read,table,rgb,setup
@torch.inference_mode()
def main():
    pipe=os.fdopen(os.dup(1),'w',buffering=1);os.dup2(2,1);setup()
    from readouts import load_model,scorer_for
    reader=load_model(False);proxy=scorer_for(load_model(True),REBUILD/'relation_candidates_20261120/gallery/train.pt')
    bank=torch.load(RUN/'records/text.pt',weights_only=True);hard=bank['hard'].double();native=bank['native'].double()
    even=np.array([int(hashlib.sha256(','.join(map(str,t)).encode()).hexdigest()[-1],16)%2==0 for t in bank['native_tokens'].tolist()])
    t=table();cases={c['image']:c for c in read(RUN/'pilot/train.json')}
    print('READY',file=pipe)
    for line in sys.stdin:
        req=json.loads(line);name=req['image'];c=cases[name];ix=bank['maps'][name];pos=ix['native'];neg=even.copy();neg[pos]=False
        pix=reader.preprocess(rgb(t,c))[None];states=np.frombuffer(base64.b64decode(req['hidden']),dtype=np.float32).reshape(-1,3584);cache={};result=[]
        for h in states:
            key=h.tobytes()
            if key not in cache:
                z=torch.from_numpy(h.copy());v=reader.img_embeds(pix,agent_hidden=z[None])['image_feats'][0].double()
                score=native@v;hs=hard@v
                cache[key]=dict(rank=proxy.score(z.tolist(),c['row'])[0],native=float(score[pos[0]]-score[neg].max()),A=float(hs[ix['A'][0]]-hs[ix['A'][1]]))
            result.append(cache[key])
        print(json.dumps(result),file=pipe)
if __name__=='__main__':main()

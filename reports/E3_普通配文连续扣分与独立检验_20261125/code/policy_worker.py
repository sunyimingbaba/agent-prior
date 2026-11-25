"""Reward-visible scores only. There are 16 sampled trajectories and one genuine greedy state."""
import os,sys,json,base64,hashlib
from pathlib import Path
pipe=os.fdopen(os.dup(1),'w',buffering=1);os.dup2(2,1)
import numpy as np,torch
from locked_selector import candidate_features,neighbors,ordinal_prediction
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'multicap_rfpo_20261124/code'))
from runtime import OLD,REBUILD,table,rgb,pool,read,sha,setup
from readouts import load_model,scorer_for
from score_worker import reciprocal
@torch.inference_mode()
def main():
    setup();torch.set_num_threads(3);m=load_model(False);proxy=scorer_for(load_model(True),REBUILD/'relation_candidates_20261120/gallery/train.pt')
    data=table();cases=pool();donors=read(ROOT/'inputs/donors.json');lock=read(ROOT/'inputs/multi_nomination.json')
    banks={}
    for sub in ('','confirm'):
        folder=OLD/sub/'records';f=folder/'text.pt';assert sha(f)==read(folder/'read_complete.json')['text']
        b=torch.load(f,weights_only=True);n=b['native'].double().numpy()
        hashes=[hashlib.sha256(','.join(map(str,x)).encode()).hexdigest() for x in b['native_tokens'].tolist()]
        ctx={}
        for sid,ds in donors.items():
            p=np.stack([n[q].mean(0) for _,q in ds]);p/=np.linalg.norm(p,axis=1,keepdims=True);ctx[int(sid)]=(p,ds)
        banks[sub]=(n,b['hard'].double().numpy(),b['maps'],hashes,np.array([int(x[-1],16)%2==0 for x in hashes]),ctx)
    print('READY',file=pipe)
    for line in sys.stdin:
        msg=json.loads(line);c=cases[msg['image']];im=m.preprocess(rgb(data,c))[None];cache={};vs=[];ranks=[]
        hidden=np.frombuffer(base64.b64decode(msg['data']),dtype=np.float32).reshape(17,3584)
        for state in hidden:
            key=state.tobytes()
            if key not in cache:
                h=torch.from_numpy(state.copy());v=m.img_embeds(im,agent_hidden=h[None])['image_feats'][0].double().numpy()
                cache[key]=v,int(proxy.score(h.tolist(),c['row'])[0])
            v,r=cache[key];vs.append(v);ranks.append(r)
        v=np.stack(vs);n,b,maps,hashes,even,ctx=banks[c['sub']];ix=maps[c['image']]
        p0=ix['native'][0];pos=[p0]+sorted(set(ix['native'])-{p0},key=hashes.__getitem__);assert len(pos)>=3
        vis=pos[:2];ok=even.copy();ok[pos]=False;ns=v@n.T;hs=v@b.T;left=ns[:,ok];neg=left.max(1);sim=ns[:,vis]
        h=np.column_stack([hs[:,ix[k][0]]-hs[:,ix[k][1]] for k in 'AB']);tr=np.array([[reciprocal(y,no) for y in row] for row,no in zip(sim,left)])
        nn,w=neighbors(ctx[c['scene_id']],vis,pos,n);sorted_n=np.sort(left,axis=1)
        if nn:
            nei=ns[:,nn];nr=1/(1+np.stack([len(no)-np.searchsorted(no,row,side='right') for no,row in zip(sorted_n,nei)]))
            nsim=nei@w;nmrr=nr@w
        else:nsim=sim.mean(1);nmrr=tr.mean(1)
        x=candidate_features(sim,tr,neg,h,v,nsim,nmrr)
        order=ordinal_prediction(lock['model'],lock['weights'],x)
        pack=dict(rank=ranks,native=(ns[:,p0]-neg).tolist(),A=h[:,0].tolist(),B=h[:,1].tolist(),
                  keep=tr.mean(1).tolist(),order=order.tolist(),features=x.tolist(),visible=vis,donors=nn)
        print(json.dumps(pack,allow_nan=False),file=pipe)
if __name__=='__main__':main()

import sys,json,hashlib
import numpy as np,torch
from runtime import ROOT,REBUILD,OLD,read,dump,sha,pool,setup,table,rgb
from readouts import load_model,scorer_for

def reciprocal(p,neg):
    i=1+sum(x>p for x in neg);n=1+sum(x==p for x in neg)
    return sum(1/(i+j) for j in range(n))/n

@torch.inference_mode()
def main():
    setup();m=load_model(False);proxy=scorer_for(load_model(True),REBUILD/'relation_candidates_20261120/gallery/train.pt');t=table();ps=pool();out={}
    for seed in 'ab':
        p=ROOT/'records'/(seed+'_first_batch.pt');z=torch.load(p,weights_only=True);nm=z['image'];c=ps[nm]
        bank=torch.load(OLD/c['sub']/'records/text.pt',weights_only=True)
        hashes=[hashlib.sha256(','.join(str(x) for x in ids).encode()).hexdigest() for ids in bank['native_tokens'].tolist()]
        ix=bank['maps'][nm];pos=ix['native'];alternatives=sorted(set(pos)-{pos[0]},key=hashes.__getitem__)
        train=[pos[0]]+(alternatives[:1] if len(alternatives)>=2 else []);held=alternatives[1:] if len(alternatives)>=2 else alternatives
        xs=[i for i,k in enumerate(hashes) if int(k,16)%2==0 and i not in pos];ys=[i for i,k in enumerate(hashes) if int(k,16)%2 and i not in pos]
        raw=json.loads((ROOT/'pilot'/(seed+'_R2')/'steps.jsonl').open().readline());target=raw['scores']+[raw['anchor']['scores']]
        pix=m.preprocess(rgb(t,c))[None]
        vs=torch.stack([m.img_embeds(pix,agent_hidden=h.float()[None])['image_feats'][0] for h in z['states']])
        sim=(vs.double()@bank['native'].double().T).numpy();hard=(vs.double()@bank['hard'].double().T).numpy();err=0.
        for v,ss,hh,expect in zip(z['states'],sim,hard,target):
            got=dict(rank=proxy.score(v.float().tolist(),c['row'])[0],native=ss[pos[0]]-max(ss[xs]),keep=np.mean([reciprocal(ss[i],ss[xs]) for i in train]),rr=reciprocal(max(ss[held]),ss[ys]))
            for k in 'ABCD':
                if k in ix:got[k]=hh[ix[k][0]]-hh[ix[k][1]]
            assert got['rank']==expect['rank'];err=max(err,max(abs(got[k]-expect[k]) for k in got))
        assert err<1e-9
        out[seed]=dict(states=17,exact_tokens_and_logp=True,score_max_abs_error=err,states_sha256=sha(p))
    dump(ROOT/'results/step_replay.json',out)
if __name__=='__main__':main()

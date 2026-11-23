"""Measure fresh sampled priors on the 24 training photos using the raw R2 objective."""
import hashlib
import numpy as np
import torch
from policy import policy
from train import sample, SAMPLES, NATIVE
from readouts import load_model
from shared import ROOT, read, dump, sha, setup, table, rgb
from metrics import csvwrite, ci

@torch.inference_mode()
def main():
    setup()
    reader=load_model(False)
    index=read(NATIVE/'index.json')
    assert sha(NATIVE/'features.pt')==index['sha256']
    bank=torch.load(NATIVE/'features.pt',weights_only=True,map_location='cpu')
    features=bank['features'].double()
    even=np.array([int(hashlib.sha256(','.join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in bank['tokens'].tolist()])
    t=table()
    panel=read(SAMPLES)
    output=[]
    for arm in ['R0','R1','R2']:
        model,proc=policy(arm)
        for i,row in enumerate(panel):
            im=rgb(t,row)
            torch.manual_seed(721000+i)
            _,traces=sample(model,proc,im,row['scene_query'])
            mapping=index['maps'][row['image']]
            assert mapping['captions']==t['caption'][row['row']].as_py()
            keep=even.copy()
            keep[mapping['positive']]=False
            pix=reader.preprocess(im)[None].cuda()
            margins=[]
            for _,h,_,_,_ in traces:
                v=reader.img_embeds(pix,agent_hidden=h[None].cuda())['image_feats'].cpu().double()
                sims=(v@features.T)[0]
                margins.append(float(sims[mapping['first']]-sims[keep].max()))
            output.append(dict(arm=arm,image=row['image'],pair=row['pair'],mean_margin=float(np.mean(margins)),
                best_margin=max(margins),unique_tokens=len({tuple(z[0]) for z in traces}),
                eos_only=sum(len(z[0])==1 and z[0][-1] in model.generation_config.eos_token_id for z in traces),
                **{f'margin_{j}':value for j,value in enumerate(margins)}))
        del model,proc
        torch.cuda.empty_cache()
        print('Fresh training-panel check',arm,flush=True)
    out=ROOT/'results'
    out.mkdir(exist_ok=True)
    csvwrite(out/'training_panel_by_image.csv',output)
    result={}
    for a,b in [('R2','R1'),('R2','R0'),('R1','R0')]:
        va={r['image']:r for r in output if r['arm']==a}
        vb={r['image']:r for r in output if r['arm']==b}
        d={}
        for metric in ('mean_margin','best_margin','unique_tokens','eos_only'):
            paired={}
            for k in va:
                paired.setdefault(va[k]['pair'],[]).append(va[k][metric]-vb[k][metric])
            assert len(paired)==12 and all(len(x)==2 for x in paired.values())
            d[metric]=ci([np.mean(x) for x in paired.values()])
        result[a+'-'+b]=d
    dump(out/'training_panel_contrasts.json',result)
if __name__=='__main__':
    main()

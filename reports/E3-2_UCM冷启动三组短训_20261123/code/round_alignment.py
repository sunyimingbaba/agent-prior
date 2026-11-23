"""Follow each fixed cold-start first-round choice through the real tool loop on training images."""
import hashlib
from types import SimpleNamespace
import numpy as np
import torch
from shared import ROOT, ASSETS, read, dump, setup, table, rgb, sha
from readouts import load_model, clip
from rollout import rollout_one, SYSTEM_PROMPT, parse_tool_call
from tools import scene_classify_with_conf
from train import sample, SAMPLES, NATIVE
from policy import policy
from metrics import ci, csvwrite

@torch.inference_mode()
def main():
    setup()
    model,proc=policy('R0')
    reader=load_model(False)
    scene,pre=clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False)
    scene=scene.float().eval().cuda()
    cp=torch.load(ASSETS/'ucm_scene_cls.pt',weights_only=True,map_location='cpu')
    head=torch.nn.Linear(cp['feat_dim'],len(cp['classes']))
    head.load_state_dict(cp['head'])
    head=head.eval().cuda()
    native=read(NATIVE/'index.json')
    assert sha(NATIVE/'features.pt')==native['sha256']
    bank=torch.load(NATIVE/'features.pt',weights_only=True,map_location='cpu')
    feats=bank['features'].double()
    even=np.array([int(hashlib.sha256(','.join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in bank['tokens'].tolist()])
    generator=model.generate
    rows,photos=[],[]
    t=table()
    for i,r in enumerate(read(SAMPLES)):
        im=rgb(t,r)
        torch.manual_seed(721000+i)
        single,traces=sample(model,proc,im,r['scene_query'])
        answer=scene_classify_with_conf(im,model=scene,preprocess=pre,cls_head=head,cls_names=list(cp['classes']))[0]
        ix=native['maps'][r['image']]
        keep=even.copy()
        keep[ix['positive']]=False
        pixels=reader.preprocess(im)[None].cuda()
        def score(h):
            v=reader.img_embeds(pixels,agent_hidden=h[None].cuda())['image_feats'].cpu().double()
            sims=(v@feats.T)[0]
            return float(sims[ix['first']]-sims[keep].max())
        cache={}
        before,after=[],[]
        for j,(ids,h,_,_,txt) in enumerate(traces):
            first=score(h)
            used=0
            if parse_tool_call(txt) is None:
                final=first
            else:
                key=tuple(ids)
                if key not in cache:
                    first_pass=True
                    def continued(**kw):
                        nonlocal first_pass
                        if first_pass:
                            first_pass=False
                            assert torch.equal(kw['input_ids'],single.input_ids)
                            return SimpleNamespace(sequences=torch.cat([kw['input_ids'],torch.tensor([ids],device='cuda')],dim=1),
                                hidden_states=((h[None,None,:],),))
                        return generator(**kw)
                    model.generate=continued
                    try:
                        final_h,used,filtered=rollout_one(model,proc,im,r['scene_query'],{'scene_classify':lambda image:answer},
                            3,64,'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
                    finally:
                        model.generate=generator
                    assert used>=1 and filtered==0 and not first_pass
                    cache[key]=(score(final_h),used)
                final,used=cache[key]
            rows.append(dict(image=r['image'],pair=r['pair'],slot=j,ids=' '.join(map(str,ids)),tool_calls=used,
                first_margin=first,final_margin=final))
            before.append(first)
            after.append(final)
        a,b=np.array(before),np.array(after)
        ai,bi=np.triu_indices(len(a),1)
        da,db=a[ai]-a[bi],b[ai]-b[bi]
        chosen=a==a.max()
        photos.append(dict(image=r['image'],pair=r['pair'],strict_comparable=int(np.count_nonzero(da*db)),
            reversed=int(np.sum(da*db<0)),after_ties=int(np.sum((da!=0)&(db==0))),
            first_selected_final_lift=float(b[chosen].mean()-b.mean()),final_oracle_lift=float(b.max()-b.mean()),
            overlap=int(np.any(chosen & (b==b.max()))),first_variation=int(np.ptp(a)>0)))
    import csv
    with (ROOT/'results/training_panel_by_image.csv').open(encoding='utf-8-sig') as fp:
        reference={r['image']:r for r in csv.DictReader(fp) if r['arm']=='R0'}
    err=max(abs(row['first_margin']-float(reference[row['image']][f"margin_{row['slot']}"])) for row in rows)
    assert err<1e-6,err
    def paired(metric):
        by={}
        for z in photos:
            by.setdefault(z['pair'],[]).append(z[metric])
        return ci([np.mean(v) for v in by.values()])
    out=ROOT/'results'
    csvwrite(out/'round_alignment_candidates.csv',rows)
    csvwrite(out/'round_alignment_by_image.csv',photos)
    dump(out/'round_alignment.json',dict(images=len(photos),candidates=len(rows),tool_candidates=sum(r['tool_calls']>0 for r in rows),
        comparable=sum(r['strict_comparable'] for r in photos),reversed=sum(r['reversed'] for r in photos),
        after_ties=sum(r['after_ties'] for r in photos),with_first_variation=sum(r['first_variation'] for r in photos),
        overlap_with_variation=sum(r['overlap'] for r in photos if r['first_variation']),
        first_winner_lift=paired('first_selected_final_lift'),final_oracle_lift=paired('final_oracle_lift'),first_replay_error=err))
if __name__=='__main__':
    main()

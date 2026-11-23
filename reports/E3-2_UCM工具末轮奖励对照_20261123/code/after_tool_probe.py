"""On training images only, vary the second response while keeping the first real tool result fixed."""
import hashlib
import numpy as np
import torch
from policy import policy, cut
from rollout import rollout_one, SYSTEM_PROMPT, parse_tool_call
from terminal_prior import scene_tool
from readouts import load_model, scorer_for
from train import SAMPLES, NATIVE
from shared import ROOT, REBUILD, read, dump, sha, setup, rgb, table
from metrics import csvwrite

class Captured(Exception):
    """The true second-round deployment input is available."""

def cluster_ci(values,pairs):
    labels=sorted(set(pairs))
    sums=np.array([sum(v for v,p in zip(values,pairs) if p==k) for k in labels])
    ns=np.array([pairs.count(k) for k in labels])
    draws=np.random.default_rng(20261123).integers(len(labels),size=(20000,len(labels)))
    lo,hi=np.quantile(sums[draws].sum(1)/ns[draws].sum(1),[.025,.975])
    return dict(mean=float(np.mean(values)),lo=float(lo),hi=float(hi),images=len(values),pairs=len(labels))

@torch.inference_mode()
def main():
    setup()
    model,proc=policy('R0')
    reader=load_model(False)
    proxy=scorer_for(load_model(True),REBUILD/'relation_candidates_20261120/gallery/train.pt')
    classify=scene_tool()
    index=read(NATIVE/'index.json')
    assert sha(NATIVE/'features.pt')==index['sha256']
    bank=torch.load(NATIVE/'features.pt',weights_only=True,map_location='cpu')
    feats=bank['features'].double()
    even=np.array([int(hashlib.sha256(','.join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in bank['tokens'].tolist()])
    t=table()
    rows,aggregate,skips=[],[],[]
    generate=model.generate
    for i,case in enumerate(read(SAMPLES)):
        image=rgb(t,case)
        answer,_=classify(image)
        turns=[]
        def capture(**kw):
            turns.append(kw)
            if len(turns)==2:
                raise Captured
            return generate(**kw)
        model.generate=capture
        try:
            rollout_one(model,proc,image,case['scene_query'],{'scene_classify':lambda _:answer},3,64,'cuda',
                        temperature=0.,system_prompt=SYSTEM_PROMPT)
        except Captured:
            pass
        finally:
            model.generate=generate
        if len(turns)<2:
            skips.append(dict(image=case['image'],reason='first round ended'))
            continue
        mapping=index['maps'][case['image']]
        assert mapping['captions']==t['caption'][case['row']].as_py()
        alt=sorted(set(mapping['positive'])-{mapping['first']})
        if not alt:
            skips.append(dict(image=case['image'],reason='no separate positive token sequence'))
            continue
        prefix=turns[1]
        kw={k:(v.repeat(8,*([1]*(v.ndim-1))) if isinstance(v,torch.Tensor) else v) for k,v in prefix.items()}
        kw.update(do_sample=True,temperature=1.)
        torch.manual_seed(908000+i)
        out=generate(**kw)
        n=prefix['input_ids'].shape[1]
        first_half=even.copy()
        other_half=~even
        first_half[mapping['positive']]=False
        other_half[mapping['positive']]=False
        pix=reader.preprocess(image)[None].cuda()
        margins,proxies,reciprocal=[],[],[]
        for j,row in enumerate(out.sequences[:,n:].tolist()):
            ids=cut(row,model.generation_config.eos_token_id)
            assert all(z==model.generation_config.pad_token_id for z in row[len(ids):])
            text=proc.decode(ids,skip_special_tokens=True)
            h=out.hidden_states[len(ids)-1][-1][j,-1].cpu()
            third=bool(parse_tool_call(text))
            if third:
                feedback=proc(text=[text+f'\n<result>{answer}</result>\n'],return_tensors='pt').input_ids.to('cuda')
                allids=torch.cat([prefix['input_ids'],feedback],dim=1)
                last=generate(**dict(prefix,input_ids=allids,attention_mask=torch.ones_like(allids)))
                h=last.hidden_states[-1][-1][0,-1].cpu()
            v=reader.img_embeds(pix,agent_hidden=h[None].cuda())['image_feats'].cpu().double()
            sim=(v@feats.T)[0].numpy()
            margin=float(sim[mapping['first']]-sim[first_half].max())
            neg=sim[other_half]
            target=sim[alt].max()
            greater=int(np.count_nonzero(neg>target))
            same=int(np.count_nonzero(neg==target))
            mrr=float(np.mean(1/np.arange(greater+1,greater+same+2)))
            proxy_rank,_=proxy.score(h.float().tolist(),case['row'])
            reciprocal.append(mrr)
            margins.append(margin)
            proxies.append(-proxy_rank)
            rows.append(dict(image=case['image'],pair=case['pair'],slot=j,ids=' '.join(map(str,ids)),
                third_round=int(third),R2_margin=margin,R1_rank=proxy_rank,alternative_MRR=mrr))
        mrr=np.array(reciprocal)
        best=lambda arr:float(mrr[np.array(arr)==max(arr)].mean())
        aggregate.append(dict(image=case['image'],pair=case['pair'],random=float(mrr.mean()),R1=best(proxies),
                R2=best(margins),oracle=float(mrr.max()),distinct=len(set(margins))))
        del out
    assert len(aggregate)+len(skips)==24
    assert aggregate
    folder=ROOT/'results'
    csvwrite(folder/'after_tool_candidates.csv',rows)
    csvwrite(folder/'after_tool_images.csv',aggregate)
    summary={k:float(np.mean([x[k] for x in aggregate])) for k in ('R1','R2','random','oracle','distinct')}
    for a,b in [('R2','random'),('R2','R1'),('oracle','random')]:
        summary[a+'-'+b]=cluster_ci([r[a]-r[b] for r in aggregate],[r['pair'] for r in aggregate])
    dump(folder/'after_tool_probe.json',dict(reachable=len(aggregate),skips=skips,measures=summary))
    print('After-tool probe finished',len(aggregate),flush=True)
if __name__=='__main__':
    main()

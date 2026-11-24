"""Archive frozen reader vectors, proxy ranks and the exact text columns used in evaluation."""
import hashlib,time,sys
import numpy as np
import torch
from base import RUN,REBUILD,ASSETS,sha,read,dump,panel,table,rgb,setup,savept
from readouts import load_model,scorer_for,clip

@torch.inference_mode()
def main():
    setup();rows=panel();t=table()
    reader=load_model(False);proxy=scorer_for(load_model(True),REBUILD/'relation_candidates_20261120/gallery/train.pt')
    native=REBUILD/'relation_candidates_20261120/native';index=read(native/'index.json')
    assert sha(native/'features.pt')==index['sha256']
    bank=torch.load(native/'features.pt',weights_only=True,map_location='cpu')
    wording='ABCD' if 'D' in rows[0] else 'ABC'
    texts=sorted({s for r in rows for key in wording for s in r[key]})
    tt=clip.tokenize(texts,context_length=77-reader.prompt_length)
    assert len(tt.unique(dim=0))==len(texts)
    hard=torch.cat([reader.txt_embeds(tok)['text_feats'] for tok in tt.split(12)])
    ix={q:j for j,q in enumerate(texts)};maps={}
    for r in rows:
        ps=[index['columns'][q] for q in r['captions']]
        tok=clip.tokenize(r['captions'],context_length=77-reader.prompt_length)
        assert torch.equal(tok,bank['tokens'][ps]) and len(set(ps))>=2
        hard_ids={k:[ix[q] for q in r[k]] for k in wording}
        # Confirm separation of rephrased evaluation pairs from the first reward pair.
        assert len(set(sum(hard_ids.values(),[])))==2*len(wording)
        maps[r['image']]=dict(native=ps,**hard_ids)
    extra='--extra' in sys.argv
    if extra:
        previous=torch.load(RUN/'records/text.pt',weights_only=True,map_location='cpu')
        assert torch.equal(previous['hard'],hard) and previous['maps']==maps
    else:
        savept(RUN/'records/text.pt',dict(texts=texts,hard=hard,tokens=tt,native=bank['features'],native_tokens=bank['tokens'],maps=maps))
    for arm in ('a','b'):
        arm=('extra_' if extra else '')+arm
        src=RUN/'records'/arm;out=RUN/'records'/f'{arm}_readouts';out.mkdir(exist_ok=True)
        for c in rows:
            f=src/(c['image']+'.pt');dest=out/f.name
            while not f.is_file():time.sleep(1)
            if dest.is_file():
                assert torch.load(dest,weights_only=True,map_location='cpu')['source']==sha(f)
                continue
            sample=torch.load(f,weights_only=True,map_location='cpu')
            assert sample['rgb']==c['rgb_sha256'] and sample['query']==c['scene_query']
            pix=reader.preprocess(rgb(t,c))[None]
            cache={};vis=[];ranks=[]
            for h in sample['hidden'].float():
                key=h.numpy().tobytes()
                if key not in cache:
                    v=reader.img_embeds(pix,agent_hidden=h[None])['image_feats'][0]
                    rank,_=proxy.score(h.tolist(),c['row'])
                    assert torch.isfinite(v).all() and abs(v.norm().item()-1)<1e-4
                    cache[key]=(v,rank)
                v,rank=cache[key];vis.append(v);ranks.append(rank)
            savept(dest,dict(image=c['image'],source=sha(f),visual=torch.stack(vis),proxy_rank=ranks,post_tool=sample['post_tool']))
            print(arm,c['image'],len(cache),flush=True)
        while not (src/'complete.json').is_file():time.sleep(1)
        proof=read(src/'complete.json')
        for c in rows:
            assert proof['files'][c['image']]==sha(src/(c['image']+'.pt'))
        dump(out/'complete.json',dict(adapter=proof['adapter'],files={c['image']:sha(out/(c['image']+'.pt')) for c in rows}))
    dump(RUN/'records'/('extra_read_complete.json' if extra else 'read_complete.json'),dict(text=sha(RUN/'records/text.pt'),gallery=index['sha256'],
            reader=sha(ASSETS/'ucm_main_domain_tool.ckpt'),torch=str(torch.__version__)))
if __name__=='__main__':main()

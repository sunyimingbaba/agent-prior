"""Hold inference at the last token of the input prompt for all four policies."""
import hashlib,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from shared import ROOT,ASSETS,read,dump,sha,setup,table,rgb
from rollout import rollout_one,load_mllm,SYSTEM_PROMPT
from readouts import load_model
from analysis import fixed,recalls
def main(arm):
    setup()
    out=ROOT/'first_state'/arm;out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'eval'/arm;assert (source/'scored.json').exists()
    adapter=ASSETS/'ucm_main_lora' if arm=='base' else ROOT/('policy_'+arm)/'adapter'
    qwen,proc=load_mllm(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'),'cuda',str(adapter))
    model=load_model(False);t=table('val')
    text=torch.load(source/'text.pt',map_location='cpu',weights_only=True)
    natural=torch.load(source/'vectors.pt',map_location='cpu',weights_only=True)
    allrows=[];hs=[];vs=[];golds=read(ROOT/'inputs/detail.json')
    with torch.inference_mode():
        for part,records in [('native',read(ROOT/'inputs/validation.json')),('detail',golds)]:
            for r in records:
                im=rgb(t,r)
                for view in (['upright'] if part=='native' else ['upright','rotated']):
                    im2=im if view=='upright' else im.transpose(Image.Transpose.ROTATE_180)
                    q=r['orig_query'] if part=='native' else r['scene_query']
                    h,used,filtered=rollout_one(qwen,proc,im2,q,{'scene_classify':lambda _:None},1,1,'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
                    assert used==filtered==0
                    v=model.img_embeds(model.preprocess(im2)[None].cuda(),agent_hidden=h[None].cuda())['image_feats'].cpu()
                    path=source/part/(Path(r['image']).stem+'_'+view+'.json');record=read(path)
                    ref=record['traces'][0]
                    same=hashlib.sha256(h.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==ref['hidden']
                    endpoints=len(ref['rounds'])==1 and len(ref['rounds'][0]['ids'])==1
                    if endpoints:
                        assert same
                        i=natural[part]['order'].index((r['image'],view,-1))
                        assert torch.equal(v[0],natural[part]['visual'][i])
                    hs.append(h);vs.append(v)
                    allrows.append(dict(image=r['image'],part=part,view=view,endpoint_matches=endpoints,hidden_equal=same))
    torch.save(dict(hidden=torch.stack(hs),visual=torch.cat(vs)),out/'vectors.pt');dump(out/'rows.json',allrows)
    s=fixed(torch.cat(vs[:210]),text['native'])[:,text['caption_order']]
    gt=np.arange(210)[:,None]==np.repeat(np.arange(210),5)[None,:];ds={}
    for direction,a,b in [('I2T',s,gt),('T2I',s.T,gt.T)]:
        for k,val in recalls(a,b).items():ds[f'{direction}_R{k}']=float(val.mean())
    ds['mR']=float(np.mean(list(ds.values())))
    rel=fixed(torch.cat(vs[210:]),text['relation'])
    items=[];allrows=allrows[210:]
    for r in golds:
        rows=[i for i,z in enumerate(allrows) if z['image']==r['image']]
        assert [allrows[i]['view'] for i in rows]==['upright','rotated']
        means=[];both=[]
        for w in 'CD':
            p,n=[text['texts'].index(s) for s in r[w]]
            m=(rel[rows,p]-rel[rows,n])*np.array([1.,-1.]);means.append(float((m>0).mean()));both.append(float((m>0).all()))
        items.append(dict(image=r['image'],both=float(np.mean(both)),acc=float(np.mean(means))))
    dump(out/'summary.json',dict(arm=arm,native=ds,detail_both=float(np.mean([z['both'] for z in items])),
        detail_accuracy=float(np.mean([z['acc'] for z in items])),detail_items=items,vectors_sha=sha(out/'vectors.pt'),
        native_and_detail_rows=242))
if __name__=='__main__':main(sys.argv[1])

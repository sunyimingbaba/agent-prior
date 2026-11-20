"""Strict frozen AGRS deployment readout, original RFPO proxy and native-caption split gallery."""
import sys
from pathlib import Path
import torch
from PIL import Image
from shared import ROOT,ASSETS,BASE,REBUILD,SCORER,table,rgb,read,dump,sha,setup
from readouts import clip,load_model,RetrievalScorer
from agent.grpo import reward

def main(split):
    setup()
    for name,expected in read(ROOT/'inputs/lock.json').items():assert sha(ROOT/name)==expected,name
    if split=='val':assert (ROOT/'train/decision.json').exists()
    allcases=read(ROOT/'inputs/retained.json')
    cases=[r for r in allcases if r['split']==split]
    t=table(split); names=t['path'].to_pylist()
    out=ROOT/split;out.mkdir(exist_ok=True)
    prev=REBUILD/'relation_candidates_20261120'
    gallery=prev/'gallery/train.pt' if split=='train' else REBUILD/'proxy_gallery_20261113/grpo_val_cache_rebuilt_last50.pt'
    if split=='train':
        ix=read(prev/'gallery/index.json');assert ix['gallery_sha256']==sha(gallery) and ix['names']==names
    else:
        ix=read(gallery.parent/'row_mapping.json')
        assert [r['image'] for r in ix]==names and all(r['arrow_row']==r['feature_row']==r['similarity_target_column']==i for i,r in enumerate(ix))
    proxy=load_model(True);model=load_model(False)
    sc=RetrievalScorer.__new__(RetrievalScorer);sc.pl=proxy;sc.device='cuda';sc.cache_path=str(gallery)
    sc.image_features,sc.image_indices=sc._load_or_build_cache()
    assert sc.image_indices==list(range(len(t))) and sc.image_features.shape==(len(t),768)
    textlist=sorted({s for r in cases for w in 'ABCD' for s in r[w]})
    tokens=clip.tokenize(textlist,context_length=77-model.prompt_length,truncate=False)
    assert len(set(map(tuple,tokens.tolist())))==len(textlist)
    captions=t['caption'].to_pylist()
    strings=list(dict.fromkeys(s for caps in captions for s in caps))
    cap_tok=clip.tokenize(strings,context_length=77-model.prompt_length,truncate=False)
    cap_group={}; keep=[]; cap_index={}
    for i,ids in enumerate(cap_tok.tolist()):
        key=tuple(ids)
        if key not in cap_group:cap_group[key]=len(keep);keep.append(i)
        cap_index[strings[i]]=cap_group[key]
    cap_tok=cap_tok[keep]
    with torch.inference_mode():
        def encode(ts):
            return torch.cat([model.txt_embeds(z.cuda())['text_feats'].cpu() for z in ts.split(32)])
        text=encode(tokens);caps=encode(cap_tok)
        visuals=[];rows=[];proxies=[]
        old_encode=proxy.model.encode_image
        seen=[]
        def capture(*args,**kwargs):
            f=old_encode(*args,**kwargs)
            seen.append((f/f.norm(dim=-1,keepdim=True)).cpu());return f
        proxy.model.encode_image=capture
        for r in cases:
            im=rgb(t,r)
            modes=['scene','original']
            if split=='val' and r['contrast'] in ('hv','diag','lr','tb','nose'):modes.append('rotated')
            for mode in modes:
                rot=mode=='rotated'
                x=im.transpose(Image.Transpose.ROTATE_90 if r['contrast'] in ('hv','diag') else Image.Transpose.ROTATE_180) if rot else im
                img=model.preprocess(x).unsqueeze(0).cuda()
                folder=ROOT/'samples'/mode/Path(r['image']).stem
                for k in range(-1,8):
                    file=folder/('greedy.pt' if k<0 else f'sample{k}.pt')
                    rec=read(file.with_suffix('.json'));assert rec['k']==k and rec['image']==r['image'] and rec['mode']==mode and rec['sha256']==sha(file)
                    h=torch.load(file,weights_only=True,map_location='cpu')
                    v=model.img_embeds(img,agent_hidden=h.unsqueeze(0).cuda())['image_feats'].cpu()
                    visuals.append(v)
                    z={key:r[key] for key in ('image','pair','side','row','contrast','scene','split')}
                    z.update(mode=mode,k=k,vector=rec['vector_sha256'],seed=rec['seed'],tokens=sum(len(q['ids']) for q in rec['rounds']))
                    if not rot:
                        seen.clear();rank,soft=sc.score(h.float().tolist(),r['row']);assert len(seen)==1
                        proxies.append(seen[0]);z['proxy_i']=len(proxies)-1
                        tr='\n'.join(q['decoded'] for q in rec['rounds'])
                        z.update(rank0=rank,fmt=reward.r_format(tr),old=reward.total_reward(tr,rank,lam=1.,mode='soft'))
                        assert z['old']==soft+z['fmt']
                    rows.append(z)
            print('scored',split,r['image'],flush=True)
    target={r['image']:sorted({cap_index[s] for s in captions[r['row']]}) for r in cases}
    native0={r['image']:cap_index[captions[r['row']][0]] for r in cases}
    torch.save(dict(text=text,tokens=tokens,native=caps,native_tokens=cap_tok,visual=torch.cat(visuals),proxy=torch.cat(proxies),
                    proxy_gallery=sc.image_features.cpu()),out/'features.pt')
    dump(out/'index.json',dict(texts=textlist,raw_captions=sum(map(len,captions)),raw_strings=len(strings),native_count=len(keep),
                               native_target=target,native_first=native0,names=names))
    dump(out/'rows.json',rows)
    dump(out/'binding.json',dict(gallery=str(gallery),gallery_hash=sha(gallery),arrow_hash=sha(ASSETS/'ucm'/f'ucm_caption_karpathy_label_{split}.arrow'),
          assets={str(p):sha(p) for p in [SCORER,ASSETS/'ucm_main_domain_tool.ckpt',ASSETS/'ucm_main_lora/adapter_model.safetensors',ASSETS/'ucm_scene_cls.pt',
          BASE/'CUP/agent/rollout.py',BASE/'CUP/agent/grpo/reward.py',BASE/'CUP/agent/grpo/scorer.py']},lock=sha(ROOT/'inputs/lock.json'),
          files={p:sha(out/p) for p in ['features.pt','index.json','rows.json']},device=torch.cuda.get_device_name(),torch=torch.__version__))
if __name__=='__main__':main(sys.argv[1])

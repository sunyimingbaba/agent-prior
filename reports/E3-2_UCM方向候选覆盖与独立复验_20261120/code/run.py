"""Generate paired candidate streams and intervene on the hidden input to the frozen retrieval model."""
import argparse,hashlib,random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from shared import ROOT,ASSETS,REBUILD,read,dump,sha,setup,rgb,table
from readouts import load_model,clip
from rollout import SYSTEM_PROMPT,load_mllm,rollout_one
from tools import scene_classify_with_conf
from cases import cases,query

PREV=REBUILD/'paired_views_20261120'
BASE_EVAL=PREV/'eval/base_00'
def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def save(path,values):
    tmp=path.with_suffix('.part')
    torch.save(values,tmp)
    tmp.replace(path)

@torch.inference_mode()
def main(group):
    setup()
    src=ROOT/'inputs'
    loc=ROOT/'outputs'/group;loc.mkdir(parents=True,exist_ok=True)
    design=read(src/'design.json')
    lock=read(src/(group+'_lock.json'))
    for name,h in lock.items():assert sha(ROOT/name)==h,name
    cfg=read(PREV/'environment.json')['files']
    for f in [ASSETS/'ucm_main_lora/adapter_model.safetensors',ASSETS/'models/ViT-L-14.pt',ASSETS/'ucm_scene_cls.pt']:
        assert sha(f)==cfg[str(f)]['sha256']
    model,processor=load_mllm(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'),'cuda',str(ASSETS/'ucm_main_lora'))
    active=hashlib.sha256()
    for name,p in model.named_parameters():
        if '.lora_' in name:active.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    proof=read(BASE_EVAL/'complete.json')
    assert active.hexdigest()==proof['active']
    reader=load_model()
    backbone,pre=clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False)
    backbone=backbone.float().eval().cuda()
    cp=torch.load(ASSETS/'ucm_scene_cls.pt',weights_only=True,map_location='cpu')
    head=torch.nn.Linear(cp['feat_dim'],len(cp['classes']))
    head.load_state_dict(cp['head']);head=head.cuda().eval()
    native=read(BASE_EVAL/'trainpanel.json')
    assert sha(BASE_EVAL/'trainpanel.json')==proof['parts']['trainpanel'] and sha(BASE_EVAL/'trainpanel.pt')==native['bundle']
    archived=torch.load(BASE_EVAL/'trainpanel.pt',weights_only=True,map_location='cpu')
    training=cases('train')
    records=cases(group);t=table('train' if group=='train' else 'val')
    original=model.generate;rounds=[]
    def capture(**kwargs):
        out=original(**kwargs)
        ids=out.sequences[0,kwargs['input_ids'].shape[1]:].cpu().tolist()
        rounds.append({'ids':ids,'text':processor.decode(ids,skip_special_tokens=True)})
        return out
    model.generate=capture
    for r in records:
        tag=Path(r['image']).stem
        meta=loc/(tag+'.json');path=loc/(tag+'.pt')
        if meta.exists():
            m=read(meta)
            assert m['bundle']==sha(path) and m['lock']==sha(src/(group+'_lock.json'))
            continue
        orig=rgb(t,r);images=[orig,orig.transpose(Image.Transpose.ROTATE_180)]
        prompts=[query(r,c) for c in design['conditions']]
        hiddens=[];logs=[]
        for cond,prompt in zip(design['conditions'],prompts,strict=True):
            for view,im in enumerate(images):
                # The scene classifier is deterministic at inference; reuse its real answer on repeated calls for the same pixels.
                answer,conf=scene_classify_with_conf(im,model=backbone,preprocess=pre,cls_head=head,cls_names=list(cp['classes']))
                for slot in range(9):
                    seed=(394000+10*r['row'] if slot==0 else design['seed']+100*r['row']+10*view+slot+(0 if group=='train' else 300000))
                    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed);np.random.seed(seed)
                    rounds.clear()
                    h,used,filtered=rollout_one(model,processor,im,prompt,{'scene_classify':lambda image:answer},3,64,'cuda',
                                               temperature=float(slot>0),system_prompt=SYSTEM_PROMPT)
                    assert filtered==0 and h.shape==(3584,) and h.dtype==torch.bfloat16 and torch.isfinite(h).all()
                    logs.append(dict(cue=cond,view=view,slot=slot,seed=seed,rounds=list(rounds),used=int(used),filtered=int(filtered),
                                     hidden=digest(h),scene_answer=answer,confidence=float(conf)))
                    hiddens.append(h)
        # Both pixels see exactly the same list of hidden states. Foreign states exclude both members of the target's reciprocal training pair.
        donors=[j for j,x in enumerate(native['records']) if group=='fresh' or training[j//2]['pair']!=r['pair']]
        own=torch.stack(hiddens);all_h=torch.cat([own,archived['hidden'][donors]])
        vis=[];rawvis=[]
        for im in images:
            pic=reader.preprocess(im)[None].cuda()
            # Single-item readout matches the historical inference path and its rounding.
            v=torch.stack([reader.img_embeds(pic,agent_hidden=h[None].cuda())['image_feats'][0].cpu() for h in all_h])
            vis.append(v)
            pix=pre(im)[None].cuda()
            empty=pix.new_empty(1,0,backbone.visual.conv1.weight.shape[0])
            rawvis.append(torch.nn.functional.normalize(backbone.encode_image(pix,empty).float(),dim=1)[0].cpu())
        visual=torch.stack(vis)
        phrases=[s for w in ('A','B') for s in r[w]]
        # Historical training text bank uses one canonical batch. Bind to these exact features when available.
        if group=='train':
            assert sha(BASE_EVAL/'train_text.pt')==proof['train_text']
            bank=torch.load(BASE_EVAL/'train_text.pt',weights_only=True,map_location='cpu')
            txt=bank['features'][[bank['phrases'].index(s) for s in phrases]]
            ix=[j for j,z in enumerate(native['records']) if r['image']==z['image']]
            assert len(ix)==2
            for view,j in enumerate(ix):
                k=9*view
                assert logs[k]['rounds']==native['records'][j]['rounds'] and torch.equal(own[k],archived['hidden'][j])
                assert torch.equal(visual[view,k],archived['visual'][j])
        else:
            txt=reader.txt_embeds(clip.tokenize(phrases,context_length=77-reader.prompt_length).cuda())['text_feats'].cpu()
        rawtxt=torch.nn.functional.normalize(backbone.encode_text(clip.tokenize(phrases).cuda()).float(),dim=1).cpu()
        assert torch.isfinite(visual).all() and (visual.norm(dim=-1)-1).abs().max()<1e-5
        save(path,dict(own=own,visual=visual,text=txt,clip_visual=torch.stack(rawvis),clip_text=rawtxt))
        dump(meta,dict(image=r['image'],lock=sha(src/(group+'_lock.json')),readout=proof['readout'],active=active.hexdigest(),
                       bundle=sha(path),donor_input=native['bundle'],donors=[{'index':j,'image':native['records'][j]['image'],'view':native['records'][j]['view'],'hidden':native['records'][j]['hidden_sha']} for j in donors],
                       queries=prompts,phrases=phrases,rgb=[hashlib.sha256(im.tobytes()).hexdigest() for im in images],logs=logs))
        print(group,tag,36,len(donors),flush=True)
    dump(loc/'complete.json',{r['image']:sha(loc/(Path(r['image']).stem+'.json')) for r in records})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('group',choices=['train','fresh']);main(ap.parse_args().group)

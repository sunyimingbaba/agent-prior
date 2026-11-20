"""Generate blinded E1 priors via the original rollout and capture native token IDs."""
import argparse
from pathlib import Path
from environment import ROOT, PREV, BASE, ASSETS, QUERY, read, dump, digest, image_for, val_data
import torch
from rollout import SYSTEM_PROMPT, PLAIN_PROMPT, load_mllm, rollout_one
from tools import scene_classify_with_conf
from prompt_clip import clip

def main(method):
    use_tools = method != 'ASPF'
    model, proc = load_mllm(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'),'cuda',
                            str(ASSETS/'ucm_main_lora') if use_tools else '')
    calls=[]
    original=model.generate
    def capture(*args,**kwargs):
        out=original(*args,**kwargs)
        ids=out.sequences[0,kwargs['input_ids'].shape[-1]:].cpu().tolist()
        calls.append(dict(ids=ids,decoded=proc.decode(ids,skip_special_tokens=True)))
        return out
    model.generate=capture
    tools={}
    if use_tools:
        tool, preprocess = clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False)
        tool=tool.float().eval().cuda()
        cp=torch.load(ASSETS/'ucm_scene_cls.pt',map_location='cpu',weights_only=True)
        head=torch.nn.Linear(cp['feat_dim'],len(cp['classes']))
        head.load_state_dict(cp['head'],strict=True)
        head=head.eval().cuda()
        def scene(im):
            text,conf=scene_classify_with_conf(im,model=tool,preprocess=preprocess,cls_head=head,cls_names=list(cp['classes']))
            events.append(dict(text=text,confidence=float(conf)))
            return text if conf>=0 else None
        tools['scene_classify']=scene
    folder=ROOT/'priors'/method
    folder.mkdir(parents=True,exist_ok=True)
    eos=model.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else list(eos)
    data=val_data()
    records=read(ROOT/'inputs/candidates.json')
    if method=='AGSE_original':
        scored={r['image'] for r in read(ROOT/'inputs/cases.json')}
        records=[r for r in records if r['image'] in scored]
    for k,r in enumerate(records):
        name=Path(r['image']).stem
        pt=folder/(name+'.pt')
        record=folder/(name+'.json')
        query=r['scene_query'] if method=='AGSE_scene' else QUERY
        if method=='AGSE_original':
            query=data['val']['caption'][r['row']].as_py()[0]
            assert query==r['captions'][0]
        if record.exists():
            saved=read(record)
            assert saved['sha256']==digest(pt) and saved['query']==query and saved['rgb_sha256']==r['rgb_sha256']
            continue
        torch.manual_seed(0)
        calls.clear()
        events=[]
        with torch.inference_mode():
            h,used,filtered=rollout_one(model,proc,image_for(r,data),query,tools,3,64,'cuda',
                                        system_prompt=SYSTEM_PROMPT if use_tools else PLAIN_PROMPT,temperature=0.)
        assert h.shape==(3584,) and h.dtype==torch.bfloat16 and torch.isfinite(h).all()
        path=PREV/'priors'/method/(name+'.pt')
        prior_check=None
        if path.exists():
            base=read(path.with_suffix('.json'))
            assert base['sha256']==digest(path) and base['input']['rgb_sha256']==r['rgb_sha256']
            prior_check=torch.equal(h,torch.load(path,map_location='cpu',weights_only=True))
            assert prior_check
        archive_sha=None
        archive_cos=None
        if method=='AGSE_original':
            archive=BASE.parent/'ucm_prior_archive/agent_cache_domain_tool/ucm_caption_karpathy_label_val'/(str(r['row'])+'.pt')
            ref=torch.load(archive,map_location='cpu',weights_only=True).reshape_as(h)
            prior_check=torch.equal(h,ref)
            archive_sha=digest(archive)
            archive_cos=float(torch.nn.functional.cosine_similarity(h.float(),ref.float(),dim=0))
        tmp=pt.with_suffix('.part')
        torch.save(h,tmp)
        tmp.replace(pt)
        dump(record,dict(image=r['image'],rgb_sha256=r['rgb_sha256'],method=method,query=query,rounds=calls,
                         used=int(used),filtered=int(filtered),events=events,first_eos=calls[0]['ids'] in [[v] for v in eos],
                         old_bit_exact=prior_check,archive_sha256=archive_sha,archive_cos=archive_cos,sha256=digest(pt)))
        if (k+1)%20==0:
            print(method,k+1,flush=True)
    print('DONE',method,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('method',choices=['ASPF','AGSE','AGSE_scene','AGSE_original'])
    main(p.parse_args().method)

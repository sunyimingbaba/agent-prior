#!/usr/bin/env python3
"""E1 frozen CLIP/AGSE scorer. Defaults to four historical images on CPU."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import sys
from pathlib import Path

SMOKE = {'1983.tif','1981.tif','983.tif','988.tif'}
EXPECTED_MODELS = {
    'agse': '09ef20e0b354efbf8a7d4d49105cd92e1ae9412b3ea58820961265a065a874dc',
    'clip': 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836',
}

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def validate_cases(cases, audit, scope):
    if len(cases)!=200 or len(audit)!=100:
        raise ValueError('expected 200 text pairs and 100 audited images')
    ids = {r['图像编号'] for r in cases}
    if len(ids)!=100 or len({r['图像编号'] for r in audit})!=100:
        raise ValueError('image count or audit uniqueness mismatch')
    audit_by = {r['图像编号']:r for r in audit}
    if ids != set(audit_by):
        raise ValueError('case/audit images differ')
    seen=set()
    for r in cases:
        name=r['图像编号']; key=(name,r['句式'])
        if key in seen or r['句式'] not in ('A_原句式','B_改写句式'):
            raise ValueError(f'duplicate or invalid wording {key}')
        seen.add(key)
        prior=audit_by[name]
        if r['数据来源']!='UCM '+prior['来源'] or r['原始第一条描述']!=prior['第一条描述']:
            raise ValueError(f'split/caption mismatch: {name}')
        position=int(r['正确文本位置'])
        if position not in (1,2) or [r['文本1'],r['文本2']][position-1]!=r['正确描述'] or [r['文本1'],r['文本2']][2-position]!=r['错误描述']:
            raise ValueError(f'correct/incorrect text mismatch: {name}')
    if len(seen)!=200:
        raise ValueError('missing wording')
    selected=[r for r in cases if scope=='all' or r['图像编号'] in SMOKE]
    if scope=='smoke' and (len(selected)!=8 or {r['图像编号'] for r in selected}!=SMOKE):
        raise ValueError('expected 4 named historical images and 8 pairs')
    return selected,audit_by

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/private/AGSE_E1_20261110'))
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--scope',choices=('smoke','all'),default='smoke')
    args=parser.parse_args()
    root=args.root.resolve(); out=args.output_dir.resolve()
    if out==root or root not in out.parents:
        raise ValueError('output directory must be a new subdirectory of the private experiment root')
    cases=read_csv(args.cases);audit=read_csv(args.audit);selected,by=validate_cases(cases,audit,args.scope)
    base=root/'legacy'; ckpt_path=base/'assets/ucm_main_domain_tool.ckpt';clip_path=base/'assets/models/ViT-L-14.pt'
    for name,path in [('agse',ckpt_path),('clip',clip_path)]:
        actual=sha256(path)
        if actual!=EXPECTED_MODELS[name]:
            raise ValueError(f'{name} model SHA-256 mismatch: {actual}')
        print('MODEL_SHA_OK',name,actual,flush=True)
    # Import dependencies only after static input and model checks.
    import pyarrow as pa
    import torch
    from PIL import Image
    random.seed(0);torch.manual_seed(0);torch.set_num_threads(min(8,os.cpu_count() or 1))
    splits={r['数据来源'].split()[-1] for r in selected}
    tables={s:pa.ipc.open_file(str(base/f'assets/ucm/ucm_caption_karpathy_label_{s}.arrow')).read_all() for s in splits}
    priorroot=root/'ucm_prior_archive/agent_cache_domain_tool';data={}
    for case in selected:
        name=case['图像编号']
        if name in data:continue
        ref=by[name];split=ref['来源'];idx=int(ref['Arrow行号']);table=tables[split]
        if table['path'][idx].as_py()!=name or table['caption'][idx].as_py()[0]!=ref['第一条描述']:
            raise ValueError(f'Arrow mapping or first caption mismatch: {name}')
        img=Image.open(io.BytesIO(table['image'][idx].as_py())).convert('RGB')
        if hashlib.sha256(img.tobytes()).hexdigest()!=ref['图像RGB_SHA256']:
            raise ValueError(f'Arrow image mismatch: {name}')
        path=priorroot/ref['先验文件']
        if sha256(path)!=ref['先验SHA256']:
            raise ValueError(f'prior SHA-256 mismatch: {name}')
        hidden=torch.load(path,map_location='cpu',weights_only=True).float().reshape(1,-1)
        if tuple(hidden.shape)!=(1,3584) or not bool(torch.isfinite(hidden).all()):
            raise ValueError(f'invalid prior tensor: {name}')
        data[name]=(idx,img,hidden)
    print('INPUTS_VERIFIED',len(data),'images',len(selected),'pairs',flush=True)
    out.mkdir(parents=True,exist_ok=False)
    meta={'scope':args.scope,'device':'cpu','python':platform.python_version(),'torch':torch.__version__,
          'pyarrow':pa.__version__,'cases_sha256':sha256(args.cases),'audit_sha256':sha256(args.audit),
          'model_sha256':EXPECTED_MODELS,'count_images':len(data),'count_pairs_per_model':len(selected),
          'original_first_caption_used_for_prior':True,'status':'started'}
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    sys.path.insert(0,str(base/'CUP'))
    from prompt_clip import clip
    real_load=clip.load
    def cpu_load(name,device='cpu',jit=False,download_root=None):
        if name=='ViT-L/14':name=str(clip_path)
        return real_load(name,device='cpu',jit=jit,download_root=download_root)
    clip.load=cpu_load
    from vilt.modules.vilt_module import ViLTransformerSS
    class MyBackboneFinetuning:
        def __init__(self,*args,**kwargs):pass
    import __main__
    __main__.MyBackboneFinetuning=MyBackboneFinetuning
    results=[]
    for method in ('CLIP','AGSE'):
        if method=='CLIP':
            model,preprocess=real_load(str(clip_path),device='cpu',jit=False);model.float().eval()
        else:
            ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
            config=dict(ckpt['hyper_parameters']['config']);config['load_path']='';config['test_only']=False
            if config.get('use_agent_prior') is not True:raise ValueError('AGSE agent prior disabled')
            model=ViLTransformerSS(config).cpu().eval()
            print('CHECKPOINT',model.load_state_dict(ckpt['state_dict'],strict=True),flush=True)
            del ckpt
            preprocess=model.preprocess
        with torch.inference_mode():
            for i,case in enumerate(selected,1):
                name=case['图像编号'];idx,img,hidden=data[name];image=preprocess(img).unsqueeze(0)
                texts=[case['文本1'],case['文本2']]
                if method=='CLIP':
                    visual=model.encode_image(image,torch.zeros((1,16,1024)));textual=model.encode_text(clip.tokenize(texts))
                    visual=visual/visual.norm(dim=-1,keepdim=True);textual=textual/textual.norm(dim=-1,keepdim=True)
                else:
                    visual=model.img_embeds(image,agent_hidden=hidden)['image_feats']
                    textual=model.txt_embeds(clip.tokenize(texts,context_length=77-model.prompt_length))['text_feats']
                scores=(visual@textual.T).flatten()
                if not bool(torch.isfinite(scores).all()):raise ValueError(f'non-finite score: {method} {name}')
                s1,s2=scores.tolist();correct_position=int(case['正确文本位置'])
                correct_score=(s1,s2)[correct_position-1];wrong_score=(s1,s2)[2-correct_position]
                results.append({'model':method,'image':name,'split':case['数据来源'],'arrow_row':idx,
                                'object':case['目标'],'actual_count':int(case['实际数量']),
                                'incorrect_count':int(case['错误数量']),'wording':case['句式'],
                                'text1':texts[0],'text2':texts[1],'correct_text_position':correct_position,
                                'score1':s1,'score2':s2,'correct_score':correct_score,'wrong_score':wrong_score,
                                'margin':correct_score-wrong_score,'correct':correct_score>wrong_score,
                                'tie':correct_score==wrong_score,'original_first_caption':case['原始第一条描述']})
                print('SCORED',method,i,'/',len(selected),name,case['句式'],flush=True)
        del model
    if len(results)!=2*len(selected):raise ValueError('incomplete results')
    with (out/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    meta['status']='complete';meta['scored_pairs']=len(results)
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('FINISHED',len(results),'model-image-wording comparisons',flush=True)

if __name__=='__main__':
    main()

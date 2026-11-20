"""Frozen retrieval scores on the locked UCM validation contrasts."""
import argparse
from pathlib import Path
from environment import ROOT, PREV, BASE, ASSETS, PLAIN_CKPT, read, dump, digest, val_data, image_for
import torch
from prompt_clip import clip
from run_frozen import MyBackboneFinetuning
from vilt.modules.vilt_module import ViLTransformerSS

CLIP_PATH=ASSETS/'models/ViT-L-14.pt'
weights={
    'CLIP':(CLIP_PATH,'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'),
    'CUP_CCB':(PREV/'models/ucm_ccb_clustering_prior.ckpt','688126e8587d7d91e0b93ed131a0095c2923904775e0cf9efb6f3a4e044ec607'),
    'ASPF':(PLAIN_CKPT,'e36801d50807519273408e4caf18604ad2f43db2c7b520cee98675631eac9a59'),
    'AGSE':(ASSETS/'ucm_main_domain_tool.ckpt','09ef20e0b354efbf8a7d4d49105cd92e1ae9412b3ea58820961265a065a874dc')}
methods=['CLIP','CUP_CCB','ASPF','AGSE','AGSE_archived','AGSE_scene']

def main(method):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    for name, sha in read(ROOT/'inputs/lock.json').items():
        assert digest(ROOT/name)==sha, name
    path,sha=weights['AGSE' if method.startswith('AGSE') else method]
    assert digest(path)==sha
    load=clip.load
    def from_disk(name,device='cpu',jit=False,download_root=None):
        return load(str(CLIP_PATH if name=='ViT-L/14' else name),device='cpu',jit=jit,download_root=download_root)
    clip.load=from_disk
    if method=='CLIP':
        model, preprocess=from_disk(CLIP_PATH)
        context=77
    else:
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        cfg=dict(saved['hyper_parameters']['config'])
        cfg.update(load_path='',test_only=False)
        model=ViLTransformerSS(cfg)
        model.load_state_dict(saved['state_dict'],strict=True)
        assert model.use_agent_prior==(method!='CUP_CCB')
        del saved
        preprocess=model.preprocess
        context=77-model.prompt_length
    model=model.float().eval().requires_grad_(False).cuda()
    cases=read(ROOT/'inputs/cases.json')
    names=sorted({r['image'] for r in cases})
    images={r['image']:r for r in read(ROOT/'inputs/candidates.json')}
    texts=sorted({r[k] for r in cases for k in ('positive','generic','negative')})
    tokens=clip.tokenize(texts,context_length=context,truncate=False)
    assert len(set(map(tuple,tokens.tolist())))==len(texts)
    data=val_data()
    vectors=[]
    sources={}
    with torch.inference_mode():
        parts=[]
        for t in tokens.split(32):
            t=t.cuda()
            out=torch.nn.functional.normalize(model.encode_text(t),dim=1) if method=='CLIP' else model.txt_embeds(t)['text_feats']
            parts.append(out.cpu())
        textual=torch.cat(parts)
        for start in range(0,len(names),8):
            group=names[start:start+8]
            im=torch.stack([preprocess(image_for(images[n],data)) for n in group]).cuda()
            if method=='CLIP':
                no_prompt=torch.empty(len(group),0,1024,device='cuda')
                out=torch.nn.functional.normalize(model.encode_image(im,no_prompt),dim=1)
            elif method=='CUP_CCB':
                out=model.img_embeds(im)['image_feats']
            else:
                prior=[]
                for n in group:
                    if method=='AGSE_archived':
                        file=BASE.parent/'ucm_prior_archive/agent_cache_domain_tool/ucm_caption_karpathy_label_val'/(str(images[n]['row'])+'.pt')
                    else:
                        file=ROOT/'priors'/method/(Path(n).stem+'.pt')
                        proof=read(file.with_suffix('.json'))
                        assert proof['rgb_sha256']==images[n]['rgb_sha256'] and proof['method']==method and proof['sha256']==digest(file)
                    sources[n]=digest(file)
                    prior.append(torch.load(file,map_location='cpu',weights_only=True).reshape(3584))
                out=model.img_embeds(im,agent_hidden=torch.stack(prior).float().cuda())['image_feats']
            vectors.append(out.cpu())
    visual=torch.cat(vectors)
    for bank in (textual,visual):
        assert torch.isfinite(bank).all() and (bank.norm(dim=1)-1).abs().max()<1e-5
    output=ROOT/'outputs'/method
    output.mkdir(parents=True,exist_ok=True)
    bank=output/'features.pt'
    torch.save(dict(images=names,texts=texts,image_features=visual,text_features=textual,tokens=tokens),bank)
    sims=visual.double()@textual.double().T
    rows=[]
    for r in cases:
        triple={k:float(sims[names.index(r['image']),texts.index(r[k])]) for k in ('positive','generic','negative')}
        rows.append(dict(**r,scores=triple))
    dump(output/'rows.json',rows)
    dump(output/'settings.json',dict(method=method,sha256=sha,weight_path=str(path),device=torch.cuda.get_device_name(),
                                    torch=torch.__version__,dtype='float32',lock_sha256=digest(ROOT/'inputs/lock.json'),
                                    features_sha256=digest(bank),priors_sha256=sources))
    print('DONE',method,len(names),len(texts),len(cases),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('method',choices=methods)
    main(parser.parse_args().method)

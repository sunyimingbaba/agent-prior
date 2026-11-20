"""Score locked captions against frozen embeddings and export inputs for offline verification."""
from common import BASE, ROOT, ASSETS, PLAIN_CKPT, QUERY, read, dump, digest, tables, image_for
import csv
import sys
import torch
from prompt_clip import clip
from run_frozen import MyBackboneFinetuning
from vilt.modules.vilt_module import ViLTransformerSS

method=sys.argv[1]
clip_path=ASSETS/'models/ViT-L-14.pt'
ckpts={'CUP_CCB':(ROOT/'models/ucm_ccb_clustering_prior.ckpt','688126e8587d7d91e0b93ed131a0095c2923904775e0cf9efb6f3a4e044ec607'),
       'ASPF':(PLAIN_CKPT,'e36801d50807519273408e4caf18604ad2f43db2c7b520cee98675631eac9a59'),
       'AGSE':(ASSETS/'ucm_main_domain_tool.ckpt','09ef20e0b354efbf8a7d4d49105cd92e1ae9412b3ea58820961265a065a874dc')}
ckpts['AGSE_archived']=ckpts['AGSE']
assert method=='AGSE_archived'
torch.set_num_threads(4);torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cases=read(ROOT/'inputs/cases.json');images=read(ROOT/'inputs/scored_images.json')
original=clip.load
def load_local(name,device='cpu',jit=False,download_root=None):
    path=clip_path if name=='ViT-L/14' else name
    return original(str(path),device='cpu',jit=jit,download_root=download_root)
clip.load=load_local
meta={'method':method,'scoring_device':'cuda_fp32','cases_sha256':digest(ROOT/'inputs/cases.json'),
      'scored_images_sha256':digest(ROOT/'inputs/scored_images.json'),'torch':torch.__version__}
if method in ckpts:
    path,h=ckpts[method];assert digest(path)==h
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    cfg=dict(saved['hyper_parameters']['config']);cfg.update(load_path='',test_only=False)
    model=ViLTransformerSS(cfg)
    model.load_state_dict(saved['state_dict'],strict=True)
    assert model.use_agent_prior
    meta.update(checkpoint=str(path),checkpoint_sha256=h,epoch=saved.get('epoch'),global_step=saved.get('global_step'),agent_prior=model.use_agent_prior)
    del saved
    preprocess=model.preprocess
else:
    h=digest(clip_path);assert h=='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
    model,preprocess=load_local(clip_path)
    meta.update(checkpoint=str(clip_path),checkpoint_sha256=h,visual_prompt_tokens=0 if method=='CLIP' else 16)
model=model.float().eval().requires_grad_(False).cuda()
texts=sorted({row[field] for row in cases for field in ('positive','negative')})
names=[r['image'] for r in images]
data=tables();visual=[];tools_used=0
with torch.inference_mode():
    context=77-model.prompt_length if method in ckpts else 77
    tokens=clip.tokenize(texts,context_length=context,truncate=False)
    assert len({tuple(v) for v in tokens.tolist()})==len(texts)
    embeds=[]
    for t in tokens.split(32):
        t=t.cuda()
        ft=model.txt_embeds(t)['text_feats'] if method in ckpts else torch.nn.functional.normalize(model.encode_text(t),dim=1)
        embeds.append(ft.cpu())
    textual=torch.cat(embeds)
    for i in range(0,len(images),8):
        group=images[i:i+8]
        im=torch.stack([preprocess(image_for(r,data)) for r in group]).cuda()
        hs=[]
        for rec in group:
            folder=BASE.parent/'ucm_prior_archive/agent_cache_domain_tool'/('ucm_caption_karpathy_label_'+rec['split'])
            path=folder/(str(rec['row'])+'.pt')
            hs.append(torch.load(path,map_location='cpu',weights_only=True).reshape(3584))
        ft=model.img_embeds(im,agent_hidden=torch.stack(hs).float().cuda())['image_feats']
        visual.append(ft.cpu())
visual=torch.cat(visual)
assert torch.isfinite(visual).all() and torch.isfinite(textual).all()
assert (visual.norm(dim=1)-1).abs().max()<1e-5 and (textual.norm(dim=1)-1).abs().max()<1e-5
scores=visual.double()@textual.double().T
rows=[]
for item in cases:
    i=names.index(item['image'])
    pos=float(scores[i,texts.index(item['positive'])]);neg=float(scores[i,texts.index(item['negative'])])
    rows.append(dict(**item,model=method,positive_cos=pos,negative_cos=neg,margin=pos-neg,correct=pos>neg,tie=pos==neg))
path=ROOT/'outputs'/method;path.mkdir(parents=True,exist_ok=True)
torch.save(dict(image_names=names,texts=texts,image_features=visual,text_features=textual,tokens=tokens),path/'features.pt')
dump(path/'rows.json',rows);dump(path/'settings.json',{**meta,'images':len(names),'pairs':len(rows),'unique_texts':len(texts),'images_with_tool_use':None,'prior_source':'archived original first-caption cache','feature_sha256':digest(path/'features.pt')})
print('SCORED',method,len(rows),flush=True)

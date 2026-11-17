"""One frozen readout for proxy-ranking, paired-count and hard-count rewards."""
import io
import hashlib
from pathlib import Path
import numpy as np
import pyarrow as pa
from PIL import Image
import torch
from milestone_common import ROOT,ASSETS,private_clip,setup
from matched_common_v3 import load_initial,install_arm,state_hash
from checkpoint_metadata import load_checkpoint
from stage_io import read,dump,sha

class RewardScorer:
 def __init__(self,prepare=False):
  setup();self.plan=read(ROOT/'plan.json');self.cohort=read(ROOT/'inputs/cohort.json');self.cases={x['image']:x for x in read(ROOT/'inputs/quantity_cases.json')};self.gallery=read(ROOT/'inputs/quantity_gallery.json');self.byname={x['image']:x for x in self.cohort};self.tables={s:pa.ipc.open_file(str(ASSETS/f'ucm/ucm_caption_karpathy_label_{s}.arrow')).read_all() for s in ['train','val']}
  cp=self.plan['readout']['checkpoint'];path=Path(self.plan['matched_root'])/cp['path'];assert sha(path)==cp['sha256'];self.model,_=load_initial();saved=load_checkpoint(path);missing=self.model.load_state_dict(saved['state_dict'],strict=False);assert not missing.unexpected_keys and all(k.startswith('model.') for k in missing.missing_keys)
  del saved;self.model.eval().requires_grad_(False).cuda();self.handles=install_arm(self.model,'calibrated_own');self.hash_before=state_hash(self.model.state_dict().items());self.images={}
  path=ROOT/'assets/scoring_assets.pt'
  if prepare:
   assert not path.exists();clip=private_clip();qtokens=clip.tokenize(self.gallery['quantity_text_order'],context_length=61,truncate=True)
   with torch.no_grad():qtext=self.model.txt_embeds(qtokens.cuda())['text_feats'].cpu()
   features=[];names=[]
   for start in range(0,1680,32):
    inputs=torch.stack([self.model.preprocess(self.image('train',i)) for i in range(start,min(start+32,1680))]).cuda()
    with torch.no_grad():f=self.model.model.encode_image(inputs,torch.zeros(len(inputs),16,1024,device='cuda',dtype=inputs.dtype));f=f/f.norm(dim=1,keepdim=True)
    features.append(f.cpu());names.extend(self.tables['train']['path'][start:min(start+32,1680)].to_pylist())
   path.parent.mkdir(exist_ok=True);torch.save({'proxy_train_features':torch.cat(features),'proxy_train_names':names,'quantity_text_features':qtext,'quantity_tokens':qtokens},path)
   dump(ROOT/'assets/scoring_assets.json',{'sha256':sha(path),'proxy_rows':1680,'quantity_texts':56,'readout_checkpoint_sha256':cp['sha256'],'train_arrow_sha256':sha(ASSETS/'ucm/ucm_caption_karpathy_label_train.arrow'),'readout_state_hash':self.hash_before,'proxy_library_generation':'fixed CLIP encode_image(real image, zero16x1024prompt), unit normalized; noAgentprior'})
  meta=read(ROOT/'assets/scoring_assets.json');assert sha(path)==meta['sha256'] and meta['readout_state_hash']==self.hash_before
  self.assets=torch.load(path,map_location='cpu',weights_only=True);self.proxy=self.assets['proxy_train_features'].cuda();self.qtext=self.assets['quantity_text_features'].cuda()
  expected=read(ROOT/'inputs/dataset_sources_reference.json')['train'];assert self.tables['train'].num_rows==expected['images']==1680 and meta['train_arrow_sha256']==expected['arrow_sha256']==sha(ASSETS/'ucm/ucm_caption_karpathy_label_train.arrow')
  names=self.tables['train']['path'].to_pylist();assert len(set(names))==1680 and self.assets['proxy_train_names']==names
  assert self.proxy.shape[0]==1680 and torch.isfinite(self.proxy).all() and torch.max(torch.abs(self.proxy.norm(dim=1)-1))<1e-5
  self.frozen_offset=torch.load(ROOT/'inputs/frozen_path_assets.pt',map_location='cpu',weights_only=True)['offset'].reshape(-1)
  self.verify_offset()
  signature={'readout_state_hash':self.hash_before,'fixed_offset_numeric_sha256':hashlib.sha256(self.frozen_offset.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),'train_arrow_sha256':expected['arrow_sha256'],'scoring_assets_sha256':sha(path),'all1680_unique_names_order_bound':True}
  assert len(self.proxy)==1680 and len(self.qtext)==56
  self.words={w:i+1 for i,w in enumerate('one two three four five six seven'.split())};self.last_tensors=None
  names_sha=hashlib.sha256(__import__('json').dumps(names,separators=(',',':')).encode()).hexdigest();assert names_sha=='4157799a7d5bb0d42462127267ecabfe0ca3882e175c91a49756617b1edc3d11'
  signature['ordered_train_names_sha256']=names_sha
  contract=ROOT/'assets/scoring_contract_v2.json'
  if contract.exists():assert read(contract)==signature
  else:dump(contract,signature)
 def verify_offset(self):
  captured=[]
  for fn in self.model.agent_proj[1]._forward_pre_hooks.values():
   for cell in fn.__closure__ or []:
    value=cell.cell_contents
    if isinstance(value,torch.Tensor) and value.numel()==1024:captured.append(value.detach().cpu().reshape(-1))
  assert len(captured)==1 and torch.equal(captured[0],self.frozen_offset),'calibration closure differs from frozen input'
 def image(self,split,row):
  return Image.open(io.BytesIO(self.tables[split]['image'][row].as_py())).convert('RGB')
 def case_image(self,c):
  assert self.tables[c['split']]['path'][c['row']].as_py()==c['image'];return self.image(c['split'],c['row'])
 @torch.no_grad()
 def score(self,h,c,strict_completed):
  assert h.shape==(3584,) and h.dtype==torch.bfloat16 and torch.isfinite(h).all()
  if c['image'] not in self.images:self.images[c['image']]=self.model.preprocess(self.case_image(c)).unsqueeze(0).cuda()
  hidden=h.unsqueeze(0).cuda();visual=self.model.img_embeds(self.images[c['image']],agent_hidden=hidden)['image_feats'];qs=(visual@self.qtext.T).cpu().flatten();fmt=.2 if strict_completed else 0.;result={'r_format':fmt,'r_proxy_rank':None,'r_quantity_pair':None,'r_quantity_hard':None,'proxy_rank0':None};tensors={'image_feature':visual.cpu(),'quantity_scores':qs}
  if c['split']=='train':
   prior=self.model.agent_proj(hidden.float()).unsqueeze(1).expand(-1,16,-1);ratio=self.model.cluster_mapping(prior).clamp(0,1);learned=self.model.visual_prompt_proj(self.model.visual_prompt_embeddings).expand(1,-1,-1);vp=self.model.visual_prompt_proj_gather((1-ratio)*prior+ratio*learned)
   pf=self.model.model.encode_image(torch.zeros(1,3,224,224,device='cuda'),vp);pf=pf/pf.norm(dim=1,keepdim=True);ps=(pf@self.proxy.T).cpu().flatten();assert self.assets['proxy_train_names'][c['row']]==c['image'];rank=int((ps>ps[c['row']]).sum());result.update(proxy_rank0=rank,r_proxy_rank=fmt+1/(1+rank),proxy_retrieval=1/(1+rank));tensors.update(proxy_feature=pf.cpu(),proxy_scores=ps)
  if c['image'] in self.cases:
   gold=self.cases[c['image']]
   for wording in ['A','B']:
    pos,neg=self.gallery['quantity_columns'][c['image']][wording];prefix=self.gallery['quantity_text_order'][pos].rsplit(' ',1)[0];wrong=[i for i,s in enumerate(self.gallery['quantity_text_order']) if s.rsplit(' ',1)[0]==prefix and self.words[s.rsplit(' ',1)[1][:-1]]!=gold['actual_count']];positive=float(qs[pos]);pair=float(qs[pos])-float(qs[neg]);hard=positive-max(float(qs[i]) for i in wrong)
    result.update({wording+'_positive':positive,wording+'_negative':float(qs[neg]),wording+'_pair_margin':pair,wording+'_hard_margin':hard,wording+'_pair_correct':pair>0,wording+'_hard_correct':hard>0})
    if wording=='A':result.update(r_quantity_pair=fmt+pair,r_quantity_hard=fmt+hard)
  self.last_tensors=tensors;return result
 def verify_frozen(self):
  self.verify_offset()
  assert state_hash(self.model.state_dict().items())==self.hash_before and all(not p.requires_grad and p.grad is None for p in self.model.parameters());return {'ok':True,'state_hash':self.hash_before,'checkpoint_sha256':self.plan['readout']['checkpoint']['sha256']}

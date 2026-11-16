"""Frozen val/quantity evaluation; preserve real retrieval images for every prior control."""
import gc
import hashlib
import io
import time
from pathlib import Path
import numpy as np
import pyarrow as pa
from PIL import Image
import torch
from milestone_common import ROOT,PLAN,ASSETS,setup,private_clip
from matched_common_v3 import load_initial,install_arm,state_hash,retrieval_metrics
from checkpoint_metadata import load_checkpoint
from stage_io import read,dump,sha
from counterfactual_rollout import tensor_sha

setup();started=time.monotonic();p=Path(PLAN['matched_parent_root']);d=read(ROOT/'diagnostic_plan.json');ep=read(ROOT/'evaluation_plan.json')
for n,h in read(ROOT/'evaluation_code_hashes.json').items():assert sha(ROOT/n)==h,n
assert read(ROOT/'generation_job_result.json')['status']=='complete'
records={(x['condition'],x['split'],x['row']):x for x in read(ROOT/'generation/records.json')}
perms=read(ROOT/'inputs/permutations.json');cohort=read(ROOT/'inputs/cohort.json');cases=read(ROOT/'inputs/quantity_cases.json');qg=read(ROOT/'inputs/quantity_gallery.json')
ref={(x['split'],x['row']):x for x in read(p/'cache/records.json') if x['split'] in ['train','val']}
dump(ROOT/'inputs/parent_development_cache_reference.json',list(ref.values()))
tables={s:pa.ipc.open_file(str(ASSETS/f'ucm/ucm_caption_karpathy_label_{s}.arrow')).read_all() for s in ['train','val']}
def image(split,row):return Image.open(io.BytesIO(tables[split]['image'][row].as_py())).convert('RGB')
def hidden(label,split,row):
 if label.startswith('permutation_'):
  mapped=perms[split][int(label[-2:])][row];x=ref[split,mapped];path=p/x['state_file']
 else:
  mapped=row;x=records[label,split,row];path=ROOT/x['state_file']
 assert sha(path)==x['state_file_sha256'];h=torch.load(path,map_location='cpu',weights_only=True)
 assert tensor_sha(h)==x['numeric_state_sha256'];return h,{'split':split,'target_row':row,'prior_source_row':mapped,'source_file':str(path),'state_file_sha256':x['state_file_sha256'],'numeric_state_sha256':x['numeric_state_sha256']}
lock=read(ROOT/'inputs/parent_selection_lock.json');summaries={};updates=0;encodings=0
for arm,labels in ep['readout_conditions'].items():
 model,_=load_initial();cp=lock['arms'][arm]['checkpoint'];assert sha(p/cp['path'])==cp['sha256']
 saved=load_checkpoint(p/cp['path']);assert saved['matched_readout']['arm']==arm
 missing=model.load_state_dict(saved['state_dict'],strict=False)
 assert not missing.unexpected_keys and all(n.startswith('model.') for n in missing.missing_keys)
 del saved;model.eval().requires_grad_(False).cuda();before=state_hash(model.state_dict().items());handles=install_arm(model,arm)
 captures={'projected':[],'cluster_output_after_arm_hooks':[]}
 hooks=[model.agent_proj.register_forward_hook(lambda m,a,o:captures['projected'].append(o.detach().cpu())),model.cluster_mapping.register_forward_hook(lambda m,a,o:captures['cluster_output_after_arm_hooks'].append(o.detach().cpu()))]
 val_images=torch.stack([model.preprocess(image('val',i)) for i in range(210)])
 qimages=torch.stack([model.preprocess(image(c['split'],c['row'])) for c in cases])
 captions=[];owners=[];names=[]
 for i in range(210):
  names.append(tables['val']['path'][i].as_py());cs=tables['val']['caption'][i].as_py();captions.extend(cs);owners.extend([i]*len(cs))
 clip=private_clip();tokens=clip.tokenize(captions,context_length=61,truncate=True);qtokens=clip.tokenize(qg['quantity_text_order'],context_length=61,truncate=True)
 groupmap={};groups=[]
 for xs in tokens.tolist():
  k=tuple(xs)
  if k not in groupmap:groupmap[k]=len(groupmap)
  groups.append(groupmap[k])
 gallery={'names':names,'captions':captions,'owners':owners,'groups':groups,'split':'val','arrow_sha256':sha(ASSETS/'ucm/ucm_caption_karpathy_label_val.arrow')}
 if not (ROOT/'evaluation/gallery.json').exists():dump(ROOT/'evaluation/gallery.json',gallery)
 else:assert read(ROOT/'evaluation/gallery.json')==gallery
 with torch.no_grad():
  text=torch.cat([model.txt_embeds(x.cuda())['text_feats'].cpu() for x in tokens.split(64)])
  qtext=model.txt_embeds(qtokens.cuda())['text_feats'].cpu();scale=float(model.model.logit_scale.exp().cpu())
 baseline_features=None;baseline_qfeatures=None;summaries[arm]={}
 for label in labels:
  out=ROOT/'evaluation'/arm/label;assert not out.exists();out.mkdir(parents=True)
  hs=[];sources=[]
  for row in range(210):
   h,s=hidden(label,'val',row);hs.append(h);sources.append(s)
  hs=torch.stack(hs);captures['projected'].clear();captures['cluster_output_after_arm_hooks'].clear()
  with torch.no_grad():
   features=torch.cat([model.img_embeds(val_images[a:a+32].cuda(),agent_hidden=hs[a:a+32].cuda())['image_feats'].cpu() for a in range(0,210,32)])
  encodings+=210
  path={k:torch.cat(v) for k,v in captures.items()};path['gate_clamped']=path['cluster_output_after_arm_hooks'].clamp(0,1)
  assert path['projected'].shape[0]==210 and path['gate_clamped'].shape==(210,16,1)
  scores=scale*features@text.T;metrics,hit=retrieval_metrics(scores,gallery)
  torch.save({**hit,'image_features':features,'text_features':text,'scores':scores,'tokens':tokens,'owners':torch.tensor(owners),'groups':torch.tensor(groups),'logit_scale':scale,'prior_hidden':hs,**path},out/'retrieval.pt')
  dump(out/'metrics.json',metrics);dump(out/'state_sources.json',sources)
  if label==ep['baseline']:
   baseline_features=features;old=torch.load(p/f"training/{arm}/validation/epoch_{lock['arms'][arm]['epoch']:02d}/tensors.pt",map_location='cpu',weights_only=True)
   assert torch.equal(features,old['image_features']) and torch.equal(text,old['text_features']),arm
   assert abs(metrics['mR']-read(p/f"training/{arm}/validation/epoch_{lock['arms'][arm]['epoch']:02d}/metrics.json")['mR'])<1e-12
  captures['projected'].clear();captures['cluster_output_after_arm_hooks'].clear();qhs=[];qsources=[]
  for c in cases:
   h,s=hidden(label,c['split'],c['row']);qhs.append(h);qsources.append(s)
  qhs=torch.stack(qhs)
  with torch.no_grad():
   qfeatures=torch.cat([model.img_embeds(qimages[i:i+1].cuda(),agent_hidden=qhs[i:i+1].cuda())['image_feats'].cpu() for i in range(85)])
  encodings+=85;qscore=qfeatures@qtext.T
  torch.save({'image_features':qfeatures,'text_features':qtext,'scores':qscore,'tokens':qtokens,'prior_hidden':qhs,**{k:torch.cat(v) for k,v in captures.items()}},out/'quantity.pt')
  dump(out/'quantity_state_sources.json',qsources)
  if label==ep['baseline']:
   baseline_qfeatures=qfeatures;old=torch.load(p/f'quantity/{arm}/tensors.pt',map_location='cpu',weights_only=True)
   assert torch.equal(qfeatures,old['image_features']) and torch.equal(qtext,old['text_features']),arm
  if arm in d['negative_control_readouts']:
   assert torch.equal(features,baseline_features) and torch.equal(qfeatures,baseline_qfeatures),(arm,label)
  summaries[arm][label]=metrics
  dump(ROOT/'evaluation_progress.json',{'arm':arm,'condition':label,'completed':sum(len(x) for x in summaries.values()),'total':sum(map(len,ep['readout_conditions'].values())),'image_encodings':encodings,'elapsed_seconds':time.monotonic()-started})
  print('FROZEN_DIAGNOSTIC_EVALUATED',arm,label,metrics['mR'],flush=True)
 assert state_hash(model.state_dict().items())==before,arm
 dump(ROOT/'evaluation'/arm/'model_immutability.json',{'ok':True,'model_state_hash_before_and_after':before,'checkpoint_sha256':cp['sha256'],'all_parameters_require_grad_false':all(not q.requires_grad for q in model.parameters()),'optimizer_created':False})
 for h in hooks+handles:h.remove()
 del model;gc.collect();torch.cuda.empty_cache()
dump(ROOT/'evaluation_result.json',{'ok':True,'all_readout_conditions':summaries,'forward_conditions':sum(map(len,summaries.values())),'retrieval_images_per_condition':210,'quantity_images_per_condition':85,'image_encodings':encodings,'all_four_parent_baselines_exact':True,'negative_control_features_exactly_invariant':True,'Agent_optimizer_updates':0,'readout_optimizer_updates':0,'test_data_opened':False,'elapsed_seconds':time.monotonic()-started})
print('ALL_FROZEN_DEPENDENCE_EVALUATIONS_COMPLETE',flush=True)

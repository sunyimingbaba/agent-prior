"""Compact frozen evaluation: global text/prior banks, all source maps and original GT."""
import gc
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

setup();d=read(ROOT/'protocol_plan.json');ep=read(ROOT/'evaluation_plan.json');started=time.monotonic();parent=Path(d['diagnostic_parent']);matched=Path(d['matched_parent'])
for n,h in read(ROOT/'evaluation_code_hashes.json').items():assert sha(ROOT/n)==h,n
assert read(ROOT/'generation_job_result_v2.json')['status']=='complete'
gresult=read(ROOT/'generation_result_v2.json');selection=read(ROOT/'prompt_selection_lock_v2.json');assert selection['chosen_prompt']==gresult['prompt_selection']['chosen_prompt']
# Reject any observed dangling tool closer before utility computation. Retain all raw evidence.
for c in read(ROOT/'generation_v2/records.json'):
 tr=read(ROOT/c['trace_file'])
 if tr['strict_protocol_usable']:
  final=tr['rounds'][-1]['generated_text'];assert '</tool_call>' not in final and '<tool_call>' not in final,(c['key'],'dangling tool tag in claimed strict answer')
perms=read(ROOT/'inputs/permutations.json');qcases=read(ROOT/'inputs/quantity_cases.json');qgallery=read(ROOT/'inputs/quantity_gallery.json');lock=read(ROOT/'inputs/parent_selection_lock.json')
banks={f:torch.load(ROOT/'state_banks'/(f+'.pt'),map_location='cpu',weights_only=True) for f in d['state_families']};bankmeta={f:read(ROOT/'state_banks'/(f+'.json')) for f in banks}
positions={s:{x['image']:i for i,x in enumerate(bankmeta['legacy_last'][s])} for s in ['train','val']};shared_index=positions['train']['102.tif']
for f in banks:
 for s,n in [('train',63),('val',210)]:assert banks[f][s].shape==(n,3584) and [x['image'] for x in bankmeta[f][s]]==[x['image'] for x in bankmeta['legacy_last'][s]]
def source(condition,split,index):
 if condition=='correct':return split,index
 if condition=='shared_training102':return 'train',shared_index
 return split,perms[split][int(condition[-2:])][index]
def sources_for(family,condition):
 val=[source(condition,'val',i) for i in range(210)];q=[source(condition,c['split'],positions[c['split']][c['image']]) for c in qcases]
 return val,q
def stacked(family,sources):return torch.stack([banks[family][s][i] for s,i in sources])
tables={s:pa.ipc.open_file(str(ASSETS/f'ucm/ucm_caption_karpathy_label_{s}.arrow')).read_all() for s in ['train','val']}
def image(s,row):return Image.open(io.BytesIO(tables[s]['image'][row].as_py())).convert('RGB')
captions=[];owners=[];names=[]
for i in range(210):
 names.append(tables['val']['path'][i].as_py());texts=tables['val']['caption'][i].as_py();captions.extend(texts);owners.extend([i]*len(texts))
summaries={};encoded=0;reused=0;completed=0
for arm,cases in ep['readout_cases'].items():
 model,_=load_initial();cp=lock['arms'][arm]['checkpoint'];assert sha(matched/cp['path'])==cp['sha256'];saved=load_checkpoint(matched/cp['path']);missing=model.load_state_dict(saved['state_dict'],strict=False);assert not missing.unexpected_keys and all(k.startswith('model.') for k in missing.missing_keys)
 del saved;model.eval().requires_grad_(False).cuda();before=state_hash(model.state_dict().items());handles=install_arm(model,arm)
 clip=private_clip();tokens=clip.tokenize(captions,context_length=61,truncate=True);qtokens=clip.tokenize(qgallery['quantity_text_order'],context_length=61,truncate=True)
 groups=[];ids={}
 for row in tokens.tolist():
  key=tuple(row)
  if key not in ids:ids[key]=len(ids)
  groups.append(ids[key])
 gallery={'names':names,'captions':captions,'owners':owners,'groups':groups,'split':'val','arrow_sha256':sha(ASSETS/'ucm/ucm_caption_karpathy_label_val.arrow')}
 if (ROOT/'evaluation/gallery.json').exists():assert read(ROOT/'evaluation/gallery.json')==gallery
 else:dump(ROOT/'evaluation/gallery.json',gallery)
 valimages=torch.stack([model.preprocess(image('val',i)) for i in range(210)]);qimages=torch.stack([model.preprocess(image(c['split'],c['row'])) for c in qcases])
 with torch.no_grad():
  text=torch.cat([model.txt_embeds(x.cuda())['text_feats'].cpu() for x in tokens.split(64)]);qtext=model.txt_embeds(qtokens.cuda())['text_feats'].cpu();scale=float(model.model.logit_scale.exp().cpu())
 armout=ROOT/'evaluation'/arm;armout.mkdir(parents=True,exist_ok=True);torch.save({'text_features':text,'quantity_text_features':qtext,'tokens':tokens,'quantity_tokens':qtokens,'owners':torch.tensor(owners),'groups':torch.tensor(groups),'logit_scale':scale},armout/'assets.pt')
 for family in banks:
  h=torch.cat([banks[family]['train'],banks[family]['val']]).float().cuda()
  with torch.no_grad():
   p=model.agent_proj(h);g=model.cluster_mapping(p[:,None,:].expand(-1,16,-1))
  folder=armout/'bank_paths';folder.mkdir(exist_ok=True);torch.save({'projected':p.cpu(),'cluster_output_after_arm_hooks':g.cpu(),'gate_clamped':g.clamp(0,1).cpu(),'split_order':['train63','val210']},folder/(family+'.pt'))
 capture=[];handle=model.cluster_mapping.register_forward_hook(lambda m,a,o:capture.append(o.detach().cpu()));reference_f=reference_q=None;summaries[arm]={}
 for case in cases:
  family,condition=case['family'],case['condition'];vs,qs=sources_for(family,condition);hv=stacked(family,vs);hq=stacked(family,qs);out=armout/family/condition;assert not out.exists();out.mkdir(parents=True)
  reuse=bool(arm in d['evaluation']['primary_readouts'] and family=='legacy_last' and (condition=='correct' or condition.startswith('permutation_')))
  capture.clear();origin=None
  if reuse:
   oldlabel='with_example__real__recorded_real' if condition=='correct' else condition;path=parent/'evaluation'/arm/oldlabel/'retrieval.pt';old=torch.load(path,map_location='cpu',weights_only=True)
   assert torch.equal(hv,old['prior_hidden']) and torch.equal(text,old['text_features']);features=old['image_features'];scores=old['scores'];gates=old['cluster_output_after_arm_hooks'];metrics,hit=retrieval_metrics(scores,gallery)
   assert torch.equal(hit['top_i'],old['top_i']) and torch.equal(hit['top_t'],old['top_t']);origin={'path':str(path),'sha256':sha(path),'exact_numeric_reuse':True};reused+=210
  else:
   with torch.no_grad():features=torch.cat([model.img_embeds(valimages[a:a+32].cuda(),agent_hidden=hv[a:a+32].cuda())['image_feats'].cpu() for a in range(0,210,32)])
   gates=torch.cat(capture);scores=scale*features@text.T;metrics,hit=retrieval_metrics(scores,gallery);encoded+=210
  capture.clear()
  with torch.no_grad():qf=torch.cat([model.img_embeds(qimages[i:i+1].cuda(),agent_hidden=hq[i:i+1].cuda())['image_feats'].cpu() for i in range(85)])
  qg=torch.cat(capture);qscore=qf@qtext.T;encoded+=85
  if family=='legacy_last' and condition=='correct':
   old=torch.load(parent/'evaluation'/arm/'with_example__real__recorded_real'/'quantity.pt',map_location='cpu',weights_only=True);assert torch.equal(qf,old['image_features']) and torch.equal(qtext,old['text_features'])
   reference_f=features;reference_q=qf
  if arm in d['evaluation']['negative_readouts']:assert torch.equal(features,reference_f) and torch.equal(qf,reference_q)
  torch.save({'image_features':features,'scores':scores,'top_i':hit['top_i'],'top_t':hit['top_t'],'quantity_image_features':qf,'quantity_scores':qscore,'val_gate_raw':gates,'quantity_gate_raw':qg},out/'tensors.pt')
  dump(out/'metrics.json',metrics);dump(out/'sources.json',{'family':family,'condition':condition,'val':vs,'quantity':qs,'bank_file':str((ROOT/'state_banks'/(family+'.pt')).relative_to(ROOT)),'bank_file_sha256':sha(ROOT/'state_banks'/(family+'.pt')),'assets_file_sha256':sha(armout/'assets.pt'),'validation_reuse':origin})
  summaries[arm][family+'/'+condition]=metrics;completed+=1;dump(ROOT/'evaluation_progress.json',{'arm':arm,'family':family,'condition':condition,'completed':completed,'total':270,'new_image_encodings':encoded,'reused_validation_image_encodings':reused,'elapsed_seconds':time.monotonic()-started});print('STATE_READOUT',completed,arm,family,condition,metrics['mR'],flush=True)
 assert state_hash(model.state_dict().items())==before;dump(armout/'model_immutability.json',{'ok':True,'state_hash_before_and_after':before,'checkpoint_sha256':cp['sha256'],'all_parameters_frozen':all(not x.requires_grad for x in model.parameters()),'optimizer_created':False})
 for h in handles+[handle]:h.remove()
 del model;gc.collect();torch.cuda.empty_cache()
assert completed==270 and encoded+reused==79650
dump(ROOT/'evaluation_result.json',{'ok':True,'protocol_gate_passed':selection['protocol_gate_passed'],'readout_conditions':completed,'new_image_encodings':encoded,'reused_validation_image_encodings':reused,'input_source_rows':encoded+reused,'all_four_quantity_baselines_exact':True,'negative_features_exactly_invariant':True,'readout_optimizer_updates':0,'Agent_optimizer_updates':gresult['Agent_optimizer_updates'],'test_examples_scored':0,'elapsed_seconds':time.monotonic()-started})
print('ALL_COMPACT_STATE_EVALUATIONS_COMPLETE',flush=True)

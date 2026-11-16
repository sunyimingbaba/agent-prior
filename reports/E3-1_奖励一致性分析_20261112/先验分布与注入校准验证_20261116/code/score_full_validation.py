from pathlib import Path
import torch
from milestone_common import ROOT,PLAN,setup,seed,case_images
from quantity_model import load_final_model
from path_runtime_v2 import intervene,clear,model_hash,tensor_sha
from stage_io import read,dump,sha
setup();seed(0);out=ROOT/'full_validation';assert read(out/'generation_result.json')['ok'];cases=read(out/'cases.json');records=read(out/'records.json');by={x['image']:x for x in records};m,clip=load_final_model();before=model_hash(m);assert before==read(ROOT/'gpu_result.json')['original_model_parameter_hash'];images=case_images(cases);pt=torch.load(ROOT/'path/tensors.pt',map_location='cpu',weights_only=True);offset=pt['offsets']['balanced32'].cuda();shared=next(x for x in read(ROOT/'states/records.json') if x['family']=='balanced32' and x['image']=='102.tif' and x['role']=='greedy');sh=torch.load(ROOT/shared['state_file'],map_location='cpu',weights_only=True)
texts=[];owners=[]
for i,c in enumerate(cases):
 for s in c['captions']:texts.append(s);owners.append(i)
tokens=clip.tokenize(texts,context_length=61,truncate=True);groups={};gids=[]
for row in tokens.tolist():
 key=tuple(row)
 if key not in groups:groups[key]=len(groups)
 gids.append(groups[key])
text_feats=[]
with torch.no_grad():
 for token in tokens.split(64):text_feats.append(m.txt_embeds(token.cuda())['text_feats'].cpu())
T=torch.cat(text_feats);arms=['native','mean_shift_auto','learned_only','mean_shift_shared102'];features={};scores={};topk={};max_replay=0.
old=read(ROOT/'scoring/all_scores.json');oldnt=torch.load(ROOT/'scoring/tensors.pt',map_location='cpu',weights_only=True);oldix={x['id']:i for i,x in enumerate(old)}
with torch.no_grad():
 for arm in arms:
  fs=[]
  for c in cases:
   x=by[c['image']];h=sh if arm=='mean_shift_shared102' else torch.load(ROOT/x['state_file'],map_location='cpu',weights_only=True);variant={'native':'auto','mean_shift_auto':'mean_shift_auto','learned_only':'gate_1','mean_shift_shared102':'mean_shift_auto'}[arm];hooks=intervene(m,variant,offset)
   try:f=m.img_embeds(m.preprocess(images[c['image']]['image']).unsqueeze(0).cuda(),agent_hidden=h[None].cuda())['image_feats']
   finally:clear(hooks)
   fs.append(f.cpu())
   if c['image'] in {z['image'] for z in records if z['reused']} and arm in ['mean_shift_auto','learned_only','mean_shift_shared102']:
    role={'mean_shift_auto':'selected_own','learned_only':'learned_only','mean_shift_shared102':'selected_shared'}[arm];z=next(z for z in old if z['phase']=='confirmation' and z['role']==role and z['target_image']==c['image'] and z['family']=='balanced32');error=float((f.cpu()-oldnt['image_features'][oldix[z['id']]][None]).abs().max());max_replay=max(max_replay,error);assert error<1e-6
  F=torch.cat(fs);S=m.model.logit_scale.exp().cpu()*F@T.T;features[arm]=F;scores[arm]=S;topk[arm]={'i2t':S.topk(10,dim=1).indices,'t2i':S.topk(10,dim=0).indices};dump(ROOT/'full_validation_progress.json',{'phase':'scoring','arm':arm,'images_completed':210});print('FULL_VAL_ARM_COMPLETE',arm,flush=True)
assert model_hash(m)==before;torch.save({'text_features':T,'caption_tokens':tokens,'caption_owner':torch.tensor(owners),'caption_group':torch.tensor(gids),'image_features':features,'similarities':scores,'topk':topk,'logit_scale':float(m.model.logit_scale.exp())},out/'retrieval_tensors.pt');dump(out/'gallery.json',{'images':[c['image'] for c in cases],'captions':texts,'caption_owners':owners,'token_groups':gids,'unique_token_groups':len(groups),'reused_images':[x['image'] for x in records if x['reused']]});dump(out/'score_result.json',{'ok':True,'images':210,'captions':len(texts),'arms':arms,'image_encodings':840,'old22_feature_replay_max_error':max_replay,'model_hash_unchanged':before,'optimizer_updates':0});print('ALL_840_FULL_VAL_ENCODINGS_COMPLETE',flush=True)

"""Fixed final policies: fullval210 greedy plus63quantitytrain and110 heldout sampled trajectories."""
import json
from pathlib import Path
import re
import sys
import time
import numpy as np
from milestone_common import ROOT,ASSETS,setup,seed,load_policy,lora_hash,private_clip
from checkpoint_io import restore_adapter_exact
from stochastic_rollout import rollout,TOOL,ANSWER
from prompt_design import messages
from reward_model_v2 import RewardScorer
from matched_common_v3 import retrieval_metrics
from tie_metrics import expected_hits
from stage_io import read,dump,sha
torch=setup();p=read(ROOT/'plan.json');name=sys.argv[1];assert name in p['evaluation']['policies'];prompt=read(ROOT/'prompt_selection_lock.json')['selected_prompt'];assert prompt
if name=='baseline_continuation0':cp=Path(p['policy']['checkpoint']);policy_hash=p['policy']['lora_hash']
else:
 trained=read(ROOT/'training_v2'/name/'result.json');assert trained['ok'];cp=ROOT/trained['checkpoint'];policy_hash=trained['final_policy_hash']
seed(0);model,processor=load_policy(cp,False);restore_adapter_exact(model,cp);assert lora_hash(model)==policy_hash;scorer=RewardScorer();cohort=read(ROOT/'inputs/cohort.json');by={x['image']:x for x in cohort};cases=read(ROOT/'inputs/quantity_cases.json');holdout=[by[x['image']] for x in cases if x['split']=='train' and x['image'] not in p['training_images']];assert len(holdout)==55
out=ROOT/'evaluation'/name;out.mkdir(parents=True);records=[];start=time.monotonic();parent=Path(p['parent_protocol_root']);reuse=name=='baseline_continuation0' and read(ROOT/'baseline_replay_v2/result.json')['baseline_reuse'];old=read(parent/'generation_v2/records.json') if reuse else [];valfeatures={}
from agent.rollout import SYSTEM_PROMPT
def classify(tr):
 rounds=tr['rounds'];final=rounds[-1]['generated_text'];result=any('<result>' in z['generated_text'] or '</result>' in z['generated_text'] for z in rounds);m=ANSWER.fullmatch(final.strip());strict=bool(tr['stop_reason']=='answer' and not result and m and m.group(1).strip() and final.count('<answer>')==final.count('</answer>')==1 and '<tool_call>' not in final and '</tool_call>' not in final and all(not z['token_limit'] and 'addCriterion' not in z['generated_text'] for z in rounds) and all(TOOL.fullmatch(z['generated_text'].strip()) for z in rounds[:-1]));consumed=any(e['after_round']<len(rounds) for e in tr['tool_events']);usable=bool(strict and consumed and tr['tools_used']>0);tr.update(strict_protocol_usable=strict,strict_completed_consumed_tool=usable,feedback_consumed=consumed,generated_result_tag=result,quality_usable=usable);return tr
items=[(c,'greedy',0,None) for c in cohort]+[(c,'sampled',sample,2026111900+10*cohort.index(c)+sample) for c in holdout for sample in range(2)];assert len(items)==383
for ii,(c,mode,sample,ss) in enumerate(items):
 origin=None
 if reuse and mode=='greedy':
  group='pilot_initial/contract_only' if c['split']=='train' else 'selected_validation';ref=next(x for x in old if x['image']==c['image'] and x['group']==group);tr=classify(read(parent/ref['trace_file']));saved=torch.load(parent/ref['tensor_file'],map_location='cpu',weights_only=True);h=saved['candidates']['last'];tensors={'prior':h,'step_states':saved['step_states'],'probability_witness':{}};origin={'tensor_path':str(parent/ref['tensor_file']),'tensor_sha256':ref['tensor_sha256'],'trace_path':str(parent/ref['trace_file']),'trace_sha256':ref['trace_sha256'],'fresh_anchor_gate_sha256':sha(ROOT/'baseline_replay_v2/result.json')}
 else:
  if ss is not None:seed(ss)
  h,tensors,tr=rollout(model,processor,scorer.case_image(c),messages(prompt,c,SYSTEM_PROMPT),{**c['feedback'],'source_image':c['image'],'source_original_trace_sha256':c['feedback_original_trace_sha256']},p['rollout'],temperature=0. if mode=='greedy' else 1.)
  classify(tr)
 values=scorer.score(h,c,tr['strict_completed_consumed_tool']);folder=out/mode;folder.mkdir(exist_ok=True);stem=c['key']+f'_{sample:02d}';sf=folder/(stem+'.pt');tf=folder/(stem+'.json');torch.save({**tensors,**scorer.last_tensors},sf);dump(tf,tr)
 row={'policy':name,'policy_hash':policy_hash,'image':c['image'],'key':c['key'],'split':c['split'],'row':c['row'],'mode':mode,'sample':sample,'seed':ss,'strict_completed_consumed_tool':tr['strict_completed_consumed_tool'],'feedback_consumed':tr['feedback_consumed'],'tools_used':tr['tools_used'],'generated_result_tag':tr['generated_result_tag'],'stop_reason':tr['stop_reason'],**values,'tensor_file':str(sf.relative_to(ROOT)),'tensor_sha256':sha(sf),'trace_file':str(tf.relative_to(ROOT)),'trace_sha256':sha(tf),'reuse_source':origin};records.append(row)
 with (out/'records.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 if mode=='greedy' and c['split']=='val':valfeatures[c['row']]=scorer.last_tensors['image_feature'][0]
 dump(ROOT/'evaluation_progress.json',{'policy':name,'records':ii+1,'total':383,'greedy_reused':reuse,'elapsed_seconds':time.monotonic()-start})
 if (ii+1)%20==0:print('FIXED_POLICY_EVAL',name,ii+1,'/383',flush=True)
assert len(valfeatures)==210 and lora_hash(model)==policy_hash;dump(out/'records.json',records)
# Dataset-provided1050 captions form the fixed gallery; generation always saw only generic query.
val=scorer.tables['val'];captions=[];owners=[];names=[]
for i in range(210):
 names.append(val['path'][i].as_py());cs=val['caption'][i].as_py();captions.extend(cs);owners.extend([i]*len(cs))
assert len(captions)==1050;tokens=private_clip().tokenize(captions,context_length=61,truncate=True);groups=[];ids={}
for row in tokens.tolist():
 k=tuple(row)
 if k not in ids:ids[k]=len(ids)
 groups.append(ids[k])
gallery={'names':names,'captions':captions,'owners':owners,'groups':groups,'split':'val','arrow_sha256':sha(ASSETS/'ucm/ucm_caption_karpathy_label_val.arrow')}
if (ROOT/'evaluation/gallery.json').exists():assert read(ROOT/'evaluation/gallery.json')==gallery
else:dump(ROOT/'evaluation/gallery.json',gallery)
with torch.no_grad():text=torch.cat([scorer.model.txt_embeds(t.cuda())['text_feats'].cpu() for t in tokens.split(64)])
f=torch.stack([valfeatures[i] for i in range(210)]);scale=float(scorer.model.model.logit_scale.exp().cpu());scores=scale*f@text.T;original,hit=retrieval_metrics(scores,gallery);ti=hit['top_i'].numpy();tt=hit['top_t'].numpy();own=np.array(owners);grp=np.array(groups);summary={};contributions={}
for protocol in ['repository_duplicate_token_positive','strict_index','strict_tie_averaged']:
 pos=own[None,:]==np.arange(210)[:,None]
 if protocol=='repository_duplicate_token_positive':pos=np.stack([np.isin(grp,grp[own==i]) for i in range(210)])
 if protocol=='strict_tie_averaged':a,b=expected_hits(scores.numpy(),pos),expected_hits(scores.numpy().T,pos.T)
 else:a=np.stack([pos[np.arange(210)[:,None],ti[:,:k]].any(1) for k in [1,5,10]],1).astype(float);b=np.stack([pos[tt[:k],np.arange(1050)[None,:]].any(0) for k in [1,5,10]],1).astype(float)
 values=np.r_[a.mean(0),b.mean(0)];summary[protocol]={k:float(x) for k,x in zip(['I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10','mR'],np.r_[values,values.mean()])};contributions[protocol]={'i2t_hits':torch.tensor(a),'t2i_hits':torch.tensor(b),'image_cluster_mR':torch.tensor((a.sum(1)+b.reshape(210,5,3).mean(1).sum(1))/6)}
assert abs(summary['repository_duplicate_token_positive']['mR']-original['mR'])<1e-12
torch.save({'image_features':f,'text_features':text,'scores':scores,'top_i':hit['top_i'],'top_t':hit['top_t'],'tokens':tokens,'owners':torch.tensor(owners),'groups':torch.tensor(groups),'logit_scale':scale,'protocol_contributions':contributions},out/'retrieval.pt');dump(out/'retrieval_metrics.json',summary);dump(out/'result.json',{'ok':True,'policy':name,'policy_hash':policy_hash,'greedy_records':273,'sampled_records':110,'sampled_images':55,'records':383,'baseline_greedy_reused':bool(reuse),'all_raw_states_saved_and_scored_identically':True,'readout_frozen':scorer.verify_frozen(),'Agent_optimizer_updates_during_eval':0,'new_test_scores':0,'elapsed_seconds':time.monotonic()-start});print('FIXED_POLICY_EVAL_COMPLETE',name,flush=True)

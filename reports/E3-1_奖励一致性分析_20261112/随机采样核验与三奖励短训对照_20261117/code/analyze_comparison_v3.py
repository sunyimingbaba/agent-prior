"""Matched diagnostics and paired image bootstrap; supports the declared bounded gate-stop outcome."""
from pathlib import Path
import hashlib
import itertools
import numpy as np
import torch
from stage_io import read,dump
r=Path(__file__).resolve().parent.parent;p=read(r/'plan.json');assert read(r/'initial_job_result_v2.json')['status']=='complete';protocol_lock=read(r/'analysis_protocol_lock.json');assert protocol_lock['primary_quantity_generation_mode']=='greedy' and protocol_lock['primary_retrieval_guard'].startswith('repository_duplicate_token_positive');assert read(r/'validation_identity_guard.json')['ok'];pilot=read(r/'pilot/records.json');lock=read(r/'prompt_selection_lock.json');case={c['image']:c for c in read(r/'inputs/quantity_cases.json')};rngseed=p['analysis']['seed'];B=p['analysis']['bootstrap']
def rng(label):return np.random.default_rng((rngseed+int(hashlib.sha256(label.encode()).hexdigest()[:8],16))%2**32)
def rank(x):
 x=np.asarray(x);return np.array([((x<v).sum()+(x<=v).sum()-1)/2 for v in x],float)
def rho(x,y):
 x=rank(x);y=rank(y)
 return float(np.corrcoef(x,y)[0,1]) if np.ptp(x)>0 and np.ptp(y)>0 else None
def signal(xs,label,training=False):
 result={};groupkey=lambda x:(str(x['attempt'])+'/'+x['image']) if training else x['image'];names=list(dict.fromkeys(groupkey(x) for x in xs))
 for arm in p['reward_arms']:
  values=[]
  for name in names:
   group=[x for x in xs if groupkey(x)==name];v=np.array([x['r_'+arm] for x in group]);util=[x['B_hard_margin'] for x in group];fmtvalid=[x for x in group if x['strict_completed_consumed_tool']];cor=rho(v,util)
   values.append({'image':name,'candidates':len(group),'strict_completed':len(fmtvalid),'unique_rewards':len(set(v.tolist())),'reward_std':float(v.std(ddof=1)) if len(v)>1 else 0.,'reward_range':float(np.ptp(v)),'spearman_vs_B_hard_margin':cor,'spearman_strict_only':rho([x['r_'+arm] for x in fmtvalid],[x['B_hard_margin'] for x in fmtvalid]) if len(fmtvalid)>1 else None,'B_hard_margin_range':float(np.ptp(util))})
  rr=np.array([x['spearman_vs_B_hard_margin'] for x in values if x['spearman_vs_B_hard_margin'] is not None]);ci=np.percentile(rr[rng(label+arm).integers(len(rr),size=(B,len(rr)))].mean(1),[2.5,97.5]).tolist() if len(rr) and not training else None
  result[arm]={'image_groups':len(values),'informative_groups':sum(x['reward_std']>1e-8 for x in values),'mean_within_image_spearman':float(rr.mean()) if len(rr) else None,'informative_correlation_groups':len(rr),'spearman_image_bootstrap95':ci,'interval_scope':'descriptive only for sequential training attempts; no independence claimed across attempts' if training else 'paired image bootstrap within fixed policy/prompt','groups':values}
 return result
signals={q:signal([x for x in pilot if x['prompt']==q],'pilot/'+q) for q in dict.fromkeys(x['prompt'] for x in pilot)}
analysis={'ok':True,'scope':p['analysis']['scope'],'pilot_records':len(pilot),'pilot_prompt_outcomes':lock['outcomes'],'selected_prompt':lock['selected_prompt'],'pilot_signals':signals,'primary_holdout_strata':{str(n):sum(c['split']=='train' and c['image'] not in p['training_images'] and c['actual_count']==n for c in case.values()) for n in range(1,5)},'new_test_scores':0,'readout_optimizer_updates':0,'training_completed':False}
job=read(r/'continuation_job_result_v2.json') if (r/'continuation_job_result_v2.json').exists() else {};analysis['job_status']=job.get('status','pending')
if (r/'probability_gate_v2/result.json').exists():analysis['probability_gate']=read(r/'probability_gate_v2/result.json')
if job.get('status')=='generation_training_evaluation_complete':
 policies=p['evaluation']['policies'];training={a:read(r/'training_v2'/a/'result.json') for a in p['reward_arms']};summary={};qval={};retrieval={};retrieval_vals={};raw={};training_signals={}
 for a in training:
  xs=[__import__('json').loads(v) for v in (r/'training_v2'/a/'samples.jsonl').read_text().splitlines()];training_signals[a]=signal(xs,'training/'+a,training=True)
 for policy in policies:
  xs=read(r/'evaluation'/policy/'records.json');raw[policy]=xs;summary[policy]={};retrieval[policy]=read(r/'evaluation'/policy/'retrieval_metrics.json');rt=torch.load(r/'evaluation'/policy/'retrieval.pt',map_location='cpu',weights_only=True);retrieval_vals[policy]={k:v['image_cluster_mR'].numpy() for k,v in rt['protocol_contributions'].items()}
  for mode in ['greedy','sampled']:
   rows=[x for x in xs if x['mode']==mode];summary[policy][mode]={'records':len(rows),'strict_completed':sum(x['strict_completed_consumed_tool'] for x in rows),'strict_rate':float(np.mean([x['strict_completed_consumed_tool'] for x in rows])),'generated_results':sum(x['generated_result_tag'] for x in rows),'generated_result_rate':float(np.mean([x['generated_result_tag'] for x in rows])),'consumed_feedback':sum(x['feedback_consumed'] for x in rows)}
   labelled=[x for x in rows if x['image'] in case]
   for cohort in (['training8','heldout_train55','val22','all85'] if mode=='greedy' else ['heldout_train55','all_sampled55']):
    selected=[x for x in labelled if (cohort=='training8' and x['image'] in p['training_images']) or (cohort=='heldout_train55' and x['split']=='train' and x['image'] not in p['training_images']) or (cohort=='val22' and x['split']=='val') or cohort in ['all85','all_sampled55']];entry={'records':len(selected),'images':len(set(x['image'] for x in selected))}
    if not selected:continue
    for word in ['A','B']:
     for metric in ['pair_correct','hard_correct','pair_margin','hard_margin']:
      key=word+'_'+metric;entry[key+'_mean']=float(np.mean([x[key] for x in selected]));strata={n:np.array([np.mean([float(x[key]) for x in selected if x['image']==name]) for name in dict.fromkeys(x['image'] for x in selected) if case[name]['actual_count']==n]) for n in range(1,5)}
      if all(len(v) for v in strata.values()):entry[key+'_macro1to4']=float(np.mean([v.mean() for v in strata.values()]))
      if cohort=='heldout_train55':qval[policy,mode,key]=strata
    summary[policy][mode][cohort]=entry
 sampled_rank_patterns={}
 for policy in policies:
  sampled=[x for x in raw[policy] if x['mode']=='sampled'];entry={}
  for arm in p['reward_arms']:
   counts={'images':55,'comparable':0,'agree':0,'reverse':0,'reward_tie_only':0,'utility_tie_only':0,'both_tie':0};perimage=[]
   for image in dict.fromkeys(x['image'] for x in sampled):
    pair=sorted([x for x in sampled if x['image']==image],key=lambda x:x['sample']);assert len(pair)==2;dr=pair[1]['r_'+arm]-pair[0]['r_'+arm];du=pair[1]['B_hard_margin']-pair[0]['B_hard_margin'];kind=('both_tie' if dr==0 and du==0 else 'reward_tie_only' if dr==0 else 'utility_tie_only' if du==0 else 'agree' if dr*du>0 else 'reverse');counts[kind]+=1
    if dr!=0 and du!=0:counts['comparable']+=1
    perimage.append({'image':image,'reward_delta':dr,'B_hard_margin_delta':du,'classification':kind,'both_strict':all(x['strict_completed_consumed_tool'] for x in pair)})
   counts['agree_fraction_comparable']=counts['agree']/counts['comparable'] if counts['comparable'] else None;counts['per_image']=perimage;entry[arm]=counts
  sampled_rank_patterns[policy]=entry
 contrasts=[];quantity_ci=[];retrieval_ci=[]
 for i,a in enumerate(policies):
  for b in policies[:i]:contrasts.append((a,b))
 for a,b in contrasts:
  for mode in ['greedy','sampled']:
   for word in ['A','B']:
    for metric in ['pair_correct','hard_correct','hard_margin']:
     key=word+'_'+metric;ds={n:qval[a,mode,key][n]-qval[b,mode,key][n] for n in range(1,5)};g=rng('quantity/'+a+'/'+b+'/'+mode+'/'+key);boot=sum(v[g.integers(len(v),size=(B,len(v)))].mean(1) for v in ds.values())/4;quantity_ci.append({'policy':a,'reference':b,'mode':mode,'metric':key,'unit':'paired heldout training image, true-count stratified','delta_macro1to4':float(np.mean([v.mean() for v in ds.values()])),'interval95':np.percentile(boot,[2.5,97.5]).tolist()})
  for protocol in retrieval[a]:
   ds=retrieval_vals[a][protocol]-retrieval_vals[b][protocol];boot=ds[rng('retrieval/'+a+'/'+b+'/'+protocol).integers(210,size=(B,210))].mean(1);retrieval_ci.append({'policy':a,'reference':b,'protocol':protocol,'delta_mR':float(ds.mean()),'interval95':np.percentile(boot,[2.5,97.5]).tolist(),'unit':'pairedvalimage cluster5captions with fixed gallery'})
 primary=next(x for x in quantity_ci if x['policy']=='quantity_hard' and x['reference']=='proxy_rank' and x['mode']=='greedy' and x['metric']=='B_hard_correct');guard=next(x for x in retrieval_ci if x['policy']=='quantity_hard' and x['reference']=='baseline_continuation0' and x['protocol']=='repository_duplicate_token_positive');checks={'primary_lower_gt0':primary['interval95'][0]>0,'retrieval_lower_gt_minus0.02':guard['interval95'][0]>-.02,'sampled_protocol_all_policies':all(summary[a]['sampled']['strict_rate']>=.90 and summary[a]['sampled']['generated_result_rate']<=.02 for a in policies),'same32actual_updates':all(x['optimizer_updates']==32 for x in training.values())};analysis.update(training_completed=True,training=training,training_signals=training_signals,evaluation=summary,sampled_rank_patterns=sampled_rank_patterns,retrieval=retrieval,quantity_contrasts=quantity_ci,retrieval_contrasts=retrieval_ci,adoption={'checks':checks,'passed':all(checks.values()),'primary':primary,'retrieval_guard':guard},Agent_optimizer_updates=sum(x['optimizer_updates'] for x in training.values()))
else:analysis['Agent_optimizer_updates']=0 if job.get('status')=='pilot_protocol_gate_failed' else '[see per-arm raw progress]'
dump(r/'analysis.json',analysis);print('ANALYSIS_COMPLETE',analysis['job_status'],analysis['training_completed'],flush=True)

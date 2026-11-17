"""Independently derive three retrieval protocols, quantity primary, all paired intervals and adoption."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
r=Path(sys.argv[1]).resolve();torch.set_num_threads(4)
def read(p):return json.loads(Path(p).read_text())
def near(a,b,tol=1e-12):assert abs(float(a)-float(b))<=tol,(a,b,tol)
def expected(s,pos):
 out=np.zeros((len(s),3),float)
 for k0,k in enumerate([1,5,10]):
  for i,row in enumerate(s):
   cut=np.sort(row)[-k];greater=row>cut;equal=row==cut
   if (pos[i]&greater).any():out[i,k0]=1.;continue
   population=int(equal.sum());good=int((equal&pos[i]).sum());take=k-int(greater.sum());miss=1.
   for draw in range(take):miss*=max(population-good-draw,0)/(population-draw)
   out[i,k0]=1-miss
 return out
p=read(r/'plan.json');a=read(r/'analysis.json');assert a['training_completed'];g=read(r/'evaluation/gallery.json');owners=np.array(g['owners']);groups=np.array(g['groups']);assert len(owners)==1050 and all((owners==i).sum()==5 for i in range(210));cases={x['image']:x for x in read(r/'inputs/quantity_cases.json')};qgallery=read(r/'inputs/quantity_gallery.json');words={w:i+1 for i,w in enumerate('one two three four five six seven'.split())};retrieval={};qdata={};checks=0;quant=[]
for policy in p['evaluation']['policies']:
 root=r/'evaluation'/policy;t=torch.load(root/'retrieval.pt',weights_only=True,map_location='cpu');score=t['scores'].numpy();assert score.shape==(210,1050);assert torch.max(torch.abs(t['logit_scale']*t['image_features']@t['text_features'].T-t['scores']))<1e-4;ti=t['top_i'].numpy();tt=t['top_t'].numpy()
 for i in range(210):assert np.all(score[i,ti[i,:-1]]>=score[i,ti[i,1:]]) and np.all(score[i,ti[i]]>=np.sort(score[i])[-10])
 for j in range(1050):assert np.all(score[tt[:-1,j],j]>=score[tt[1:,j],j]) and np.all(score[tt[:,j],j]>=np.sort(score[:,j])[-10])
 for protocol in a['retrieval'][policy]:
  pos=owners[None,:]==np.arange(210)[:,None]
  if protocol=='repository_duplicate_token_positive':pos=np.stack([np.isin(groups,groups[owners==i]) for i in range(210)])
  if protocol=='strict_tie_averaged':ih=expected(score,pos);th=expected(score.T,pos.T)
  else:
   ih=np.array([[any(pos[i,j] for j in ti[i,:k]) for k in [1,5,10]] for i in range(210)],float);th=np.array([[any(pos[i,j] for i in tt[:k,j]) for k in [1,5,10]] for j in range(1050)],float)
  vals=np.r_[ih.mean(0),th.mean(0)];m={k:float(v) for k,v in zip(['I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10','mR'],np.r_[vals,vals.mean()])}
  for k,v in m.items():near(v,a['retrieval'][policy][protocol][k]);checks+=1
  contribution=(ih.mean(1)+th.mean(1).reshape(210,5).mean(1))/2;near(contribution.mean(),m['mR']);retrieval[policy,protocol]=contribution
 rows=read(root/'records.json');assert len(rows)==383
 for mode in ['greedy','sampled']:
  xs=[x for x in rows if x['mode']==mode];sm=a['evaluation'][policy][mode];assert len(xs)==sm['records'];near(np.mean([x['strict_completed_consumed_tool'] for x in xs]),sm['strict_rate']);near(np.mean([x['generated_result_tag'] for x in xs]),sm['generated_result_rate'])
  held=[x for x in xs if x['split']=='train' and x['image'] not in p['training_images']];names=list(dict.fromkeys(x['image'] for x in held));assert len(names)==55
  for word in ['A','B']:
   for metric in ['pair_correct','hard_correct','hard_margin']:
    values={}
    for name in names:
     pos,neg=qgallery['quantity_columns'][name][word];prefix=qgallery['quantity_text_order'][pos].rsplit(' ',1)[0];wrong=[i for i,s in enumerate(qgallery['quantity_text_order']) if s.rsplit(' ',1)[0]==prefix and words[s.rsplit(' ',1)[1][:-1]]!=cases[name]['actual_count']];vv=[]
     for row in held:
      if row['image']!=name:continue
      q=torch.load(r/row['tensor_file'],weights_only=True,map_location='cpu')['quantity_scores'];pair=float(q[pos])-float(q[neg]);hard=float(q[pos])-max(float(q[i]) for i in wrong);vv.append(float(pair>0) if metric=='pair_correct' else float(hard>0) if metric=='hard_correct' else hard)
     values[name]=float(np.mean(vv))
    strata={n:np.array([values[name] for name in names if cases[name]['actual_count']==n]) for n in range(1,5)};qdata[policy,mode,word+'_'+metric]=strata;near(np.mean([v.mean() for v in strata.values()]),sm['heldout_train55'][word+'_'+metric+'_macro1to4']);checks+=1
 for arm,x in a['sampled_rank_patterns'][policy].items():
  assert x['images']==55 and x['comparable']==x['agree']+x['reverse'] and sum(x[k] for k in ['agree','reverse','reward_tie_only','utility_tie_only','both_tie'])==55
  if x['comparable']:near(x['agree_fraction_comparable'],x['agree']/x['comparable'])
  for z in x['per_image']:
   xs=sorted([v for v in rows if v['mode']=='sampled' and v['image']==z['image']],key=lambda x:x['sample']);dr=xs[1]['r_'+arm]-xs[0]['r_'+arm];du=xs[1]['B_hard_margin']-xs[0]['B_hard_margin'];near(z['reward_delta'],dr);near(z['B_hard_margin_delta'],du)
B=p['analysis']['bootstrap'];seed=p['analysis']['seed']
def rng(label):return np.random.default_rng((seed+int(hashlib.sha256(label.encode()).hexdigest()[:8],16))%2**32)
for x in a['quantity_contrasts']:
 aa,bb,mode,key=x['policy'],x['reference'],x['mode'],x['metric'];ds={n:qdata[aa,mode,key][n]-qdata[bb,mode,key][n] for n in range(1,5)};gen=rng('quantity/'+aa+'/'+bb+'/'+mode+'/'+key);boot=np.zeros(B)
 for n,v in ds.items():boot+=v[gen.integers(len(v),size=(B,len(v)))].mean(1)/4
 near(x['delta_macro1to4'],np.mean([v.mean() for v in ds.values()]));assert np.max(np.abs(np.percentile(boot,[2.5,97.5])-x['interval95']))<1e-12
for x in a['retrieval_contrasts']:
 aa,bb,proto=x['policy'],x['reference'],x['protocol'];ds=retrieval[aa,proto]-retrieval[bb,proto];boot=ds[rng('retrieval/'+aa+'/'+bb+'/'+proto).integers(210,size=(B,210))].mean(1);near(x['delta_mR'],ds.mean());assert np.max(np.abs(np.percentile(boot,[2.5,97.5])-x['interval95']))<1e-12
primary=next(x for x in a['quantity_contrasts'] if (x['policy'],x['reference'],x['mode'],x['metric'])==('quantity_hard','proxy_rank','greedy','B_hard_correct'));guard=next(x for x in a['retrieval_contrasts'] if (x['policy'],x['reference'],x['protocol'])==('quantity_hard','baseline_continuation0','repository_duplicate_token_positive'));flags={'primary_lower_gt0':primary['interval95'][0]>0,'retrieval_lower_gt_minus0.02':guard['interval95'][0]>-.02,'sampled_protocol_all_policies':all(a['evaluation'][x]['sampled']['strict_rate']>=.9 and a['evaluation'][x]['sampled']['generated_result_rate']<=.02 for x in p['evaluation']['policies']),'same32actual_updates':all(a['training'][x]['optimizer_updates']==32 for x in p['reward_arms'])};assert flags==a['adoption']['checks'] and all(flags.values())==a['adoption']['passed'];out={'ok':True,'raw_metric_values_rebuilt':checks,'retrieval_protocol_conditions':12,'quantity_intervals_rebuilt':len(a['quantity_contrasts']),'retrieval_intervals_rebuilt':len(a['retrieval_contrasts']),'all_adoption_gates_rebuilt':flags,'strict_tie_calculation':'independent per-row finite-population probability','topk_validation':'raw stored score ordering and threshold, preserves recorded source version exact-tie choice'};path=r/'independent_metric_verification.json';assert not path.exists();path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))

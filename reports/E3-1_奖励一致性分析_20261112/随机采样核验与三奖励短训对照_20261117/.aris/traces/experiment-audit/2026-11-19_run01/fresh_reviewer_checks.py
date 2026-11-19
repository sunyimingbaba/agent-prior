"""Read-only fresh reviewer deterministic checks. Writes only this trace directory."""
import json, hashlib, csv, re, math, io
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import pyarrow as pa
from PIL import Image
from safetensors.torch import load_file
torch.set_num_threads(4)
R=Path("/Users/botlv/CUP-code/reports/E3-1_奖励一致性分析_20261112/随机采样核验与三奖励短训对照_20261117/统一评估_20261119/原始运行记录")
O=Path(__file__).parent
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def near(a,b,tol=1e-12):assert abs(float(a)-float(b))<tol,(a,b)
P=read(R/'plan.json');A=read(R/'analysis.json');C={c['image']:c for c in read(R/'inputs/quantity_cases.json')};H=read(R/'inputs/cohort.json');G=read(R/'inputs/quantity_gallery.json');V=read(R/'evaluation/gallery.json')
F={'source_root':str(R),'review_independence':'same-family','acceptance_status':'provisional','local_torch':torch.__version__,'local_numpy':np.__version__}
# Bind human label ledger.
ledger=Path('/Users/botlv/CUP-code/reports/E1_UCM试验_20261110/第九步_去重与补充/100组工作清单.csv')
L={z['图像编号']:z for z in csv.DictReader(ledger.open(encoding='utf-8-sig'))}
for n,c in C.items():assert int(L[n]['用户确认数量'])==c['actual_count'] and L[n]['目标']==c['original_object']
F['human_label_ledger']={'path':str(ledger),'sha256':sha(ledger),'matched':len(C)}
# Independently load original Arrow data, compare names, captions, rows, image RGB.
D=Path('/Users/botlv/cup-deploy-assets/agse-stage-a/data/ucm')
sources=read(R/'inputs/dataset_sources_reference.json');tables={}
for sp in ('train','val'):
 path=D/f'ucm_caption_karpathy_label_{sp}.arrow'
 assert sha(path)==sources[sp]['arrow_sha256'],str(path)
 tables[sp]=pa.ipc.open_file(str(path)).read_all()
for c in H:
 tab=tables[c['split']];assert tab['path'][c['row']].as_py()==c['image']
 im=Image.open(io.BytesIO(tab['image'][c['row']].as_py())).convert('RGB')
 assert hashlib.sha256(im.tobytes()).hexdigest()==c['rgb_sha256']
 if c['image'] in C:assert C[c['image']]['image_rgb_sha256']==c['rgb_sha256']
assert V['names']==tables['val']['path'].to_pylist()
assert V['captions']==sum(tables['val']['caption'].to_pylist(),[])
assert V['owners']==np.repeat(np.arange(210),5).tolist()
assert len(H)==273 and len(C)==85
F['dataset_identity']={'cohort_RGBs_verified':273,'quantity_labels':85,'gallery_images':210,'gallery_captions':1050,'counts':dict(Counter(str(c['actual_count']) for c in C.values()))}
# All final raw paths and hashes, trace classification, quantity scores; exact identity pairing.
Q={};T={};E={};total=0;trace_count=0;qcount=0;assets=torch.load(R/'assets/scoring_assets.pt',map_location='cpu',weights_only=True)
words={w:i+1 for i,w in enumerate('one two three four five six seven'.split())}
ordered=[]
for policy in P['evaluation']['policies']:
 rows=read(R/'evaluation'/policy/'records.json')
 identities=[(r['image'],r['mode'],r['sample'],r['row'],r['seed']) for r in rows]
 if ordered:assert identities==ordered
 else:ordered=identities
 assert len(set(identities))==383
 E[policy]={};Q[policy]={}
 for row in rows:
  total+=1
  for key in ['tensor','trace']:assert sha(R/row[key+'_file'])==row[key+'_sha256']
  tr=read(R/row['trace_file']);zs=tr['rounds'];ans=zs[-1]['generated_text'];m=re.fullmatch(r'<answer>(.*?)</answer>',ans.strip(),re.S)
  result=any('<result>' in z['generated_text'] or '</result>' in z['generated_text'] for z in zs)
  strict=bool(tr['stop_reason']=='answer' and not result and m and m.group(1).strip() and ans.count('<answer>')==ans.count('</answer>')==1 and '<tool_call>' not in ans and '</tool_call>' not in ans and all(not z['token_limit'] and 'addCriterion' not in z['generated_text'] for z in zs) and all(re.fullmatch(r'<tool_call>\s*scene_classify\s*</tool_call>',z['generated_text'].strip()) for z in zs[:-1]))
  consumed=any(e['after_round']<len(zs) for e in tr['tool_events'])
  assert row['strict_completed_consumed_tool']==bool(strict and consumed and tr['tools_used']>0)
  for e in tr['tool_events']:assert e['source_image']==row['image']
  trace_count+=1
  if row['image'] not in C:continue
  tensor=torch.load(R/row['tensor_file'],map_location='cpu',weights_only=True)
  assert torch.equal(tensor['prior'],tensor['step_states'][-1][-1])
  score=tensor['quantity_scores'].numpy();assert np.max(abs((tensor['image_feature']@assets['quantity_text_features'].T).numpy().ravel()-score))<1e-6
  for word in ['A','B']:
   pos,neg=G['quantity_columns'][row['image']][word];prefix=G['quantity_text_order'][pos].rsplit(' ',1)[0]
   assert words[G['quantity_text_order'][pos].rsplit(' ',1)[1][:-1]]==C[row['image']]['actual_count']
   alternatives=[i for i,s in enumerate(G['quantity_text_order']) if s.rsplit(' ',1)[0]==prefix and i!=pos]
   pm=float(score[pos])-float(score[neg]);hm=float(score[pos])-max(float(score[i]) for i in alternatives)
   for metric,val in [('pair_correct',pm>0),('hard_correct',hm>0),('pair_margin',pm),('hard_margin',hm)]:
    near(row[word+'_'+metric],val)
   if row['split']=='train' and row['image'] not in P['training_images']:
    for metric,val in [('pair_correct',pm>0),('hard_correct',hm>0),('hard_margin',hm)]:
     Q[policy].setdefault((row['mode'],word+'_'+metric,row['image']),[]).append(float(val))
  qcount+=1
 for mode in ('greedy','sampled'):
  z=[x for x in rows if x['mode']==mode];E[policy][mode]={'n':len(z),'strict':sum(x['strict_completed_consumed_tool'] for x in z),'generated_results':sum(x['generated_result_tag'] for x in z)}
 # Three retrieval protocols independently from score matrix and frozen top-k tie choice.
 t=torch.load(R/'evaluation'/policy/'retrieval.pt',map_location='cpu',weights_only=True)
 score=t['scores'].numpy();own=np.array(V['owners']);grp=np.array(V['groups']);ti=t['top_i'].numpy();tt=t['top_t'].numpy()
 assert np.max(abs((t['logit_scale']*t['image_features']@t['text_features'].T).numpy()-score))<1e-4
 assert np.array_equal(t['owners'].numpy(),own) and np.array_equal(t['groups'].numpy(),grp)
 toks=t['tokens'].tolist(); assert all((toks[i]==toks[j])==(grp[i]==grp[j]) for i in range(1050) for j in range(i))
 # Exact stored top-k may vary across torch versions only within ties.
 for scores,indices in [(score,ti),(score.T,tt.T)]:
  assert indices.shape==(len(scores),10)
  vals=np.take_along_axis(scores,indices,axis=1)
  assert np.all(vals[:,:-1]>=vals[:,1:]) and np.all(vals[:,-1]>=np.sort(scores,axis=1)[:,-10])
 for proto in A['retrieval'][policy]:
  positive=own[None,:]==np.arange(210)[:,None]
  if proto=='repository_duplicate_token_positive':positive=np.stack([np.isin(grp,grp[own==i]) for i in range(210)])
  def hits(scores,pos,idx):
   if proto!='strict_tie_averaged':return np.array([[np.any(pos[i,idx[i,:k]]) for k in [1,5,10]] for i in range(len(scores))],float)
   ans=np.empty((len(scores),3))
   for i,s in enumerate(scores):
    for j,k in enumerate([1,5,10]):
     v=np.sort(s)[-k];hi=s>v;eq=s==v
     n=int(eq.sum());good=int((eq&pos[i]).sum());take=k-int(hi.sum())
     ans[i,j]=1 if np.any(hi&pos[i]) else 1-(math.comb(n-good,take)/math.comb(n,take) if n-good>=take else 0)
   return ans
  ih,th=hits(score,positive,ti),hits(score.T,positive.T,tt.T)
  metrics=np.r_[ih.mean(0),th.mean(0)];contrib=(ih.mean(1)+th.mean(1).reshape(210,5).mean(1))/2
  near(metrics.mean(),A['retrieval'][policy][proto]['mR'])
  for key,val in zip(['I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10'],metrics):near(val,A['retrieval'][policy][proto][key])
  assert read(R/'evaluation'/policy/'retrieval_metrics.json')[proto]==A['retrieval'][policy][proto]
  assert np.max(abs(contrib-t['protocol_contributions'][proto]['image_cluster_mR'].numpy()))<1e-12
  T[policy,proto]=contrib
 print('verified policy',policy,flush=True)
F['raw_final_evaluation']={'records_hashed':total,'traces_reclassified':trace_count,'quantity_tensor_records_recomputed':qcount,'paired_identity_order_exact':True,'protocol_rates':E}
# Paired bootstraps join by explicit image identity rather than relying on array order.
names=[c['image'] for c in H if c['split']=='train' and c['image'] not in P['training_images']]
strata={n:[x for x in names if C[x]['actual_count']==n] for n in range(1,5)}
F['primary_strata']={str(n):len(v) for n,v in strata.items()}
def rng(label):return np.random.default_rng((P['analysis']['seed']+int(hashlib.sha256(label.encode()).hexdigest()[:8],16))%2**32)
B=P['analysis']['bootstrap']
for c in A['quantity_contrasts']:
 a,b,mode,key=c['policy'],c['reference'],c['mode'],c['metric']
 ds=[np.array([np.mean(Q[a][mode,key,name])-np.mean(Q[b][mode,key,name]) for name in v]) for v in strata.values()]
 g=rng('quantity/'+a+'/'+b+'/'+mode+'/'+key)
 boot=sum(d[g.integers(len(d),size=(B,len(d)))].mean(1)/4 for d in ds)
 near(c['delta_macro1to4'],np.mean([d.mean() for d in ds]));assert np.max(abs(np.percentile(boot,[2.5,97.5])-c['interval95']))<1e-12
for c in A['retrieval_contrasts']:
 a,b,proto=c['policy'],c['reference'],c['protocol'];d=T[a,proto]-T[b,proto];g=rng('retrieval/'+a+'/'+b+'/'+proto)
 boot=d[g.integers(len(d),size=(B,len(d)))].mean(1)
 near(c['delta_mR'],d.mean());assert np.max(abs(np.percentile(boot,[2.5,97.5])-c['interval95']))<1e-12
F['intervals_verified']={'quantity':len(A['quantity_contrasts']),'retrieval':len(A['retrieval_contrasts']),'resamples':B}
# Training ledger/hash chain/optimizer saved steps, plus paused-prefix binding.
F['training']={}
for arm in P['reward_arms']:
 folder=R/'training_v2'/arm;samples=[json.loads(x) for x in (folder/'samples.jsonl').read_text().splitlines()];updates=[json.loads(x) for x in (folder/'updates.jsonl').read_text().splitlines()]
 assert len(samples)==256 and len(updates)==32
 previous=P['policy']['lora_hash'];actual=0
 for j,u in enumerate(updates):
  assert u['attempt']==j+1 and u['policy_hash_before']==previous and u['image']==P['training']['image_order'][j]
  group=samples[j*8:j*8+8]
  for i,s in enumerate(group):
   assert (s['attempt'],s['sample'],s['image'],s['seed'])==(j+1,i,u['image'],2026111800+100*j+i)
   assert s['policy_hash_before']==previous
   for key in ['tensor','trace']:assert sha(R/s[key+'_file'])==s[key+'_sha256']
  reward=np.array([s['reward'] for s in group],dtype=np.float32);sd=reward.astype(float).std(ddof=1)
  assert bool(sd>1e-8)==u['optimizer_update'];actual+=u['optimizer_update']
  assert (u['policy_hash_after']!=previous)==u['optimizer_update'];assert u['optimizer_updates_so_far']==actual
  previous=u['policy_hash_after']
 ck=folder/'final/checkpoint';opt=torch.load(ck/'optimizer.pt',map_location='cpu',weights_only=True)
 assert len(opt['state'])==584 and {int(x['step']) for x in opt['state'].values()}=={actual};del opt
 tensors=load_file(str(ck/'adapter_model.safetensors'));h=hashlib.sha256()
 for n,t in sorted((k.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight'),v) for k,v in tensors.items()):h.update(n.encode());h.update(t.contiguous().view(torch.uint8).numpy().tobytes())
 assert h.hexdigest()==previous==read(folder/'result.json')['final_policy_hash'];del tensors
 F['training'][arm]={'attempts':32,'updates':actual,'skip_attempts':[x['attempt'] for x in updates if not x['optimizer_update']],'hash_verified':previous}
 print('verified training',arm,flush=True)
restore=read(R/'resume_20261119/restore.json');pair=R/'training_v2/quantity_pair'
for fn,n,key in [('samples.jsonl',96,'samples_prefix_sha256'),('updates.jsonl',12,'updates_prefix_sha256')]:
 prefix=b''.join((pair/fn).read_bytes().splitlines(keepends=True)[:n]);assert hashlib.sha256(prefix).hexdigest()==restore[key]
F['resume_prefix_exact']=True
flags={'primary_lower_gt0':A['adoption']['primary']['interval95'][0]>0,'retrieval_lower_gt_minus0.02':A['adoption']['retrieval_guard']['interval95'][0]>-.02,'sampled_protocol_all_policies':all(v['sampled']['strict']/v['sampled']['n']>=.9 and v['sampled']['generated_results']/v['sampled']['n']<=.02 for v in E.values()),'same32actual_updates':all(v['updates']==32 for v in F['training'].values())}
assert flags==A['adoption']['checks'] and all(flags.values())==A['adoption']['passed']
F['primary']=A['adoption']['primary'];F['retrieval_guard']=A['adoption']['retrieval_guard'];F['adoption_checks']=A['adoption']['checks'];F['adoption_passed']=A['adoption']['passed']
F['single_changed_primary_image']=[{'image':n,'actual_count':C[n]['actual_count'],'hard':Q['quantity_hard']['greedy','B_hard_correct',n],'proxy':Q['proxy_rank']['greedy','B_hard_correct',n]} for n in names if C[n]['actual_count']<=4 and Q['quantity_hard']['greedy','B_hard_correct',n]!=Q['proxy_rank']['greedy','B_hard_correct',n]]
post=read(R/'post_job_result_v3.json')
for key,fn in [('analysis_sha256','analysis.json'),('independent_raw_sha256','independent_raw_verification.json'),('independent_training_sha256','independent_training_verification.json'),('independent_metrics_sha256','independent_metric_verification.json'),('final_asset_sha256','final_asset_verification.json')]:assert sha(R/fn)==post[key]
F['post_result_hashes_verified']=True
F['audited_input_hashes']={str(x.relative_to(R)):'sha256:'+sha(x) for x in [R/'plan.json',R/'analysis_protocol_lock.json',R/'analysis.json',R/'post_job_result_v3.json',R/'continuation_job_result_v3.json']+list((R/'code').glob('*.py'))+[R/'inputs/quantity_cases.json',R/'inputs/cohort.json',R/'inputs/quantity_gallery.json',R/'evaluation/gallery.json']}
(O/'fresh_reviewer_deterministic_checks.json').write_text(json.dumps(F,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in F.items() if k not in ['audited_input_hashes']},ensure_ascii=False,indent=2))


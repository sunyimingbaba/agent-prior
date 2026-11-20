"""Independent audit recomputation; writes only this trace directory."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,re,csv,sys
import torch,numpy as np
from scipy.stats import rankdata
P=Path(__file__).resolve().parents[4]
O=Path(__file__).resolve().parent
load=lambda n:json.loads((P/n).read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cases=load('inputs/selected.json'); cm={r['image']:r for r in cases}
rows=load('scoring/rows.json'); binds=load('scoring/binding.json')
f=torch.load(P/'scoring/vectors.pt',map_location='cpu',weights_only=True)
gal=torch.load(P/'gallery/train.pt',map_location='cpu',weights_only=True)
gidx=load('gallery/index.json')
checks={}
for name,h in load('inputs/lock.json').items():assert sha(P/name)==h,name
for key,n in [('rows','scoring/rows.json'),('bank','scoring/vectors.pt'),('lock','inputs/lock.json'),('gallery','gallery/train.pt')]:assert binds[key]==sha(P/n),key
checks['locked_files']=len(load('inputs/lock.json'))
assert gal['image_indices']==list(range(1680))
assert len(set(gidx['names']))==1680
assert gidx['arrow_sha256']==binds['assets']['/private/AGSE_E1_20261110/legacy/assets/ucm/ucm_caption_karpathy_label_train.arrow']
assert [(r['image'],r['mode'],r['k']) for r in rows]==list(f['order'])
assert all(r['split']=='train' and gidx['names'][r['row']]==r['image'] for r in cases)
excluded=load('inputs/excluded_e1.json')
e1root=P.parents[1]/'E1_UCM试验_20261110'
e1=set()
for name in ['三类细节基线_20261120/inputs/scored_images.json','UCM验证交付_20261120/inputs/cases.json']:
 e1|={r['image'] for r in json.loads((e1root/name).read_bytes())}
assert e1==set(excluded)
assert not e1&cm.keys()
assert len({c['rgb_sha256'] for c in cases})==len(cases)==32
checks.update(images=32,pairs=len({c['pair'] for c in cases}),excluded_ids=len(e1),train_gallery=1680)
ps=f['proxy'].double().numpy()@gal['image_features'].double().numpy().T
cs=f['visual'].double().numpy()@f['text'].double().numpy().T
errors=[];unique_seeds=set();groups=defaultdict(list);history=[];seq_groups=defaultdict(set)
for i,r in enumerate(rows):
 c=cm[r['image']]; k=r['k']; mode=r['mode']; tag='greedy' if k<0 else f'sample{k}'
 p=P/'samples'/mode/Path(r['image']).stem/(tag+'.pt');s=load(str(p.with_suffix('.json').relative_to(P)))
 assert sha(p)==s['sha256']
 h=torch.load(p,map_location='cpu',weights_only=True)
 assert h.shape==(3584,) and h.dtype==torch.bfloat16 and torch.isfinite(h).all()
 assert sha(p)==s['sha256'] and hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()==s['vector_sha256']==r['native_tensor']
 assert s['rgb_sha256']==c['rgb_sha256'] and s['row']==c['row']
 assert s['seed']==20261120+c['row']*40+(10 if mode=='original' else 0)+k+1
 assert s['seed']==r['seed'] and r['query']==s['query']==c['scene_query' if mode=='scene' else 'orig_query']
 assert s['seed'] not in unique_seeds;unique_seeds.add(s['seed'])
 tr='\n'.join(z['decoded'] for z in s['rounds']);assert tr==r['text'] and s['rounds'][-1]['decoded']==r['final']
 assert 1<=len(s['rounds'])<=3 and all(1<=len(z['ids'])<=64 for z in s['rounds'])
 tool_calls=sum(bool(re.search(r'<tool_call>\s*scene_classify\s*(?:</tool_call>)?',z['decoded'])) for z in s['rounds'])
 assert tool_calls==len(s['events'])==r['tools']==s['used'] and s['filtered']==0
 assert all(e['confidence']>=0 for e in s['events'])
 rank=int((ps[i]>ps[i,c['row']]).sum());assert rank==r['rank0']
 fmt=.2 if re.search(r'<answer>(.*?)</answer>',tr,re.S) else 0
 assert fmt==r['format'] and r['soft']==1/(1+rank) and r['reward']==fmt+1/(1+rank)
 for w in ['A','B']:
  pos,neg=cs[i,[f['texts'].index(t) for t in c[w]]]
  errors += [abs(pos-r[w+'_pos']),abs(neg-r[w+'_neg']),abs(pos-neg-r[w])]
 if k<0 and mode=='original':history.append(s['history']['bit_equal'])
 if k>=0:groups[(mode,r['image'])].append(r);seq_groups[(mode,r['image'])].add(tuple(tuple(z['ids']) for z in s['rounds']))
checks.update(n=576,seed_count=len(unique_seeds),rank_discrepancies=0,max_cos_error=float(max(errors)),history_reported_exact=sum(history))
assert max(errors)<1e-12
corr=lambda x,y:None if len(set(x))<2 or len(set(y))<2 else float(np.corrcoef(rankdata(x),rankdata(y))[0,1])
def cluster(values,pairs):
 names=list(dict.fromkeys(pairs)); blocks=[[v for v,p in zip(values,pairs) if p==name and v is not None] for name in names]
 ns=np.array([len(x) for x in blocks]);ss=np.array([sum(x) for x in blocks]);ids=np.random.default_rng(20261120).integers(len(names),size=(20000,len(names)))
 n=ns[ids].sum(1);means=ss[ids].sum(1)[n>0]/n[n>0]
 return {'mean':float(ss.sum()/ns.sum()),'lo':float(np.quantile(means,.025)),'hi':float(np.quantile(means,.975)),'n':int(ns.sum())}
summary={};per=[]
for (mode,n),g in groups.items():
 assert sorted(z['k'] for z in g)==list(range(8))
 a,b,r=[np.array([z[k] for z in g]) for k in ['A','B','reward']]
 d=dict(mode=mode,image=n,pair=cm[n]['pair'],random_correct=float(np.mean(b>0)),random_B=float(b.mean()),rho=corr(r,b),rho_A=corr(a,b))
 for name,x in [('proxy',r),('alt_A',a),('oracle',b)]:
  ix=x==max(x);d[name+'_correct']=float((b[ix]>0).mean());d[name+'_B']=float(b[ix].mean());d[name+'_gain_acc']=d[name+'_correct']-d['random_correct'];d[name+'_gain_B']=d[name+'_B']-d['random_B']
 per.append(d)
for mode in ['scene','original']:
 data=[d for d in per if d['mode']==mode]
 s={k:cluster([d[k] for d in data],[d['pair'] for d in data]) for k in data[0] if k not in ['mode','image','pair']}
 reported=load('选择统计.json')[mode]['caps']
 for k,v in s.items():
  for stat,num in v.items():assert abs(num-reported[k][stat])<1e-12,(mode,k,stat)
 summary[mode]=s
checks['sampling']={mode:dict(unique_state_counts=dict(Counter(len({r['native_tensor'] for r in g}) for (m,n),g in groups.items() if m==mode)),unique_trajectory_counts=dict(Counter(len(s) for (m,n),s in seq_groups.items() if m==mode))) for mode in ['scene','original']}
# Recompute full native token-group retrieval per row, including excluded first group.
ni=load('native/index.json');nf=torch.load(P/'native/features.pt',map_location='cpu',weights_only=True)
assert sha(P/'native/features.pt')==ni['sha256']
assert len(set(map(tuple,nf['tokens'].tolist())))==ni['token_groups']
ns=f['visual'].double().numpy()@nf['features'].double().numpy().T
with (P/'训练原描述逐候选.csv').open(encoding='utf-8-sig') as h: nt=list(csv.DictReader(h))
ntidx={(r['mode'],r['image'],int(r['k']),r['gallery']):r for r in nt}
ng=defaultdict(list);nerr=0
for i,r in enumerate(rows):
 m=ni['maps'][r['image']];pos=m['positive'];assert ni['columns'][cm[r['image']]['orig_query']]==m['first']
 assert pos==sorted({ni['columns'][s] for s in m['captions']}) and len(m['captions'])==5
 neg=np.array([j for j in range(ni['token_groups']) if j not in pos]);bestneg=ns[i,neg].max()
 for tag in ['all','drop_first']:
  pp=pos if tag=='all' else [j for j in pos if j!=m['first']]
  if not pp:continue
  best=ns[i,pp].max();rank=int((ns[i,neg]>best).sum());rr=1/(1+rank);margin=best-bestneg
  v=ntidx[(r['mode'],r['image'],r['k'],tag)]
  assert rank==int(v['rank0']) and rr==float(v['rr'])
  nerr=max(nerr,abs(margin-float(v['margin'])),abs(best-float(v['best_positive'])),abs(bestneg-float(v['best_negative'])))
  if r['k']>=0:ng[(r['mode'],tag,r['image'])].append(dict(**r,rr=rr,margin=float(margin)))
assert nerr<1e-12
nr=[]
for (mode,tag,n),g in ng.items():
 a,b,p,rr,ma=[np.array([z[key] for z in g]) for key in ['A','B','reward','rr','margin']]
 d=dict(mode=mode,gallery=tag,image=n,pair=cm[n]['pair'],proxy_rho=corr(p,ma),A_rho=corr(a,ma),B_rho=corr(b,ma))
 for met,v in [('rr',rr),('margin',ma)]:
  d['random_'+met]=float(v.mean())
  for key,sel in [('proxy',p),('A',a),('B',b)]:d[key+'_'+met]=float(v[sel==max(sel)].mean());d[key+'_gain_'+met]=d[key+'_'+met]-d['random_'+met]
 nr.append(d)
nsummary={}
for mode in ['scene','original']:
 nsummary[mode]={}
 for tag in ['all','drop_first']:
  g=sorted([r for r in nr if r['mode']==mode and r['gallery']==tag],key=lambda r:r['image']);reported=load('训练原描述汇总.json')[mode][tag]['stats'];results={}
  for key in reported:
   results[key]=cluster([r[key] for r in g],[r['pair'] for r in g])
   for stat,num in results[key].items():assert abs(num-reported[key][stat])<1e-10,(mode,tag,key,stat,num,reported[key][stat])
  nsummary[mode][tag]=results
checks['native']=dict(raw=ni['raw'],unique_strings=ni['unique_strings'],token_groups=ni['token_groups'],rows=len(nt),max_error=nerr)
# Replay evidence can now be shipped after base audit started.
if (P/'repeat/verified.json').exists():
 v=load('repeat/verified.json');assert len(v)==18
 for r in v:
  tag='greedy' if r['k']==-1 else f"sample{r['k']}";base=Path('samples')/r['mode']/Path(r['image']).stem/(tag+'.json');a=load(str(base));b=load('repeat/'+str(base))
  for key in ['rounds','events','seed','vector_sha256']:assert a[key]==b[key]
  assert torch.equal(torch.load(P/base.with_suffix('.pt'),weights_only=True),torch.load(P/'repeat'/base.with_suffix('.pt'),weights_only=True))
 checks['replay_verified']=len(v)
if (P/'archive_check').exists():
 count=0
 for c in cases:
  prior=P/'archive_check'/(Path(c['image']).stem+'.pt')
  now=P/'samples/original'/Path(c['image']).stem/'greedy.pt'
  rec=load(str(now.with_suffix('.json').relative_to(P)))
  assert sha(prior)==rec['history']['archive_sha']
  old=torch.load(prior,weights_only=True,map_location='cpu').reshape(3584)
  assert torch.equal(old,torch.load(now,weights_only=True,map_location='cpu'))
  count+=1
 checks['archive_independent_exact']=count
# Optional later C supplement.
heldout={}
if (P/'第三措辞汇总.json').exists():
 hi=load('heldout/index.json');hf=torch.load(P/'heldout/text.pt',map_location='cpu',weights_only=True)
 assert sha(P/'heldout/text.pt')==hi['text_sha256'] and sha(P/'review/heldout_C.tsv')==hi['labels_sha256'] and sha(P/'第三种措辞检验.md')==hi['protocol_sha256']
 sim=f['visual'].double().numpy()@hf['features'].double().numpy().T
 cg=defaultdict(list)
 for i,r in enumerate(rows):
  if r['image'] in hi['columns'] and r['k']>=0:
   pi,ni_=hi['columns'][r['image']];cg[(r['mode'],r['image'])].append(dict(**r,C=float(sim[i,pi]-sim[i,ni_])))
 cr=[]
 for (mode,n),g in cg.items():
  a,b,c,p=[np.array([r[k] for r in g]) for k in ['A','B','C','reward']]
  d=dict(mode=mode,image=n,pair=cm[n]['pair'],random_acc=float((c>0).mean()),random_margin=float(c.mean()),all3_any=int(any((a>0)&(b>0)&(c>0))),all3_fraction=float(((a>0)&(b>0)&(c>0)).mean()),mixed=int(min(c)<=0<max(c)))
  for key,sel in [('proxy',p),('A',a),('B',b),('AB',(rankdata(a)+rankdata(b))/2),('oracle',c)]:
   mask=sel==max(sel);d[key+'_acc']=float((c[mask]>0).mean());d[key+'_margin']=float(c[mask].mean());d[key+'_gain_acc']=d[key+'_acc']-d['random_acc'];d[key+'_gain_margin']=d[key+'_margin']-d['random_margin'];d[key+'_rho']=corr(sel,c);d[key+'_ties']=int(mask.sum())
  d['AB_vs_proxy_acc']=d['AB_acc']-d['proxy_acc'];cr.append(d)
 for mode in ['scene','original']:
  g=[r for r in cr if r['mode']==mode];reported=load('第三措辞汇总.json')[mode];res={}
  assert reported['all3_any']==sum(r['all3_any'] for r in g) and reported['mixed']==sum(r['mixed'] for r in g)
  for k in reported['estimates']:
   res[k]=cluster([r[k] for r in g],[r['pair'] for r in g])
   for stat,num in res[k].items():assert abs(num-reported['estimates'][k][stat])<1e-10,(mode,k,stat)
  heldout[mode]=dict(all3_any=reported['all3_any'],stats=res)
report=dict(checks=checks,main=summary,native=nsummary,heldout=heldout)
(O/'recomputed.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(checks,ensure_ascii=False,indent=2))
print('INDEPENDENT_RECOMPUTE_OK')

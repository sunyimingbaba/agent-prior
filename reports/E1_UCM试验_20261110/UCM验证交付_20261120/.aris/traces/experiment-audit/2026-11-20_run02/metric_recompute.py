"""Independent in-memory numerical audit; writes only next to this script."""
import ast,csv,hashlib,io,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
sys.dont_write_bytecode=True
TRACE=Path(__file__).resolve().parent
ROOT=TRACE.parents[3]
PREV=ROOT.parent/'三类细节基线_20261120'
sys.path.insert(0,str(ROOT/'code'))
METHODS=['CLIP','CUP_CCB','ASPF','AGSE','AGSE_archived','AGSE_scene']
def load(p):return json.loads(p.read_bytes())
def read(name,root=ROOT):
 with (root/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
issues=[]
def check(condition,description):
 if not condition:issues.append(description)
def eq(actual,expected,description):
 if actual==expected:return
 try:
  if abs(float(actual)-float(expected))<=1e-12:return
 except (ValueError,TypeError):pass
 issues.append({'where':description,'actual':actual,'expected':expected})
cases=load(ROOT/'inputs/cases.json');locked=load(ROOT/'inputs/lock.json')
for k,v in locked.items():eq(sha(ROOT/k),v,'lock:'+k)
key=lambda x:(x['cohort'],x['image'],x['template'])
check(len({key(x) for x in cases})==len(cases),'unique case key')
check(all(x['split']=='val' for x in cases),'all val')
candidates=load(ROOT/'inputs/candidates.json');by_image={x['image']:x for x in candidates}
previous={(r['model'],r['image'],r['wording']):r for r in read('pair_scores.csv',PREV) if r['split']=='val'}
previous_scored={r['image'] for r in read('pair_scores.csv',PREV)}
extension_images={r['image'] for r in cases if r['cohort']=='extension'}
check(not extension_images&previous_scored,'extension filename overlaps previous scored')
methods={};allrows=[]
for m in METHODS:
 folder=ROOT/'outputs'/m;settings=load(folder/'settings.json');raw=load(folder/'rows.json')
 eq(sha(folder/'features.pt'),settings['features_sha256'],m+':features hash')
 eq(sha(ROOT/'inputs/lock.json'),settings['lock_sha256'],m+':lock hash')
 check([{k:v for k,v in x.items() if k!='scores'} for x in raw]==cases,m+':raw rows equal frozen cases')
 b=torch.load(folder/'features.pt',map_location='cpu',weights_only=True)
 check(set(b['images'])=={x['image'] for x in cases},m+':feature image set')
 check(len(b['images'])==len(set(b['images'])),m+':unique feature images')
 check(len(b['texts'])==len(set(b['texts'])),m+':unique feature texts')
 i=b['image_features'].double().numpy();t=b['text_features'].double().numpy()
 check(np.isfinite(i).all() and np.isfinite(t).all(),m+':finite vectors')
 scores=i@t.T;errors=[];old_errors=[];flips=0;rows=[]
 for x in raw:
  s=x['scores'];ix=b['images'].index(x['image'])
  for name in ('positive','generic','negative'):errors.append(abs(float(np.dot(i[ix],t[b['texts'].index(x[name])]))-s[name]))
  p,g,w=(s[k] for k in ('positive','generic','negative'))
  r={k:v for k,v in x.items() if k!='scores'}
  r['locked_dimension']=r['dimension'];r['dimension']='relation' if r['template']=='shadow_direction' else r['dimension']
  r.update(model=m,pos=p,bland=g,neg=w,margin=p-w,correct=int(p>w),tie=int(p==w),p_above_g=int(p>g),g_above_w=int(g>w),p_top=int(p>g and p>w),g_top=int(g>p and g>w),w_top=int(w>p and w>g),strict_p_g_w=int(p>g>w),ranking_tie=int(len({p,g,w})<3))
  rows.append(r)
  if r['cohort']=='previous' and m!='AGSE_scene':
   q=previous[(m,r['image'],r['template'])];eq(q['positive'],r['positive'],m+str(key(r))+':positive');eq(q['negative'],r['negative'],m+str(key(r))+':negative')
   old_errors.append(abs(float(q['margin'])-r['margin']));flips+=int((q['correct']=='True')!=bool(r['correct']))
 allrows+=rows
 methods[m]={'n':len(rows),'images':len(b['images']),'texts':len(b['texts']),'cosines':len(errors),'max_error':max(errors),'image_max_norm_deviation':float(np.abs(np.linalg.norm(i,axis=1)-1).max()),'text_max_norm_deviation':float(np.abs(np.linalg.norm(t,axis=1)-1).max()),'previous_n':len(old_errors),'previous_max_margin_error':max(old_errors,default=None),'previous_decision_flips':flips}
 check(max(errors)<1e-12,m+':vector score reproduction');check(flips==0,m+':historical decision consistency');check(max(old_errors,default=0)<2e-6,m+':historical margin consistency')
rows_idx={(x['model'],*key(x)):x for x in allrows}
actual=read('逐条三句得分.csv')
check(len(actual)==len(allrows),'allrows length')
for a,b in zip(actual,allrows):
 for k,v in b.items():eq(a[k],str(v),'逐条三句得分.csv:'+str((a['model'],key(a),k)))

def select(r,subset):
 c,d,t=r['cohort'],r['dimension'],r['template']
 if subset in ('quantity_A','quantity_B'):return c=='previous' and d=='quantity' and t==subset[-1]
 if subset in ('balanced_A','balanced_B'):return c=='quantity_reversed' and t==subset[-1]
 if subset.startswith('locked_'):return c=='extension' and r['locked_dimension']==subset[7:]
 if subset.startswith('old_'):return c=='previous' and d==subset[4:]
 if subset.startswith('combined_'):return c in ('extension','previous') and d==subset[9:]
 if subset.startswith('source_'):return c=='extension' and d==subset[7:] and r['source'].startswith('captions:')
 if subset=='attribute_without_axis':return c=='extension' and d=='attribute' and r['subtype']!='orientation'
 if subset=='relation_named_refs':return c=='extension' and d=='relation' and t in {'water_shore','road_over','river_banks','house_surround','shrub_water','roof_field','roof_drive','shadow_tank','shadow_direction','infield_side','court_neighbors','junction_corners','plane_parallel'}
 return c=='extension' and d==subset

def boot(values,groups):
 unit={}
 for v,g in zip(values,groups):unit.setdefault(g,[]).append(v)
 unit=list(unit.values());rng=np.random.default_rng(20261120)
 draws=rng.integers(len(unit),size=(20000,len(unit)))
 sums=np.array([sum(x) for x in unit]);sizes=np.array([len(x) for x in unit])
 return np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975])
main=[]
for actual in read('分项结果.csv'):
 m=actual['model'];s=actual['subset'];rr=[r for r in allrows if r['model']==m and select(r,s)];pairs=defaultdict(list)
 for r in rr:pairs[r['template'],r['pair']].append(r)
 pair_correct=[];assignment=[]
 if s not in ('quantity_A','quantity_B'):
  for pair,pp in pairs.items():
   check(len(pp)==2,'pair length:'+str((m,s,pair)))
   check(pp[0]['positive']==pp[1]['negative'] and pp[0]['negative']==pp[1]['positive'] and pp[0]['image']!=pp[1]['image'],'crossed descriptions:'+str((m,s,pair)))
   check(pp[0]['scene']==pp[1]['scene'],'same scene:'+str((m,s,pair)))
   pair_correct.append(sum(x['correct'] for x in pp));assignment.append(int(sum(x['margin'] for x in pp)>0))
 expected={'n':len(rr),'unique_images':len({r['image'] for r in rr}),'scene_count':len({r['scene'] for r in rr}),'n_pairs':len(pair_correct),'both_correct':pair_correct.count(2),'both_wrong':pair_correct.count(0),'same_preference':pair_correct.count(1),'assignment_correct':sum(assignment),'mean_margin':np.mean([r['margin'] for r in rr]),'mean_pos_minus_generic':np.mean([r['pos']-r['bland'] for r in rr]),'mean_generic_minus_neg':np.mean([r['bland']-r['neg'] for r in rr])}
 for metric in ('correct','tie','p_above_g','g_above_w','p_top','g_top','w_top','strict_p_g_w','ranking_tie'):expected[metric]=sum(r[metric] for r in rr)
 if m=='AGSE_scene':
  for metric in ('p_above_g','g_above_w','p_top','g_top','w_top','strict_p_g_w','ranking_tie','mean_pos_minus_generic','mean_generic_minus_neg'):expected[metric]=''
 for k,v in expected.items():eq(actual[k],str(v),'分项结果.csv:'+str((s,m,k)))
 if s in ('quantity_A','quantity_B','attribute','relation'):main.append(actual)
for a in read('相对CUP配对差.csv'):
 rr=[r for r in allrows if r['model']==a['model'] and select(r,a['subset'])];diffs=[r['correct']-rows_idx['CUP_CCB',*key(r)]['correct'] for r in rr]
 pair_ci=boot(diffs,[r['pair'] for r in rr]);scene_ci=boot(diffs,[r['scene'] for r in rr])
 expected=dict(n=len(rr),win=diffs.count(1),loss=diffs.count(-1),delta=np.mean(diffs),ci_cluster_low=pair_ci[0],ci_cluster_high=pair_ci[1],ci_scene_low=scene_ci[0],ci_scene_high=scene_ci[1])
 for k,v in expected.items():eq(a[k],str(v),'相对CUP配对差.csv:'+str((a['subset'],a['model'],k)))
for a in read('反向图对.csv'):
 rr=[r for r in allrows if r['model']==a['model'] and select(r,a['subset']) and r['template']==a['template'] and r['pair']==a['pair']];margin=sum(r['margin'] for r in rr)
 expected=dict(correct=sum(r['correct'] for r in rr),assignment_correct=int(margin>0),assignment_tie=int(margin==0),assignment_margin=margin,scene=rr[0]['scene'])
 for k,v in expected.items():eq(a[k],str(v),'反向图对.csv:'+str((a['subset'],a['model'],a['pair'],k)))
# Execute delivered analysis with all write functions removed and captured in memory.
script_reproduction={}
for script,writer in [('analyze.py','write_csv'),('text_summary.py','save'),('sensitivity.py','write')]:
 p=ROOT/'code'/script;tree=ast.parse(p.read_text());tree.body=[n for n in tree.body if not (isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==writer)]
 captured={};env={'__name__':'audit_readonly','__file__':str(p),writer:lambda name,rows:captured.update({name:rows})}
 exec(compile(tree,str(p),'exec'),env)
 if script=='analyze.py':env['main']()
 for name,rows in captured.items():
  out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  recreated=list(csv.DictReader(io.StringIO(out.getvalue(),newline='')));actual=read(name)
  check(len(recreated)==len(actual),script+':'+name+':row length')
  before=len(issues)
  for n,(a,b) in enumerate(zip(actual,recreated),2):
   eq(set(a),set(b),name+':columns')
   for k,v in b.items():eq(a[k],v,name+':'+str(n)+':'+k)
  script_reproduction[name]={'rows':len(actual),'matches':len(issues)==before}
core=[r for r in cases if r['cohort']=='extension' or (r['cohort']=='previous' and r['dimension']=='quantity' and r['template']=='A')]
result={'issues':issues,'status':'PASS' if not issues else 'FAIL','methods':methods,'case_membership':{'all':len(cases),'images':len({r['image'] for r in cases}),'main':len(core),'main_images':len({r['image'] for r in core}),'raw_cohorts':{str(k):v for k,v in Counter((r['cohort'],r['dimension']) for r in cases).items()},'extension_images':len(extension_images),'extension_prior_overlap':sorted(extension_images&previous_scored)},'main_summary':main,'script_reproduction':script_reproduction,'sha256':{str(p.relative_to(ROOT)):sha(p) for p in [*(ROOT/'code').glob('*.py'),*(ROOT/'inputs').glob('*.json'),*(ROOT/'outputs').glob('*/*'),*ROOT.glob('*.csv')]}}
(TRACE/'metric_recompute.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('sha256','main_summary')},ensure_ascii=False,indent=2))

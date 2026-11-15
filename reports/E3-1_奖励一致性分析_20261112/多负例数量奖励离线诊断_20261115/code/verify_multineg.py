"""Reconstruct scores, selectors and bootstrap intervals directly from frozen inputs."""
import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import torch

r=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(p.read_text())
plan=read(r/'plan.json');a=read(r/'analysis.json');src=r/'source_data/原始运行记录';scores=read(src/'scoring/all_scores.json');cases=read(src/'inputs/evaluation_cases.json');by={x['image']:x for x in cases};g=read(src/'features/gallery.json');t=torch.load(src/'scoring/tensors.pt',map_location='cpu',weights_only=True);f=torch.load(src/'features/tensors.pt',map_location='cpu',weights_only=True);torch.set_num_threads(2)
for n,z in plan['input_files'].items():
    if n.endswith('.tar.gz'):continue
    p=r/'source_data'/n;assert p.stat().st_size==z['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==z['sha256'],n
assert hashlib.sha256((r/'plan.json').read_bytes()).hexdigest()==a['plan_sha256']=='ea88e212d1be7ee65ab336297cc2f055da3a49b03a481670b6a71e1162621e9e'
assert [x['id'] for x in scores]==t['record_ids']
q=t['quantity_similarities'].numpy();q64=np.asarray(t['image_features'].double()@f['quantity_text_features'].double().T);assert np.max(abs(q-q64))<1e-6
numbers={w:i+1 for i,w in enumerate('one two three four five six seven'.split())};texts=g['quantity_text_order'];derived={};numerical=[];coverage={}
rows=read(r/'candidate_scores.json');assert len(rows)==len(scores)==1148
for i,(s,x) in enumerate(zip(scores,rows)):
    assert s['id']==x['id'] and s['image']==x['image'];c=by[s['image']];z={}
    for w in ['A','B']:
        pos,neg=g['quantity_columns'][s['image']][w];prefix=texts[pos].rsplit(' ',1)[0];pool={numbers[text.rsplit(' ',1)[1][:-1]]:j for j,text in enumerate(texts) if text.rsplit(' ',1)[0]==prefix};assert pool[c['actual_count']]==pos
        coverage[prefix]=sorted(pool);other=[(k,col) for k,col in sorted(pool.items()) if k!=c['actual_count']];p=float(q[i,pos]);n=[float(q[i,col]) for k,col in other];diff=[p-v for v in n];hard=p-max(n);pred=sorted(pool,key=lambda k:(-float(q[i,pool[k]]),k))[0]
        expected={w+'_pair_margin':p-float(q[i,neg]),w+'_pair_correct':float(p>float(q[i,neg])),w+'_hard_margin':hard,w+'_hard_correct':float(all(v>0 for v in diff)),w+'_mean_margin':math.fsum(diff)/len(diff),w+'_mean_pair_acc':sum(v>0 for v in diff)/len(diff),w+'_argmax_count':pred,w+'_rank0':sum(v<=0 for v in diff),w+'_hard_negative_count':max(other,key=lambda kv:float(q[i,kv[1]]))[0]}
        for k,v in expected.items():assert abs(x[k]-v)<1e-12,(s['id'],k)
        assert expected[w+'_pair_margin']==s['margin_'+w] and bool(expected[w+'_pair_correct'])==s['correct_'+w]
        d64=q64[i,pos]-max(q64[i,col] for _,col in other);p64=max(sorted(pool),key=lambda k:q64[i,pool[k]])
        if (d64>0)!=bool(expected[w+'_hard_correct']) or p64!=pred:numerical.append((s['id'],w))
        z.update(expected)
    for k,v in [('quality',float(s['quality_usable'])),('strict_quality',float(s['strict_protocol_usable'])),('tool_usage',float(s['tools_used']>0))]:assert x[k]==v;z[k]=v
    for k,m in [('pair_reward','A_pair_margin'),('hard_reward','A_hard_margin'),('mean_reward','A_mean_margin')]:assert abs(x[k]-(.2*z['quality']+z[m]))<1e-12;z[k]=x[k]
    derived[s['id']]={**s,**z}
assert numerical==[(x['id'],x['wording']) for x in a['numerical_sensitivity']['multi_count_decision_changes']]
# Check all saved long-matrix rows against independent source indices.
long=list(csv.DictReader((r/'quantity_score_matrix.csv').open()));ix={x:i for i,x in enumerate(t['record_ids'])}
for x in long:
    i=ix[x['id']];col=int(x['gallery_column']);assert float(x['frozen_similarity'])==float(q[i,col]) and abs(float(x['float64_similarity'])-q64[i,col])<1e-12
per=read(r/'per_image_selections.json');assert len(per)==1190;lookup={};metrics=list(a['groups']['training8']['summary']['before']['uniform']['raw'])
for x in per:
    arm,image,selector=x['arm'],x['image'],x['selector'];xs=sorted([z for z in derived.values() if z['arm']==arm and z['image']==image and z['role']=='candidate'],key=lambda z:z['seed']);assert len(xs)==4
    good=[z for z in xs if z['quality']] or xs
    def best(ys,k):return sorted(ys,key=lambda z:(-z[k],z['seed']))[0]
    if selector=='uniform':ys=xs
    elif selector=='greedy':ys=[next(z for z in derived.values() if z['arm']==arm and z['image']==image and z['role']=='greedy')]
    elif selector=='hard_quality_first':ys=[best(good,'A_hard_margin')]
    elif selector=='B_oracle_quality_first':ys=[best(good,'B_hard_margin')]
    else:ys=[best(xs,{'pair_total':'pair_reward','hard_total':'hard_reward','mean_total':'mean_reward'}[selector])]
    assert x['selected_ids']==[z['id'] for z in ys]
    result={m:math.fsum(z[m] for z in ys)/len(ys) for m in metrics}
    for m,v in result.items():assert abs(v-x[m])<1e-12
    lookup[arm,image,selector]=result
# Independently rebuild each paired image/bootstrap statistic, including random draws.
intervals=0
base_seed=plan['bootstrap']['seed']
def check(st,values,counts,label,macro):
    global intervals
    v=np.array(values,dtype=np.float64);cs=np.array(counts);rng=np.random.default_rng((base_seed+int(hashlib.sha256(label.encode()).hexdigest()[:8],16))%2**32)
    if macro:
        chunks=[v[cs==c] for c in range(1,5)];assert all(len(c) for c in chunks);point=math.fsum(float(c.mean()) for c in chunks)/4;boot=np.zeros(10000)
        for c in chunks:boot+=c[rng.integers(len(c),size=(10000,len(c)))].mean(axis=1)/4
    else:point=float(v.mean());boot=v[rng.integers(len(v),size=(10000,len(v)))].mean(axis=1)
    assert abs(point-st['mean'])<1e-12 and np.allclose(np.percentile(boot,[2.5,97.5]),st['interval95'],rtol=0,atol=1e-12),label
    intervals+=1
contrasts={'hard_minus_pair':('hard_total','pair_total'),'hard_minus_uniform':('hard_total','uniform'),'mean_minus_pair':('mean_total','pair_total'),'quality_first_minus_hard':('hard_quality_first','hard_total'),'oracle_minus_hard':('B_oracle_quality_first','hard_total'),'pair_minus_uniform':('pair_total','uniform')}
for cohort,gr in a['groups'].items():
    ns=[x['image'] for x in per if x['arm']=='before' and x['selector']=='greedy' and x['cohort']==cohort];counts=[by[n]['actual_count'] for n in ns]
    assert len(ns)==gr['images'] and len(set(ns))==len(ns)
    for arm,sels in gr['summary'].items():
        for sel,scopes in sels.items():
            for scope,ms in scopes.items():
                for m,st in ms.items():check(st,[lookup[arm,n,sel][m] for n in ns],counts,cohort+'/'+arm+'/'+sel+'/'+scope+'/'+m,scope=='macro')
    for arm,cons in gr['contrasts'].items():
        for label,scopes in cons.items():
            sel,ref=contrasts[label]
            for scope,ms in scopes.items():
                for m,st in ms.items():check(st,[lookup[arm,n,sel][m]-lookup[arm,n,ref][m] for n in ns],counts,cohort+'/'+arm+'/'+label+'/'+scope+'/'+m,scope=='macro')
    for arm,scopes in gr['own_shared'].items():
        for scope,ms in scopes.items():
            for m,st in ms.items():
                shared={z['image']:z for z in derived.values() if z['arm']==arm+'_shared'}
                check(st,[lookup[arm,n,'greedy'][m]-shared[n][m] for n in ns],counts,cohort+'/'+arm+'/own_shared/'+scope+'/'+m,scope=='macro')
# Rebuild candidate-mechanics figures and feature sensitivity without relying on exported values.
mech=list(csv.DictReader((r/'candidate_mechanics.csv').open()));train_mech=list(csv.DictReader((r/'training_group_mechanics.csv').open()));assert len(mech)==170 and len(train_mech)==32
for z in mech+train_mech:
    xs=sorted([x for x in derived.values() if x['arm']==z['arm'] and x['image']==z['image'] and x['role']=='candidate' and (z['arm']!='training' or x['step']==int(z['step']))],key=lambda x:x['seed']);assert len(xs)==4
    assert int(z['unique_hidden_states'])==len({x['generation_state_sha256'] for x in xs})
    for w in ['A','B']:assert (z[w+'_hard_varies']=='True')==(len({x[w+'_hard_correct'] for x in xs})>1)
    feat=t['image_features'][[ix[x['id']] for x in xs]].double().numpy();dist=max(np.linalg.norm(v-u) for v,u in itertools.combinations(feat,2));assert abs(dist-float(z['max_image_feature_l2']))<1e-10
    for k in ['A_pair_margin','A_hard_margin','B_hard_margin']:assert abs(np.ptp([x[k] for x in xs])-float(z[k+'_range']))<1e-12
    for ka,kb,out in [('A_hard_margin','B_hard_margin','A_hard_B_hard_concordance'),('A_pair_margin','B_hard_margin','A_pair_B_hard_concordance')]:
        signs=[np.sign(v[ka]-u[ka])*np.sign(v[kb]-u[kb]) for v,u in itertools.combinations(xs,2) if v[ka]!=u[ka] and v[kb]!=u[kb]];expect=None if not signs else float(np.mean(signs));assert (expect is None and z[out]=='') or abs(expect-float(z[out]))<1e-12
updates={x['step']:x for x in [json.loads(s) for s in (src/'training/updates.jsonl').read_text().splitlines()]};adv=list(csv.DictReader((r/'training_reward_replay.csv').open()));assert len(adv)==384;max_pair_error=0.
for z in adv:
    x=derived[z['id']];u=updates[int(z['step'])];assert abs(float(z['reward'])-float(np.float32(x[z['reward_kind']])))<1e-12
    value=(torch.tensor(float(z['reward']),dtype=torch.float32)-np.float32(float(z['group_mean'])))/(np.float32(float(z['group_sample_std']))+1e-8);assert float(value)==float(z['advantage'])
    if z['reward_kind']=='pair_reward':max_pair_error=max(max_pair_error,abs(float(z['advantage'])-u['advantages'][int(z['sample'])]))
assert max_pair_error<1e-6
# Every gate is rebuilt from its underlying statistic.
conf=a['groups']['confirmation55'];checks={'all_objects_have_at_least_two_wrong_counts':all(len(p)>=3 for p in coverage.values()),'training_image_count_coverage_1_to_4_two_each':all(sum(by[n]['actual_count']==c for n in read(src/'plan.json')['training_images'])==2 for c in range(1,5)),
'after_confirmation_hard_vs_pair_B_macro_CI_lower_strictly_positive':conf['contrasts']['after']['hard_minus_pair']['macro']['B_hard_correct']['interval95'][0]>0,'after_confirmation_hard_vs_uniform_B_macro_CI_lower_strictly_positive':conf['contrasts']['after']['hard_minus_uniform']['macro']['B_hard_correct']['interval95'][0]>0,'after_confirmation_own_vs_shared_greedy_B_hard_macro_CI_lower_strictly_positive':conf['own_shared']['after']['macro']['B_hard_correct']['interval95'][0]>0,'after_confirmation_hard_selected_strict_quality_at_least_95pct':conf['summary']['after']['hard_total']['raw']['strict_quality']['mean']>=.95,'training8_hard_vs_pair_B_macro_change_nonnegative':a['groups']['training8']['contrasts']['after']['hard_minus_pair']['macro']['B_hard_correct']['mean']>=0}
assert checks==a['recommendation']['checks'] and all(checks.values())==a['recommendation']['ready_for_matched_short_training']
for z in read(r/'decisive_case_2035.json'):
    original=next(x for x in scores if x['id']==z['id']);p=r/'case_traces'/(z['id']+'.json')
    assert hashlib.sha256(p.read_bytes()).hexdigest()==original['trace_sha256']==z['trace_sha256'] and read(p)==z['trace']
proof={'ok':True,'frozen_input_files_hashed':9,'state_score_records_reconstructed':1148,'long_matrix_rows_reconstructed':len(long),'per_image_selector_rows_reconstructed':len(per),'bootstrap_intervals_reconstructed':intervals,'candidate_groups_reconstructed':170,'training_groups_reconstructed':32,'training_reward_advantages_checked':len(adv),'original_advantage_max_error':max_pair_error,'float64_similarity_max_error':float(np.max(abs(q-q64))),'multi_count_numerical_changes':len(numerical),'recommendation_gates_rebuilt':len(checks),'new_optimizer_updates':0,'new_generated_trajectories':0}
(r/'独立核验.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n');print(json.dumps(proof,ensure_ascii=False))

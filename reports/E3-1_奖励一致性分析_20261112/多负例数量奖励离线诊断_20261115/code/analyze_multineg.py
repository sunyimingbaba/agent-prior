import collections
import csv
import itertools
import json
import math
import re
import time
import numpy as np
import torch
from common import ROOT,read,dump,sha,interval

torch.set_num_threads(2);start=time.monotonic();plan=read(ROOT/'plan.json');src=ROOT/'source_data/原始运行记录'
for n,z in plan['input_files'].items():
    if not n.endswith('.tar.gz'):assert sha(ROOT/'source_data'/n)==z['sha256'],n
cases=read(src/'inputs/evaluation_cases.json');by={x['image']:x for x in cases};parent=read(src/'plan.json');scores=read(src/'scoring/all_scores.json');gallery=read(src/'features/gallery.json');t=torch.load(src/'scoring/tensors.pt',map_location='cpu',weights_only=True);f=torch.load(src/'features/tensors.pt',map_location='cpu',weights_only=True)
assert t['record_ids']==[x['id'] for x in scores] and len(scores)==1148
q=t['quantity_similarities'].double().numpy();qd=(t['image_features'].double()@f['quantity_text_features'].double().T).numpy();assert np.max(abs(q-qd))<1e-6
words={w:i+1 for i,w in enumerate(['one','two','three','four','five','six','seven'])};pools={};object_map={};coverage=[]
for i,s in enumerate(gallery['quantity_text_order']):
    m=re.fullmatch(r'The number of (.+) visible in the image is (\w+)\.',s);w='A'
    if m is None:m=re.fullmatch(r'The number of visible (.+) is (\w+)\.',s);w='B'
    assert m is not None,s
    obj,num=m.groups();pools.setdefault(obj,{}).setdefault(w,{})[words[num]]=i
for c in cases:
    for w in ['A','B']:
        pos,neg=gallery['quantity_columns'][c['image']][w];objs=[o for o,d in pools.items() if d[w].get(c['actual_count'])==pos];assert len(objs)==1
        obj=objs[0];assert c['pairs'][w]['positive']==gallery['quantity_text_order'][pos] and c['pairs'][w]['negative']==gallery['quantity_text_order'][neg]
        if c['image'] in object_map:assert object_map[c['image']]==obj
        object_map[c['image']]=obj
for obj,d in pools.items():
    assert set(d['A'])==set(d['B']);coverage.append({'object':obj,'counts':sorted(d['A']),'negative_counts_per_positive':len(d['A'])-1,'text_columns':d})
records=[];long=[];numeric=[];idx={x:i for i,x in enumerate(t['record_ids'])}
for i,x in enumerate(scores):
    c=by[x['image']];obj=object_map[x['image']];actual=c['actual_count'];cohort='training8' if x['image'] in parent['training_images'] else 'confirmation55' if c['split']=='train' else 'existing_val22'
    row={k:x[k] for k in ['id','arm','image','split','role','seed','quality_usable','strict_protocol_usable','tools_used','generation_state_sha256']};row.update(actual_count=actual,object=obj,cohort=cohort,sample=x.get('sample'),step=x.get('step'))
    for w in ['A','B']:
        cols=pools[obj][w];wrong=[n for n in sorted(cols) if n!=actual];pos=q[i,cols[actual]];ns=np.array([q[i,cols[n]] for n in wrong]);hard_n=wrong[int(np.argmax(ns))];margin=float(pos-max(ns));pairpos,pairneg=gallery['quantity_columns'][x['image']][w]
        pair=float(q[i,pairpos]-q[i,pairneg]);assert pair==x['margin_'+w] and (pair>0)==x['correct_'+w]
        ranking=sorted(cols,key=lambda n:(-q[i,cols[n]],n));mean=float(pos-ns.mean());correct=margin>0
        row.update({w+'_pair_margin':pair,w+'_pair_correct':float(pair>0),w+'_hard_margin':margin,w+'_hard_correct':float(correct),w+'_mean_margin':mean,w+'_mean_pair_acc':float(np.mean(pos>ns)),w+'_argmax_count':ranking[0],w+'_hard_negative_count':hard_n,w+'_rank0':int(np.sum(ns>=pos)),w+'_original_negative_count':c['pairs'][w]['wrong_count'],w+'_original_negative_is_hardest':bool(c['pairs'][w]['wrong_count']==hard_n),w+'_positive_score':float(pos),w+'_negative_spread':float(np.ptp(ns))})
        p64=qd[i,cols[actual]];n64=np.array([qd[i,cols[n]] for n in wrong]);m64=float(p64-n64.max());pred64=max(sorted(cols),key=lambda n:qd[i,cols[n]])
        if bool(m64>0)!=bool(correct) or pred64!=ranking[0]:numeric.append({'id':x['id'],'wording':w,'frozen_hard_correct':correct,'float64_hard_correct':bool(m64>0),'frozen_argmax':ranking[0],'float64_argmax':pred64,'frozen_hard_margin':margin,'float64_hard_margin':m64})
        for n,col in sorted(cols.items()):long.append({'id':x['id'],'image':x['image'],'arm':x['arm'],'wording':w,'object':obj,'actual_count':actual,'candidate_count':n,'positive':n==actual,'frozen_similarity':float(q[i,col]),'float64_similarity':float(qd[i,col]),'gallery_column':col})
    row.update(quality=float(x['quality_usable']),strict_quality=float(x['strict_protocol_usable']),tool_usage=float(x['tools_used']>0))
    row['pair_reward']=.2*row['quality']+row['A_pair_margin'];row['hard_reward']=.2*row['quality']+row['A_hard_margin'];row['mean_reward']=.2*row['quality']+row['A_mean_margin'];records.append(row)
byid={x['id']:x for x in records}
metrics=['A_pair_correct','B_pair_correct','A_hard_correct','B_hard_correct','A_hard_margin','B_hard_margin','A_mean_pair_acc','B_mean_pair_acc','quality','strict_quality','tool_usage']
selectors=plan['selectors'];per=[];mechanics=[];agreements=[]
def chosen(xs,key):return max(xs,key=lambda x:(x[key],-x['seed']))
def group_mechanics(xs,arm,image,step=None):
    aa=np.array([x['A_hard_margin'] for x in xs]);bb=np.array([x['B_hard_margin'] for x in xs]);ap=np.array([x['A_pair_margin'] for x in xs]);pairs=[]
    for i,k in itertools.combinations(range(4),2):
        sa=np.sign(aa[i]-aa[k]);sb=np.sign(bb[i]-bb[k]);sp=np.sign(ap[i]-ap[k]);pairs.append((sa,sb,sp))
    non_ties=[z for z in pairs if z[0]!=0 and z[1]!=0];before_non_ties=[z for z in pairs if z[2]!=0 and z[1]!=0]
    selected_pair=chosen(xs,'pair_reward');selected_hard=chosen(xs,'hard_reward');quality=[x for x in xs if x['quality']];qualified=quality or xs
    a=float(np.mean([z[0]*z[1] for z in non_ties])) if non_ties else None;b=float(np.mean([z[2]*z[1] for z in before_non_ties])) if before_non_ties else None
    feat=t['image_features'][[idx[x['id']] for x in xs]].double();dists=torch.cdist(feat,feat);cos=torch.nn.functional.cosine_similarity(feat[:,None,:],feat[None,:,:],dim=-1)
    result={'arm':arm,'image':image,'cohort':xs[0]['cohort'],'actual_count':xs[0]['actual_count'],'step':step,'pair_selected':selected_pair['id'],'hard_selected':selected_hard['id'],'selection_changed':selected_pair['id']!=selected_hard['id'],'B_hard_oracle_any':float(max(x['B_hard_correct'] for x in qualified)),
        'B_hard_all_candidates_correct':float(min(x['B_hard_correct'] for x in xs)),'B_hard_varies':len({x['B_hard_correct'] for x in xs})>1,'A_hard_varies':len({x['A_hard_correct'] for x in xs})>1,'unique_hidden_states':len({x['generation_state_sha256'] for x in xs}),
        'A_pair_margin_range':float(np.ptp(ap)),'A_hard_margin_range':float(np.ptp(aa)),'B_hard_margin_range':float(np.ptp(bb)),'A_hard_B_hard_non_tie_pairs':len(non_ties),'A_pair_B_hard_non_tie_pairs':len(before_non_ties),'A_hard_B_hard_concordance':a,'A_pair_B_hard_concordance':b,
        'max_image_feature_l2':float(dists.max()),'min_image_feature_cosine':float(cos.min()),'qualified_candidates':len(quality),'pair_reward_std':float(np.std([x['pair_reward'] for x in xs],ddof=1)),'hard_reward_std':float(np.std([x['hard_reward'] for x in xs],ddof=1))}
    return result
for arm in ['before','after']:
    for c in cases:
        n=c['image'];xs=sorted([x for x in records if x['arm']==arm and x['image']==n and x['role']=='candidate'],key=lambda x:x['seed']);assert len(xs)==4
        qual=[x for x in xs if x['quality']] or xs
        greedy=next(x for x in records if x['arm']==arm and x['image']==n and x['role']=='greedy')
        picks={'uniform':xs,'greedy':[greedy],'pair_total':[chosen(xs,'pair_reward')],'hard_total':[chosen(xs,'hard_reward')],'mean_total':[chosen(xs,'mean_reward')],'hard_quality_first':[chosen(qual,'A_hard_margin')],'B_oracle_quality_first':[chosen(qual,'B_hard_margin')]}
        assert set(picks)==set(selectors)
        for sel,ys in picks.items():per.append({'arm':arm,'image':n,'cohort':xs[0]['cohort'],'object':xs[0]['object'],'actual_count':c['actual_count'],'selector':sel,'selected_ids':[x['id'] for x in ys],**{m:math.fsum(x[m] for x in ys)/len(ys) for m in metrics}})
        mechanics.append(group_mechanics(xs,arm,n))
per_lookup={(x['arm'],x['image'],x['selector']):x for x in per};stats=[];groups={}
contrasts=[('hard_minus_pair','hard_total','pair_total'),('hard_minus_uniform','hard_total','uniform'),('mean_minus_pair','mean_total','pair_total'),('quality_first_minus_hard','hard_quality_first','hard_total'),('oracle_minus_hard','B_oracle_quality_first','hard_total'),('pair_minus_uniform','pair_total','uniform')]
for cohort in ['training8','confirmation55','existing_val22']:
    ns=[c['image'] for c in cases if next(x for x in records if x['image']==c['image'])['cohort']==cohort];counts=[by[n]['actual_count'] for n in ns];group={'images':len(ns),'macro_images':sum(c in [1,2,3,4] for c in counts),'summary':{},'contrasts':{},'own_shared':{},'candidate_mechanics':{}}
    for arm in ['before','after']:
        group['summary'][arm]={};group['contrasts'][arm]={}
        for sel in selectors:
            group['summary'][arm][sel]={}
            for scope in ['raw','macro']:
                group['summary'][arm][sel][scope]={m:interval([per_lookup[arm,n,sel][m] for n in ns],counts,cohort+'/'+arm+'/'+sel+'/'+scope+'/'+m,scope=='macro') for m in metrics}
        for label,sel,ref in contrasts:
            group['contrasts'][arm][label]={}
            for scope in ['raw','macro']:
                group['contrasts'][arm][label][scope]={}
                for m in ['B_hard_correct','B_hard_margin','B_pair_correct','strict_quality']:
                    st=interval([per_lookup[arm,n,sel][m]-per_lookup[arm,n,ref][m] for n in ns],counts,cohort+'/'+arm+'/'+label+'/'+scope+'/'+m,scope=='macro');group['contrasts'][arm][label][scope][m]=st;stats.append({'cohort':cohort,'arm':arm,'contrast':label,'scope':scope,'metric':m,**st})
        shared=[next(x for x in records if x['arm']==arm+'_shared' and x['image']==n) for n in ns];own=[per_lookup[arm,n,'greedy'] for n in ns]
        group['own_shared'][arm]={scope:{m:interval([a[m]-b[m] for a,b in zip(own,shared)],counts,cohort+'/'+arm+'/own_shared/'+scope+'/'+m,scope=='macro') for m in ['B_hard_correct','B_hard_margin','B_pair_correct','A_hard_correct']} for scope in ['raw','macro']}
        mm=[x for x in mechanics if x['arm']==arm and x['cohort']==cohort]
        group['candidate_mechanics'][arm]={'groups':len(mm),'selection_changed':sum(x['selection_changed'] for x in mm),'B_hard_varies':sum(x['B_hard_varies'] for x in mm),'A_hard_varies':sum(x['A_hard_varies'] for x in mm),'qualified_oracle_B_hard':float(np.mean([x['B_hard_oracle_any'] for x in mm])),'unique_states_min':min(x['unique_hidden_states'] for x in mm),'unique_states_max':max(x['unique_hidden_states'] for x in mm),'max_feature_l2':max(x['max_image_feature_l2'] for x in mm),'min_feature_cosine':min(x['min_image_feature_cosine'] for x in mm),'mean_hard_A_B_concordance':float(np.mean([x['A_hard_B_hard_concordance'] for x in mm if x['A_hard_B_hard_concordance'] is not None])),'mean_pair_A_B_concordance':float(np.mean([x['A_pair_B_hard_concordance'] for x in mm if x['A_pair_B_hard_concordance'] is not None])),'median_A_hard_range':float(np.median([x['A_hard_margin_range'] for x in mm]))}
    groups[cohort]=group
train_mech=[];adv_rows=[];updates=[json.loads(s) for s in (src/'training/updates.jsonl').read_text().splitlines()]
for u in updates:
    xs=sorted([x for x in records if x['arm']=='training' and x['step']==u['step']],key=lambda x:x['sample']);assert len(xs)==4
    tr=group_mechanics(xs,'training',u['image'],u['step']);train_mech.append(tr)
    for key in ['pair_reward','hard_reward','mean_reward']:
        reward=torch.tensor([x[key] for x in xs],dtype=torch.float32);mean=float(reward.mean());std=float(reward.std(unbiased=True))
        if key=='pair_reward':mean=u['reward_mean'];std=u['reward_std']
        adv=(reward-np.float32(mean))/(np.float32(std)+1e-8)
        if key=='pair_reward':assert np.allclose(adv.numpy(),u['advantages'],atol=1e-6,rtol=0)
        for i,x in enumerate(xs):adv_rows.append({'step':u['step'],'image':u['image'],'sample':i,'id':x['id'],'reward_kind':key,'reward':float(reward[i]),'group_mean':mean,'group_sample_std':std,'advantage':float(adv[i]),'B_hard_margin':x['B_hard_margin'],'B_hard_correct':x['B_hard_correct'],'quality':x['quality']})
# Independent image-state comparisons, retaining both continuous scores and decisions.
shared_rows=[]
for arm in ['before','after']:
    for c in cases:
        own=next(x for x in records if x['arm']==arm and x['image']==c['image'] and x['role']=='greedy');shared=next(x for x in records if x['arm']==arm+'_shared' and x['image']==c['image']);fo,fs=t['image_features'][idx[own['id']]].double(),t['image_features'][idx[shared['id']]].double()
        shared_rows.append({'arm':arm,'image':c['image'],'cohort':own['cohort'],'object':own['object'],'actual_count':c['actual_count'],'own_id':own['id'],'shared_id':shared['id'],'state_identical':own['generation_state_sha256']==shared['generation_state_sha256'],'feature_l2':float((fo-fs).norm()),'feature_cosine':float(torch.nn.functional.cosine_similarity(fo,fs,dim=0)),'B_hard_correct_own':own['B_hard_correct'],'B_hard_correct_shared':shared['B_hard_correct'],'B_hard_margin_delta':own['B_hard_margin']-shared['B_hard_margin'],'B_argmax_own':own['B_argmax_count'],'B_argmax_shared':shared['B_argmax_count']})
confusion=[]
for cohort,g in groups.items():
    for arm in ['before','after']:
        for role in ['candidate','greedy','shared']:
            xs=[x for x in records if x['cohort']==cohort and x['arm']==(arm+'_shared' if role=='shared' else arm) and x['role']==role]
            freq=collections.Counter((x['object'],x['actual_count'],x['B_argmax_count']) for x in xs)
            confusion += [{'cohort':cohort,'arm':arm,'role':role,'object':o,'actual_count':a,'predicted_count':p,'records':n} for (o,a,p),n in sorted(freq.items())]
g=groups['confirmation55'];checks={'all_objects_have_at_least_two_wrong_counts':all(x['negative_counts_per_positive']>=2 for x in coverage),'training_image_count_coverage_1_to_4_two_each':all(sum(by[n]['actual_count']==c for n in parent['training_images'])==2 for c in [1,2,3,4]),
'after_confirmation_hard_vs_pair_B_macro_CI_lower_strictly_positive':g['contrasts']['after']['hard_minus_pair']['macro']['B_hard_correct']['interval95'][0]>0,'after_confirmation_hard_vs_uniform_B_macro_CI_lower_strictly_positive':g['contrasts']['after']['hard_minus_uniform']['macro']['B_hard_correct']['interval95'][0]>0,
'after_confirmation_own_vs_shared_greedy_B_hard_macro_CI_lower_strictly_positive':g['own_shared']['after']['macro']['B_hard_correct']['interval95'][0]>0,'after_confirmation_hard_selected_strict_quality_at_least_95pct':g['summary']['after']['hard_total']['raw']['strict_quality']['mean']>=.95,'training8_hard_vs_pair_B_macro_change_nonnegative':groups['training8']['contrasts']['after']['hard_minus_pair']['macro']['B_hard_correct']['mean']>=0}
assert set(checks)==set(plan['recommendation_gate'])
result={'ok':True,'plan_sha256':sha(ROOT/'plan.json'),'source_records':len(records),'quantity_texts':56,'coverage':coverage,'candidate_groups':len(mechanics),'training_groups':32,'groups':groups,'training_mechanics':train_mech,'numerical_sensitivity':{'maximum_abs_float32_vs_float64_similarity':float(abs(q-qd).max()),'multi_count_decision_changes':numeric},'recommendation':{'ready_for_matched_short_training':all(checks.values()),'checks':checks},'elapsed_seconds':time.monotonic()-start}
dump(ROOT/'analysis.json',result);dump(ROOT/'candidate_scores.json',records);dump(ROOT/'per_image_selections.json',per)
for name,rows in [('candidate_scores.csv',records),('quantity_score_matrix.csv',long),('per_image_selections.csv',per),('paired_contrasts.csv',stats),('candidate_mechanics.csv',mechanics),('training_reward_replay.csv',adv_rows),('training_group_mechanics.csv',train_mech),('own_shared_state_sensitivity.csv',shared_rows),('count_confusion.csv',confusion)]:
    fields=list(dict.fromkeys(k for x in rows for k in x))
    with (ROOT/name).open('w',encoding='utf-8',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in x.items()} for x in rows)
print(json.dumps({'recommendation':checks,'after_confirmation_hard_vs_pair':g['contrasts']['after']['hard_minus_pair']['macro']['B_hard_correct'],'after_confirmation_hard_vs_uniform':g['contrasts']['after']['hard_minus_uniform']['macro']['B_hard_correct'],'own_shared':g['own_shared']['after']['macro']['B_hard_correct'],'candidate_mechanics':g['candidate_mechanics']['after'],'numerical_changes':len(numeric)},ensure_ascii=False,indent=2))

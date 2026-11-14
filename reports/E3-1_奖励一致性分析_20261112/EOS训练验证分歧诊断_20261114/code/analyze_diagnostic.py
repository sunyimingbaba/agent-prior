import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
import torch
from diagnostic_io import read,dump,hydrate,paired_records

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;plan=read(r/'plan.json');parent=hydrate(r)
torch.set_num_threads(2);cases=read(r/'inputs/evaluation_cases.json');by={x['image']:x for x in cases};new,old=paired_records(parent)
nt=torch.load(parent/'scoring/tensors.pt',map_location='cpu',weights_only=True);ot=torch.load(parent/'reference_run/scoring/tensors.pt',map_location='cpu',weights_only=True)
ni={x:i for i,x in enumerate(nt['record_ids'])};oi={x:i for i,x in enumerate(ot['record_ids'])};gallery=read(parent/'features/gallery.json');texts=gallery['quantity_text_order']
words={x:i+1 for i,x in enumerate(['one','two','three','four','five','six','seven'])}
def choices(item,wording='B'):
    template=item['pairs'][wording]['positive'].rsplit(' ',1)[0]+' '
    return {words[t.rsplit(' ',1)[1].rstrip('.')]:i for i,t in enumerate(texts) if t.startswith(template)}
def prefer(q,item):
    cs=choices(item);assert 1 in cs and 2 in cs
    return float(q[cs[2]]-q[cs[1]]),max(sorted(cs),key=lambda k:float(q[cs[k]]))
def seed_for(name):return (plan['bootstrap']['seed']+int(hashlib.sha256(name.encode()).hexdigest()[:8],16))%2**32
def stat(values,label):
    d=np.asarray(values,dtype=np.float64);rng=np.random.default_rng(seed_for(label));ix=rng.integers(0,len(d),size=(10000,len(d)));ci=np.percentile(d[ix].mean(axis=1),[2.5,97.5])
    return {'mean':float(d.mean()),'interval95':ci.tolist(),'images':len(d),'seed':seed_for(label)}
pairs=[]
for x in new:
    item=by[x['image']];o=old[x['image'],x['role'],x['seed']];nq=nt['quantity_similarities'][ni[x['id']]];oq=ot['quantity_similarities'][oi[o['id']]]
    bp,bc=prefer(oq,item);ap,ac=prefer(nq,item)
    nh=torch.load(parent/x['state_file'],map_location='cpu',weights_only=True).float();oh=torch.load(parent/'reference_run'/o['state_file'],map_location='cpu',weights_only=True).float()
    tr=read(parent/x['trace_file']);prev=read(parent/'reference_run'/o['trace_file']);assert tr['rounds'][-1]['generated_text']==prev['rounds'][-1]['generated_text'] and x['quality_usable']==o['quality_usable']
    pairs.append({'image':x['image'],'split':x['split'],'role':x['role'],'seed':x['seed'],'new_id':x['id'],'old_id':o['id'],
     'before_pairacc_B':float(o['correct_B']),'after_pairacc_B':float(x['correct_B']),'delta_pairacc_B':float(x['correct_B'])-float(o['correct_B']),
     'before_margin_B':o['margin_B'],'after_margin_B':x['margin_B'],'delta_margin_B':x['margin_B']-o['margin_B'],
     'before_pairacc_A':float(o['correct_A']),'after_pairacc_A':float(x['correct_A']),
     'before_two_minus_one':bp,'after_two_minus_one':ap,'delta_two_minus_one':ap-bp,
     'before_argmax_two':float(bc==2),'after_argmax_two':float(ac==2),'hidden_cosine':float(torch.nn.functional.cosine_similarity(nh,oh,dim=0)),
     'same_final_text':True,'quality_unchanged':True})
metrics=[k for k in pairs[0] if k.startswith(('before_','after_','delta_')) or k=='hidden_cosine'];per=[]
for item in cases:
    word=next(k for k,v in words.items() if v==item['actual_count']);mention=bool(re.search(r'\b'+word+r'\b',item['original_first_caption'],re.I))
    for role in ['greedy','candidate']:
        xs=[x for x in pairs if x['image']==item['image'] and x['role']==role];assert len(xs)==(1 if role=='greedy' else 4)
        per.append({'image':item['image'],'split':item['split'],'selector':'greedy' if role=='greedy' else 'sample_mean','actual_count':item['actual_count'],'object':item['object'],
          'B_pair':str(item['actual_count'])+'_vs_'+str(item['pairs']['B']['wrong_count']),'first_caption_exact_true_number_word':mention,
          **{k:math.fsum(x[k] for x in xs)/len(xs) for k in metrics}})
summaries=[]
for selector in ['greedy','sample_mean']:
    for split in ['train','val']:
        allrows=[x for x in per if x['selector']==selector and x['split']==split]
        for dimension in ['all',*plan['groups']]:
            keys=['all'] if dimension=='all' else sorted({str(x[dimension]) for x in allrows})
            for key in keys:
                xs=allrows if dimension=='all' else [x for x in allrows if str(x[dimension])==key];label=selector+'/'+split+'/'+dimension+'/'+key
                d={k:math.fsum(x[k] for x in xs)/len(xs) for k in metrics}
                summaries.append({'selector':selector,'split':split,'dimension':dimension,'group':key,'images':len(xs),**d,
                  'relative_pairacc_B_change':d['delta_pairacc_B']/d['before_pairacc_B'] if d['before_pairacc_B'] else None,
                  'delta_pairacc_B_interval95':stat([x['delta_pairacc_B'] for x in xs],label+'/pairacc')['interval95'],
                  'delta_margin_B_interval95':stat([x['delta_margin_B'] for x in xs],label+'/margin')['interval95']})
standardized=[];decomposition=[]
for selector in ['greedy','sample_mean']:
    strata={s:{c:[x for x in per if x['selector']==selector and x['split']==s and x['actual_count']==c] for c in [1,2,3,4]} for s in ['train','val']}
    for split in ['train','val']:
        rng=np.random.default_rng(seed_for(selector+'/'+split+'/macro'));boot=np.zeros((10000,2));point=np.zeros(2)
        for c in [1,2,3,4]:
            xs=strata[split][c];v=np.array([[x['before_pairacc_B'],x['after_pairacc_B']] for x in xs]);point+=v.mean(axis=0)/4
            ix=rng.integers(0,len(xs),size=(10000,len(xs)));boot+=v[ix].mean(axis=1)/4
        standardized.append({'selector':selector,'split':split,'counts':[1,2,3,4],'images':sum(len(z) for z in strata[split].values()),
            'before_pairacc_B':float(point[0]),'after_pairacc_B':float(point[1]),'delta_pairacc_B':float(point[1]-point[0]),
            'delta_pairacc_B_interval95':np.percentile(boot[:,1]-boot[:,0],[2.5,97.5]).tolist()})
    n={s:sum(len(z) for z in strata[s].values()) for s in ['train','val']};parts=[]
    for c in [1,2,3,4]:
        wt,wv=[len(strata[s][c])/n[s] for s in ['train','val']];dt,dv=[np.mean([x['delta_pairacc_B'] for x in strata[s][c]]) for s in ['train','val']]
        parts.append({'count':c,'train_images':len(strata['train'][c]),'val_images':len(strata['val'][c]),'train_weight':wt,'val_weight':wv,'train_delta':float(dt),'val_delta':float(dv),
            'composition':float((wt-wv)*(dt+dv)/2),'within_count':float((wt+wv)*(dt-dv)/2)})
    composition=math.fsum(x['composition'] for x in parts);within=math.fsum(x['within_count'] for x in parts)
    decomposition.append({'selector':selector,'common_counts':[1,2,3,4],'train_images':n['train'],'val_images':n['val'],
        'split_delta_gap':composition+within,'composition_component':composition,'within_count_component':within,'composition_fraction':composition/(composition+within),'parts':parts})
scores=read(r/'scoring/scores.json');ct=torch.load(r/'scoring/tensors.pt',map_location='cpu',weights_only=True);lookup={x:i for i,x in enumerate(ct['record_ids'])};control_rows=[]
for x in scores:
    q=ct['quantity_similarities'][lookup[x['id']]];pref,count=prefer(q,by[x['image']])
    control_rows.append({**x,'two_minus_one':pref,'argmax_two':float(count==2),'argmax_count':count,'actual_count':by[x['image']]['actual_count']})
control_summary=[]
for arm in plan['controls']+plan['anchors']:
    for split in ['train','val']:
        xs=[x for x in control_rows if x['arm']==arm and x['split']==split]
        control_summary.append({'arm':arm,'split':split,'images':len(xs),'pairacc_B':float(np.mean([x['correct_B'] for x in xs])),
          'margin_B':float(np.mean([x['margin_B'] for x in xs])),'two_minus_one':float(np.mean([x['two_minus_one'] for x in xs])),
          'argmax_two':float(np.mean([x['argmax_two'] for x in xs])),'retrieval_rr':float(np.mean([x['retrieval_rr'] for x in xs])) if split=='val' else None})
shared=[]
for split in ['train','val']:
    names=[x['image'] for x in cases if x['split']==split];idx={(x['arm'],x['image']):x for x in control_rows}
    for metric in ['two_minus_one','margin_B','correct_B']:
        vals=[float(idx['shared_eos',name][metric])-float(idx['shared_answer_close',name][metric]) for name in names]
        shared.append({'split':split,'metric':metric,**stat(vals,'shared/'+split+'/'+metric)})
result={'ok':True,'role':'exploratory split and state-bias diagnostic','optimizer_updates':0,'new_generation_records':0,'per_image':per,
 'stratified':summaries,'count_standardized':standardized,'decomposition':decomposition,'control_summary':control_summary,'shared_state_contrasts':shared,
 'source_records':425,'all_final_texts_same':True,'all_quality_labels_unchanged':True,'scoring':read(r/'scoring/result.json'),'checkpoint_contracts':read(r/'checkpoint_cache_contracts.json')}
dump(r/'analysis.json',result)
for name,rows in [('paired_records.csv',pairs),('per_image_differences.csv',per),('stratified_results.csv',summaries),('count_standardized.csv',standardized),('control_scores.csv',control_rows),('control_summary.csv',control_summary)]:
    with (r/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in x.items()} for x in rows)
print(json.dumps({'standardized':standardized,'decomposition':[{k:v for k,v in d.items() if k!='parts'} for d in decomposition],'control_summary':control_summary},ensure_ascii=False),flush=True)

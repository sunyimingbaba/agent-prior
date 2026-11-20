"""Recompute cosine margins from saved embeddings, verify paired labels, aggregate by split."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import numpy as np
import torch

r=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save_csv(p, rows):
    with p.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
cases=read(r/'inputs/cases.json')
previous=read(Path(sys.argv[1]) if len(sys.argv)>1 else r/'inputs/historical_quantity.json')
inventory=read(r/'inputs/scored_images.json')
keys=lambda row:(row['image'],row['dimension'],row['wording'])
old={(x['model'],x['image'],x['wording'][0]):x for x in previous}
flat=[];summary=[];reversals=[];allrows={};worst=0;tool_coverage={}
for model in ('CLIP','CUP_CCB','ASPF','AGSE','CLIP_legacy','AGSE_archived'):
    p=r/'outputs'/model
    settings=read(p/'settings.json')
    assert settings['cases_sha256']==sha(r/'inputs/cases.json')
    assert settings['feature_sha256']==sha(p/'features.pt')
    f=torch.load(p/'features.pt',map_location='cpu',weights_only=True)
    I=f['image_features'].numpy().astype('float64');T=f['text_features'].numpy().astype('float64')
    assert f['image_names']==[x['image'] for x in inventory] and len(f['texts'])==len(set(f['texts']))
    rows=read(p/'rows.json');assert len(rows)==len(cases)
    grouped=collections.defaultdict(list); rm={keys(v):v for v in rows};allrows[model]=rm
    for row,case in zip(rows,cases):
        assert all(row[k]==v for k,v in case.items()) and row['model']==model
        name=row['image']; vi=I[f['image_names'].index(name)]
        pos=float(vi@T[f['texts'].index(row['positive'])]); neg=float(vi@T[f['texts'].index(row['negative'])])
        err=max(abs(pos-row['positive_cos']),abs(neg-row['negative_cos']),abs(pos-neg-row['margin']))
        assert err<1e-10 and bool(row['correct'])==(pos>neg);worst=max(worst,err)
        cat=row['dimension']; group=cat+'_'+row['wording'] if cat=='quantity' else cat
        for split in (row['split'],'all'):grouped[split,group].append(row)
        if cat!='quantity':
            other=rm[row['matched_image'],cat,row['wording']]
            assert row['positive']==other['negative'] and row['negative']==other['positive'] and row['split']==other['split'] and row['image']==other['matched_image']
            if row['side']=='a':
                reversals.append(dict(model=model,split=row['split'],dimension=cat,template=row['wording'],image_a=name,image_b=row['matched_image'],correct_images=int(row['correct'])+int(other['correct']),both_correct=row['correct'] and other['correct']))
        flat.append({k:row[k] for k in ('model','split','dimension','wording','image','positive','negative','positive_cos','negative_cos','margin','correct','tie','source')})
    for (split,group),x in sorted(grouped.items()):
        n=len(x);assert n==len({e['image'] for e in x})
        summary.append(dict(model=model,split=split,task=group,images=n,correct=sum(y['correct'] for y in x),accuracy=sum(y['correct'] for y in x)/n,mean_cosine_margin=statistics.mean(y['margin'] for y in x),ties=sum(y['tie'] for y in x)))
    if model in ('ASPF','AGSE'):
        used=0
        for x in inventory:
            stem=x['image'].split('.')[0];p0=r/'priors'/model
            info=read(p0/(stem+'.json'));hfile=p0/(stem+'.pt')
            assert info['input']==x and info['method']==model and info['query']=='Describe the visible scene.' and info['sha256']==sha(hfile)
            h=torch.load(hfile,map_location='cpu',weights_only=True)
            assert h.dtype==torch.bfloat16 and h.shape==(3584,) and torch.isfinite(h).all()
            assert info['used']==len(info['events']);used+=bool(info['used'])
        assert settings['images_with_tool_use']==used;tool_coverage[model]=used
legacy=[]; historical_agse=[]
for case in cases:
    if case['dimension']!='quantity':continue
    for historical,new,target in [('CLIP','CLIP_legacy',legacy),('AGSE','AGSE',historical_agse)]:
        prv=old[historical,case['image'],case['wording']];v=allrows[new][keys(case)]
        assert {prv['text1'],prv['text2']}=={case['positive'],case['negative']}
        target.append(dict(image=case['image'],split=case['split'],wording=case['wording'],old_margin=prv['margin'],new_margin=v['margin'],old_correct=prv['correct'],new_correct=v['correct']))
archived_errors=[]
for v in allrows['AGSE_archived'].values():
    if v['dimension']=='quantity':
        prev=old['AGSE',v['image'],v['wording']]
        assert prev['correct']==v['correct']
        archived_errors.append(abs(prev['margin']-v['margin']))
assert len(archived_errors)==200
save_csv(r/'baseline_summary.csv',summary);save_csv(r/'pair_scores.csv',flat);save_csv(r/'reversal_check.csv',reversals)
save_csv(r/'original_query_vs_generic_AGSE.csv',historical_agse)
proof=dict(models=len(allrows),archived_agse_agreement=len(archived_errors),archived_agse_max_margin_difference=max(archived_errors),unique_scored_images=len(inventory),items_per_method=len(cases),scores_verified=len(flat),max_fp64_difference=worst,tool_use_on_scored_images=tool_coverage,legacy_clip_agreement=sum(x['old_correct']==x['new_correct'] for x in legacy),legacy_clip_max_margin_difference=max(abs(x['old_margin']-x['new_margin']) for x in legacy),historical_quantity_comparisons=len(legacy),manifest_cases_sha256=sha(r/'inputs/cases.json'))
(r/'score_verification.json').write_text(json.dumps(proof,indent=2))
print(proof)

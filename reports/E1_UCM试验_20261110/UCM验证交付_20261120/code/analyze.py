"""Rebuild the scores from frozen embeddings and analyse paired validation comparisons."""
import csv
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import torch
from dimensions import report_dimension

ROOT=Path(__file__).resolve().parents[1]
PREV=ROOT.parent/'三类细节基线_20261120'
METHODS=['CLIP','CUP_CCB','ASPF','AGSE','AGSE_archived','AGSE_scene']
def load(p):
    return json.loads(p.read_bytes())
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(name, rows):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
def key(r):
    return r['cohort'],r['image'],r['template']
def mask(name,r):
    c,d,t=r['cohort'],r['dimension'],r['template']
    if name in ('quantity_A','quantity_B'):
        return c=='previous' and d=='quantity' and t==name[-1]
    if name.startswith('balanced_'):
        return c=='quantity_reversed' and t==name[-1]
    if name.startswith('locked_'):
        return c=='extension' and r['locked_dimension']==name[7:]
    if name.startswith('old_'):
        return c=='previous' and d==name[4:]
    if name.startswith('combined_'):
        return c in ('extension','previous') and d==name[9:]
    if name.startswith('source_'):
        return c=='extension' and d==name[7:] and r['source'].startswith('captions:')
    if name=='attribute_without_axis':
        return c=='extension' and d=='attribute' and r['subtype']!='orientation'
    if name=='relation_named_refs':
        return c=='extension' and d=='relation' and t in {'water_shore','road_over','river_banks','house_surround','shrub_water',
            'roof_field','roof_drive','shadow_tank','shadow_direction','infield_side','court_neighbors','junction_corners','plane_parallel'}
    return c=='extension' and d==name
def intervals(diffs,groups):
    grouped=defaultdict(list)
    for x,g in zip(diffs,groups):
        grouped[g].append(x)
    units=list(grouped.values())
    rng=np.random.default_rng(20261120)
    picks=rng.integers(len(units),size=(20000,len(units)))
    sums=np.array(list(map(sum,units)))[picks].sum(1)
    sizes=np.array(list(map(len,units)))[picks].sum(1)
    return list(np.quantile(sums/sizes,[.025,.975]))
def main():
    cases=load(ROOT/'inputs/cases.json')
    images={r['image']:r for r in load(ROOT/'inputs/candidates.json')}
    keys={key(r) for r in cases}
    assert len(keys)==len(cases)
    with (PREV/'pair_scores.csv').open(encoding='utf-8-sig') as f:
        old={(r['model'],r['image'],r['wording']):r for r in csv.DictReader(f) if r['split']=='val'}
    bound=load(ROOT/'inputs/lock.json')
    assert all(sha(ROOT/k)==v for k,v in bound.items())
    modelrows={}
    allrows=[]
    ver=[]
    gen=[]
    for m in METHODS:
        folder=ROOT/'outputs'/m
        settings=load(folder/'settings.json')
        bank=torch.load(folder/'features.pt',map_location='cpu',weights_only=True)
        assert sha(folder/'features.pt')==settings['features_sha256'] and sha(ROOT/'inputs/lock.json')==settings['lock_sha256']
        scores=bank['image_features'].double().numpy()@bank['text_features'].double().numpy().T
        received=load(folder/'rows.json')
        assert len(received)==len(cases) and [{k:v for k,v in r.items() if k!='scores'} for r in received]==cases
        rows={}
        core_errors=[]
        parity=[]
        parity_flips=0
        for r in received:
            r['locked_dimension']=r['dimension']
            r['dimension']=report_dimension(r['dimension'],r['template'])
            s=r.pop('scores')
            ix=bank['images'].index(r['image'])
            for k,v in s.items():
                core_errors.append(abs(scores[ix,bank['texts'].index(r[k])]-v))
            p,g,w=[s[k] for k in ('positive','generic','negative')]
            row=dict(**r,model=m,pos=p,bland=g,neg=w,margin=p-w,correct=int(p>w),tie=int(p==w),p_above_g=int(p>g),g_above_w=int(g>w),
                     p_top=int(p>max(g,w)),g_top=int(g>max(p,w)),w_top=int(w>max(p,g)),strict_p_g_w=int(p>g>w),
                     ranking_tie=int(len({p,g,w})<3))
            if r['cohort']=='previous' and m!='AGSE_scene':
                prev=old[m,r['image'],r['template']]
                assert prev['positive']==r['positive'] and prev['negative']==r['negative']
                parity.append(abs(float(prev['margin'])-row['margin']))
                parity_flips+=int((prev['correct']=='True')!=bool(row['correct']))
            rows[key(r)]=row
            allrows.append(row)
        assert max(core_errors)<1e-12 and parity_flips==0 and max(parity,default=0)<2e-6
        ver.append(dict(method=m,raw_cosines=len(core_errors),max_error=max(core_errors),previous_items=len(parity),
                        previous_max_margin_diff=max(parity,default=None),previous_decision_flips=parity_flips))
        modelrows[m]=rows
        if m in ('ASPF','AGSE','AGSE_scene'):
            for name in bank['images']:
                pr=ROOT/'priors'/m/Path(name).with_suffix('.json')
                trace=load(pr)
                assert sha(pr.with_suffix('.pt'))==settings['priors_sha256'][name]==trace['sha256']
                assert trace['rgb_sha256']==images[name]['rgb_sha256']
                last=trace['rounds'][-1]['decoded']
                gen.append(dict(method=m,image=name,query=trace['query'],n_rounds=len(trace['rounds']),tools=trace['used'],
                                first_single_eos=int(trace['first_eos']),last_visible=int(bool(last.strip())),
                                addCriterion=int('addCriterion' in last),last_text=last,old_bits=trace['old_bit_exact']))
    write_csv('逐条三句得分.csv',allrows)
    write_csv('逐图生成.csv',gen)
    write_csv('数值复算.csv',ver)
    subsets=['quantity_A','quantity_B','attribute','relation','old_attribute','old_relation','combined_attribute','combined_relation',
             'balanced_A','balanced_B','source_attribute','source_relation','attribute_without_axis','relation_named_refs','locked_attribute','locked_relation']
    summary=[]
    reversals=[]
    contrasts=[]
    for group in subsets:
        for m,rr in modelrows.items():
            items=[r for r in rr.values() if mask(group,r)]
            assert items
            pairs=defaultdict(list)
            for r in items:
                pairs[(r['template'],r['pair'])].append(r)
            pair_groups=[]
            if group not in ('quantity_A','quantity_B'):
                for (tpl,pair),v in pairs.items():
                    left,right=v
                    assert left['positive']==right['negative'] and left['negative']==right['positive'] and left['image']!=right['image']
                    delta=left['margin']+right['margin']
                    pair_groups.append(dict(subset=group,model=m,template=tpl,pair=pair,scene=left['scene'],correct=sum(i['correct'] for i in v),
                                            assignment_correct=int(delta>0),assignment_tie=int(delta==0),assignment_margin=delta))
                reversals.extend(pair_groups)
            base=[modelrows['CUP_CCB'][key(r)] for r in items]
            diffs=[r['correct']-b['correct'] for r,b in zip(items,base)]
            per_pair=intervals(diffs,[r['pair'] for r in items])
            scene_ci=intervals(diffs,[r['scene'] for r in items])
            contrasts.append(dict(subset=group,model=m,reference='CUP_CCB',n=len(items),win=diffs.count(1),loss=diffs.count(-1),
                                  delta=np.mean(diffs),ci_cluster_low=per_pair[0],ci_cluster_high=per_pair[1],ci_scene_low=scene_ci[0],ci_scene_high=scene_ci[1]))
            metrics={k:sum(r[k] for r in items) for k in ('correct','tie','p_above_g','g_above_w','p_top','g_top','w_top','strict_p_g_w','ranking_tie')}
            if m=='AGSE_scene':
                for k in ('p_above_g','g_above_w','p_top','g_top','w_top','strict_p_g_w','ranking_tie'):
                    metrics[k]=''
            summary.append(dict(subset=group,model=m,n=len(items),unique_images=len({r['image'] for r in items}),
                            scene_count=len({r['scene'] for r in items}),**metrics,mean_margin=np.mean([r['margin'] for r in items]),
                            mean_pos_minus_generic=np.mean([r['pos']-r['bland'] for r in items]) if m!='AGSE_scene' else '',
                            mean_generic_minus_neg=np.mean([r['bland']-r['neg'] for r in items]) if m!='AGSE_scene' else '',
                            n_pairs=len(pair_groups),both_correct=sum(r['correct']==2 for r in pair_groups),both_wrong=sum(r['correct']==0 for r in pair_groups),
                            same_preference=sum(r['correct']==1 for r in pair_groups),assignment_correct=sum(r['assignment_correct'] for r in pair_groups)))
    write_csv('分项结果.csv',summary)
    write_csv('相对CUP配对差.csv',contrasts)
    write_csv('反向图对.csv',reversals)
    gensummary=[dict(method=m,images=len(group),eos=sum(r['first_single_eos'] for r in group),visible=sum(r['last_visible'] for r in group),
                     tools=sum(r['tools']>0 for r in group),abnormal_string=sum(r['addCriterion'] for r in group))
                for m in ('ASPF','AGSE','AGSE_scene') for group in [[r for r in gen if r['method']==m]]]
    write_csv('生成汇总.csv',gensummary)
    print(len(allrows),'model-items,',sum(r['raw_cosines'] for r in ver),'cosines verified')
if __name__=='__main__':
    main()

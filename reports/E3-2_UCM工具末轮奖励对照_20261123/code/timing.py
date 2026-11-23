"""Compare reward timing within each arm and compute the paired change in the R2 vs R1 advantage."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from combine import aligned
from metrics import ci

ROOT=Path(__file__).resolve().parents[1]
PREV=ROOT.parent/'E3-2_UCM冷启动三组短训_20261123'
SEEDS=('.', 'repeat')
ARMS=('R0','R1','R2')

def read(path):
    return json.loads(path.read_bytes())

def write(path, obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def paired(deltas):
    stats=ci(np.mean(deltas,axis=0))
    stats['by_seed']=[float(v.mean()) for v in deltas]
    return stats

def timing():
    specs=[('retrieval_by_image.csv','',m) for m in ('mR','I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10')]
    specs.extend(('detail_by_case.csv',g,'correct') for g in ('quantity_A','quantity_B','attribute','relation'))
    result={}
    for name,group,indicator in specs:
        comparisons=defaultdict(list)
        for sub in SEEDS:
            v={}
            keys=None
            for run,base in [('early',PREV),('final',ROOT)]:
                for arm in ARMS:
                    x=aligned(base/sub,name,group,indicator,arm)
                    if keys is None:
                        keys=sorted(x)
                    assert sorted(x)==keys
                    v[run,arm]=100*np.array([x[k] for k in keys])
            assert np.array_equal(v['early','R0'],v['final','R0'])
            r1=v['final','R1']-v['early','R1']
            r2=v['final','R2']-v['early','R2']
            comparisons['R1_final_minus_early'].append(r1)
            comparisons['R2_final_minus_early'].append(r2)
            comparisons['change_in_R2_vs_R1'].append(r2-r1)
        result[group or indicator]={k:paired(d) for k,d in comparisons.items()}
    write(ROOT/'results/timing_contrasts.json',result)

def behavior_changes():
    result={}
    for sub in SEEDS:
        rows={}
        for label,root in [('early',PREV),('final',ROOT)]:
            rows[label]={}
            for filename,field in [('behavior_by_image.csv','first_eos_only'),('retrieval_by_image.csv','mR')]:
                with (root/sub/'results'/filename).open(encoding='utf-8-sig') as f:
                    rows[label][field]={(x['arm'],x['image']):x[field] for x in csv.DictReader(f)}
        summarized={}
        for arm in ('R1','R2'):
            cells=defaultdict(list)
            for key,v in rows['early']['mR'].items():
                if key[0]!=arm:
                    continue
                start=rows['early']['first_eos_only'][key]=='True'
                end=rows['final']['first_eos_only'][key]=='True'
                cells[f'{int(start)}_to_{int(end)}'].append(100*(float(rows['final']['mR'][key])-float(v)))
            summarized[arm]={k:dict(images=len(x),mean_mR_delta_pp=float(np.mean(x)),
                contribution_to_overall_pp=float(sum(x)/210)) for k,x in cells.items()}
        result[sub]=summarized
    write(ROOT/'results/first_eos_changes.json',result)

def audit():
    result={}
    for sub in SEEDS:
        cur=ROOT/sub/'records'
        old=PREV/sub/'records'
        assert read(cur/'initial.json')==read(old/'initial.json')
        assert read(cur/'R0/validation.json')==read(old/'R0/validation.json')
        arms={}
        steps={}
        for a in ('R1','R2'):
            new=read(cur/a/'steps.json')
            previous=read(old/a/'steps.json')
            assert [(s['image'],s['row'],s['seed']) for s in new]==[(s['image'],s['row'],s['seed']) for s in previous]
            assert new[0]['trajectories']==previous[0]['trajectories']
            total,called,changed=0,0,0
            tool_advantages=[]
            for st in new:
                ranks=1/(1+np.array(st['proxy_rank']))
                comp=np.array(st['retrieval'])
                if a=='R1':
                    assert np.array_equal(comp,ranks)
                else:
                    margin=np.array(st['native_margin'])
                    order=np.argsort(margin,kind='stable')
                    ordered=np.sort(ranks)
                    for level in np.unique(margin):
                        group=margin[order]==level
                        assert np.allclose(comp[order[group]],ordered[group].mean(),rtol=0,atol=1e-14)
                score=comp+np.array(st['format'])
                assert np.allclose(score,st['total']) and np.allclose((score-score.mean())/(score.std(ddof=1)+1e-8),st['advantage'])
                assert len(st['terminal'])==8
                uses=np.array([z['tool_calls']>0 for z in st['terminal']])
                if uses.any() and (~uses).any():
                    adv=np.array(st['advantage'])
                    tool_advantages.append(float(adv[uses].mean()-adv[~uses].mean()))
                for z in st['terminal']:
                    total+=1
                    if z['tool_calls']:
                        assert len(z['continuation_ids'])==min(z['tool_calls'],2) and z['tool_response']
                        changed+=z['final_sha']!=z['first_sha']
                        called+=1
                    else:
                        assert z['final_sha']==z['first_sha'] and z['continuation_ids']==[] and z['tool_response'] is None
            steps[a]=new
            arms[a]=dict(candidates=total,called=called,changed_after_tool=changed,groups_with_both=len(tool_advantages),
                groups_favoring_tool=sum(d>0 for d in tool_advantages),mean_tool_advantage_gap=float(np.mean(tool_advantages)),updates=sum(s['norm']>0 for s in new))
        assert steps['R1'][0]['terminal']==steps['R2'][0]['terminal']
        assert steps['R1'][0]['proxy_rank']==steps['R2'][0]['proxy_rank']
        result[sub]=arms
    write(ROOT/'results/training_audit.json',result)

def panel():
    values=defaultdict(list)
    levels={}
    behaviors={}
    for sub in SEEDS:
        behaviors[sub]={}
        arms={}
        for arm in ['R0','old_R1','old_R2','R1','R2']:
            root=ROOT/sub/'panel'
            csvpath=root/(arm+'.csv')
            meta=read(root/(arm+'.json'))
            assert hashlib.sha256(csvpath.read_bytes()).hexdigest()==meta['sha256']
            with csvpath.open(encoding='utf-8-sig') as f:
                raw=list(csv.DictReader(f))
            assert len(raw)==192 and len(set((r['image'],r['slot']) for r in raw))==192
            with_tool=[float(r['margin']) for r in raw if int(r['tools'])>0]
            without_tool=[float(r['margin']) for r in raw if int(r['tools'])==0]
            behaviors[sub][arm]=dict(candidates=len(raw),called=len(with_tool),
                margin_with_tool=float(np.mean(with_tool)) if with_tool else None,
                margin_without_tool=float(np.mean(without_tool)) if without_tool else None)
            pair=defaultdict(list)
            for row in raw:
                pair[row['pair']].append(float(row['margin']))
            assert len(pair)==12 and all(len(v)==16 for v in pair.values())
            arms[arm]={k:np.mean(v) for k,v in pair.items()}
        levels[sub]={a:float(np.mean(list(x.values()))) for a,x in arms.items()}
        keys=sorted(arms['R0'])
        for a,b in [('R2','R1'),('R2','R0'),('R1','R0'),('R2','old_R2'),('R1','old_R1')]:
            assert sorted(arms[a])==sorted(arms[b])==keys
            values[a+'-'+b].append(np.array([arms[a][k]-arms[b][k] for k in keys]))
        values['change_in_R2_vs_R1'].append(values['R2-old_R2'][-1]-values['R1-old_R1'][-1])
    write(ROOT/'results/terminal_training_panel.json',dict(levels=levels,behavior=behaviors,contrasts={k:paired(d) for k,d in values.items()}))

if __name__=='__main__':
    timing()
    behavior_changes()
    audit()
    panel()

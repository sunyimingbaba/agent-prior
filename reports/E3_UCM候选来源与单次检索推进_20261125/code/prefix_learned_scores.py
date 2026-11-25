"""Score complete paired RFPO endpoints. Preserve single-seed and two-seed estimates separately."""
from collections import defaultdict
import sys
import numpy as np
from rosters import ROOT,read,sha,dump
from report_scores import load
from single_scores import RFPO,FORM,calculate
from run_stage import boot
from orders import score,advantages,calibrate
from gate import K

ARMS=('locked_R3','free_R3','locked_R2','free_R2')
CONTRASTS=(('free_R3','locked_R3'),('free_R2','locked_R2'),('free_R3','free_R2'),
           ('locked_R3','locked_R2'),('free_R3','locked_R2'),('free_R3','retained'),('locked_R3','retained'),('free_R3','R0'),('locked_R3','R0'),('free_R2','R0'),('locked_R2','R0'))

def summarize(runs,groups):
    return {k:dict(**{t:x for t,x in boot(np.stack([r[k] for r in runs.values()]).mean(0),np.ones(len(groups)),groups).items()
                      if t in ('mean','interval')},seeds={s:float(r[k].mean()) for s,r in runs.items()})
            for k in next(iter(runs.values()))}

def audit(tag,decision):
    seed,mode,reward=tag.split('_');p=ROOT/'pilot'/tag;proof=read(p/'complete.json');rows=read(p/'steps.json')
    plan=read(ROOT/'inputs/training_roster.json')['rows']
    assert proof['nomination']==decision and proof['roster']==sha(ROOT/'inputs/training_roster.json')
    assert proof['steps_sha']==sha(p/'steps.json') and len(rows)==proof['steps']==len(plan)
    for i,(step,c) in enumerate(zip(rows,plan)):
        assert step['image']==c['image'] and step['step']==i+1 and step['prompts']==['scene','scene']
        assert step['rngs']==[(13100000 if seed=='a' else 23100000)+1000*i+100*j for j in range(2)]
        assert len(step['paths'])==16 and all(1<=len(z)<=3 for z in step['paths'])
        if mode=='locked':assert len({tuple(z[0]) for z in step['paths']})==1
        assert step['sampled_tokens']==sum(len(ids) for path in step['paths'] for ids in path[(mode=='locked'):])
        rw,floor=score(step['scores'],K);assert abs(floor-step['floor'])<1e-12
        if mode=='ordinal':
            assert np.allclose(rw['R3'],step['rewards']['original_R3'],atol=1e-12,rtol=0)
            sc=step['scores'];native,hard,keep=(np.array([r[k] for r in sc]) for k in ('native','A','keep'))
            dense=np.unique(native+K*hard,return_inverse=True)[1]
            rw['R3']=calibrate(dense+17*(keep+1e-14>=floor),np.arange(16))
        for k in ('R2','R3'):assert np.allclose(rw[k],step['rewards'][k],atol=1e-12,rtol=0)
        adv=advantages(rw[reward]);assert np.allclose(adv,step['advantages'],atol=1e-8,rtol=0)
        if not np.any(adv):assert step['grad_norm']==0.
    changed=sum(u['grad_norm']>0 for u in rows);assert changed==proof['updated']
    ns=np.array([u['replay_tokens'] for u in rows])
    return dict(initial=proof['initial'],final=proof['final'],steps=len(rows),updated=changed,
                sampled_tokens=sum(u['sampled_tokens'] for u in rows),replay_tokens=int(sum(ns)),
                average_replay_error=float(np.average([u['mean_abs_logp_error'] for u in rows],weights=ns)) if ns.sum() else None,
                max_replay_error=max(u['max_abs_logp_error'] for u in rows))

def main(kind):
    p=ROOT/'results'/(kind+'_nomination.json');assert read(p)['eligible']
    if kind=='prefix':
        arms=ARMS;contrasts=CONTRASTS
    else:
        assert kind=='resolution'
        base='free_R3' if read(ROOT/'results/prefix_nomination.json')['eligible'] else 'resolutionbase_R3'
        arms=(base,'ordinal_R3')
        contrasts=(('ordinal_R3',base),('ordinal_R3','retained'),('ordinal_R3','R0'),(base,'retained'))
    runs={};response={};individual=[]
    for seed in 'ab':
        for a in arms:
            tag=seed+'_'+a
            if (ROOT/'pilot'/tag/'complete.json').is_file():runs[tag]=audit(tag,sha(ROOT/'results/prefix_nomination.json') if kind=='resolution' and a=='free_R3' else sha(p))
        assert len({v['initial'] for t,v in runs.items() if t.startswith(seed+'_')})<=1
    for sub,rows in [('primary',read(RFPO/'inputs/panel.json')),('official',read(FORM/'inputs/ucm.json'))]:
        ids=[c['image'] for c in rows];groups=[c.get('pair',c['image']) for c in rows];levels=defaultdict(dict)
        for seed in 'ab':
            for arm in ('R0','retained')+arms:
                tag=seed+'_'+arm
                if arm in ('R0','retained'):
                    label='deployed' if arm=='retained' else arm
                    p=RFPO/'vectors'/(seed+'_'+label+'_'+sub+'.pt')
                    assert sha(p)==read(RFPO/'archives/index.json')['vectors'][p.name]
                    o=load(p)
                    if arm=='R0':
                        assert all(v['initial']==o['adapter'] for t,v in runs.items() if t.startswith(seed+'_'))
                else:
                    base=ROOT/'records/learned'/tag
                    if tag not in runs or not (base/(sub+'_read.json')).is_file():continue
                    f=base/(sub+'_features.pt');proof=read(base/(sub+'_read.json'))
                    assert proof['features']==sha(f) and proof['archive']==sha(base/(sub+'.tar.gz'))
                    o=load(f);assert proof['adapter']==o['adapter']==runs[tag]['final']
                assert ids==o['images']
                scores=calculate(sub,o['features'],rows);levels[arm][seed]=scores
                for j,c in enumerate(rows):individual.append(dict(seed=seed,arm=arm,sub=sub,image=c['image'],**{k:float(v[j]) for k,v in scores.items()}))
        changes={}
        for a,b in contrasts:
            common=sorted(levels[a].keys() & levels[b].keys())
            if common:
                paired={s:{k:x-levels[b][s][k] for k,x in levels[a][s].items()} for s in common}
                changes[a+'_minus_'+b]=summarize(paired,groups)
        response[sub]=dict(levels={k:summarize(v,groups) for k,v in levels.items() if v},changes=changes)
    dump(ROOT/'results'/(kind+'_learned.json'),response);dump(ROOT/'results'/(kind+'_training_audit.json'),runs)
    dump(ROOT/'records'/(kind+'_learned_images.json'),individual)
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'prefix')

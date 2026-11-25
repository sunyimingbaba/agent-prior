"""Verify optimizer activity and compare the four equal-budget arms at their final checkpoints."""
import numpy as np
from rosters import ROOT,REP,sha,read,dump
from report_scores import load
from single_scores import RFPO,FORM,calculate,summary_by_seed
from orders import score,advantages
from gate import K

def main():
    nomination=read(ROOT/'results/nomination.json');selected=nomination['selected']
    assert selected
    plan=read(ROOT/'inputs/training_roster.json')['rows'];audit={};out={};individual=[]
    names=['scene_R2','scene_R3','selected_R2','selected_R3']
    for s in 'ab':
        initial=None
        for arm in names:
            tag=s+'_'+arm;p=ROOT/'pilot'/tag;cp=read(p/'complete.json');steps=read(p/'steps.json')
            assert cp['nomination']==sha(ROOT/'results/nomination.json') and cp['roster']==sha(ROOT/'inputs/training_roster.json')
            assert sha(p/'steps.json')==cp['steps_sha'] and len(steps)==cp['steps']==len(plan)
            assert initial is None or initial==cp['initial'];initial=cp['initial']
            errs=[];ns=[]
            for j,(row,ref) in enumerate(zip(steps,plan)):
                assert row['step']==j+1 and row['image']==ref['image']
                assert row['rngs']==[(13100000 if s=='a' else 23100000)+1000*j+n*100 for n in range(2)]
                assert len(row['paths'])==16 and all(1<=len(v)<=3 for v in row['paths'])
                rw,floor=score(row['scores'],K);assert abs(floor-row['floor'])<1e-12
                for m in ('R2','R3'):assert np.allclose(rw[m],row['rewards'][m],atol=1e-12,rtol=0)
                adv=advantages(rw[arm[-2:]])
                assert np.allclose(adv,row['advantages'],rtol=0,atol=1e-8)
                if not np.any(adv):assert row['grad_norm']==0
                ns.append(row['replay_tokens']);errs.append(row['mean_abs_logp_error'])
            audit[tag]=dict(updated=cp['updated'],actions=16*len(plan),
                replayed_tokens=sum(ns),avg_abs_replay_error=float(np.average(errs,weights=ns)) if sum(ns) else None,
                maximum_abs_error=max(r['max_abs_logp_error'] for r in steps),final=cp['final'])
    for sub,rows in [('primary',read(RFPO/'inputs/panel.json')),('official',read(FORM/'inputs/ucm.json'))]:
        labels=[c.get('pair',c['image']) for c in rows];ids=[r['image'] for r in rows]
        levels={n:{} for n in names+['retained']}
        for s in 'ab':
            for arm in levels:
                if arm=='retained':obj=load(RFPO/'vectors'/(s+'_deployed_'+sub+'.pt'))
                else:
                    tag=s+'_'+arm;p=ROOT/'records/learned'/tag;proof=read(p/(sub+'_read.json'));f=p/(sub+'_features.pt')
                    assert proof['features']==sha(f) and proof['archive']==sha(p/(sub+'.tar.gz'))
                    obj=load(f);assert obj['adapter']==audit[tag]['final']
                assert obj['images']==ids
                scores=calculate(sub,obj['features'],rows);levels[arm][s]=scores
                for j,c in enumerate(rows):
                    individual.append(dict(seed=s,arm=arm,sub=sub,image=c['image'],**{k:float(v[j]) for k,v in scores.items()}))
        changes={}
        for a,b in [('selected_R3','scene_R2'),('selected_R3','selected_R2'),('selected_R3','scene_R3'),('selected_R3','retained')]:
            z={s:{k:x-levels[b][s][k] for k,x in levels[a][s].items()} for s in 'ab'}
            changes[a+'_minus_'+b]=summary_by_seed(z,labels)
        out[sub]=dict(levels={a:summary_by_seed(z,labels) for a,z in levels.items()},changes=changes)
    dump(ROOT/'results/learned.json',out);dump(ROOT/'results/training_audit.json',audit)
    dump(ROOT/'records/learned_images.json',individual)
if __name__=='__main__':main()

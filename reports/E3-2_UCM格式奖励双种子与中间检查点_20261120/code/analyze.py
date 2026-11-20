"""Recompute paired endpoints, intermediate trajectories and the decoding control from native outputs."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

from shared import ROOT, read, dump, sha
from metrics import fixed, recalls, ci, csvwrite

E1SETS = ('quantity_A','quantity_B','attribute','relation')
SEEDS = ('a','b')
CHECKS = (6,12,24)


def scalar_ci(values):
    return ci(np.asarray(values,dtype=np.float64))


def study_states():
    return ['base_00']+[f'{seed}_{arm}_{step:02d}' for seed in SEEDS for arm in ('off','on') for step in CHECKS]+[f'{seed}_{arm}_24_force' for seed in SEEDS for arm in ('off','on')]


def e1_group(row):
    if row['cohort']=='previous' and row['dimension']=='quantity' and row['template'] in ('A','B'):
        return 'quantity_'+row['template']
    if row['cohort']=='extension':
        return 'relation' if row['template']=='shadow_direction' else row['dimension']
    return ''


def details(scores, rows, meta, texts, state):
    index={(r['image'],r['view']):i for i,r in enumerate(rows)}
    result=[]
    for r in meta:
        margins={}
        for word in ('C','D'):
            pos,neg=[texts.index(x) for x in r[word]]
            for view,sign in [('upright',1),('rotated',-1)]:
                ix=index[r['image'],view]
                margins[word,view]=float(sign*(scores[ix,pos]-scores[ix,neg]))
        result.append(dict(state=state,image=r['image'],
               **{w+'_'+v:z for (w,v),z in margins.items()},
               both=np.mean([margins[w,'upright']>0 and margins[w,'rotated']>0 for w in ('C','D')]),
               pairacc=np.mean([m>0 for m in margins.values()]),margin=np.mean(list(margins.values()))))
    return result


def summarize(baseline_only=False):
    root=ROOT
    text=torch.load(root/'inputs/retrieval_text.pt',map_location='cpu',weights_only=True)
    e1=root/'inputs/e1'
    for name,h in read(root/'inputs/e1_extension.json')['files'].items():
        assert sha(e1/name)==h
    e1bank=torch.load(e1/'features.pt',map_location='cpu',weights_only=True)
    et={s:i for i,s in enumerate(e1bank['texts'])}
    e1cases=read(e1/'cases.json')
    prior=read(e1/'rows.json')
    assert len(prior)==len(e1cases)==156
    details_meta=read(root/'inputs/detail.json')
    val=read(root/'inputs/validation.json')
    ids=[r['image'] for r in val]
    owner=np.arange(len(val))[:,None]==np.repeat(np.arange(len(val)),5)[None,:]
    retrieval=[]; relation=[]; e1rows=[]; trajectory=[]; complete={}; baseline_e1=[]
    for state in (['base_00'] if baseline_only else study_states()):
        out=root/'eval'/state
        proof=read(out/'complete.json')
        assert proof['text']==sha(root/'inputs/retrieval_text.pt') and proof['readout']==read(e1/'settings.json')['sha256']
        summary={}
        for part,h in proof['parts'].items():
            meta=read(out/(part+'.json'))
            assert sha(out/(part+'.json'))==h and meta['bundle']==sha(out/(part+'.pt')) and meta['active']==proof['active']
            bundle=torch.load(out/(part+'.pt'),map_location='cpu',weights_only=True)
            vs=bundle['visual']; hs=bundle['hidden']; rs=meta['records']
            assert len(rs)==len(vs)==len(hs)
            length=[];tools=[];eos=[];formats=[]
            for r,hidden in zip(rs,hs):
                assert r['hidden_sha']==hashlib.sha256(hidden.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                length.append(sum(len(x['ids']) for x in r['rounds']))
                tools.append(bool(r['events']));eos.append(r['immediate_eos']);formats.append(r['format']>0)
                trajectory.append(dict(state=state,part=part,image=r['image'],view=r['view'],tokens=length[-1],tool=int(tools[-1]),eos=int(eos[-1]),format=int(formats[-1])))
            summary[part]=dict(generations=len(rs),eos=sum(eos),tools=sum(tools),formatted=sum(formats),mean_tokens=np.mean(length))
            if part=='detail':
                rows=details(fixed(vs,text['relation']),rs,details_meta,text['texts'],state)
                relation.extend(rows)
                summary[part].update(both=int(round(2*sum(v['both'] for v in rows))),out_of=32,pairacc=float(np.mean([v['pairacc'] for v in rows])),margin=float(np.mean([v['margin'] for v in rows])))
                continue
            assert [r['image'] for r in rs]==ids
            scores=fixed(vs,text['native'])[:,text['caption_order']]
            i2t=recalls(scores,owner);t2i=recalls(scores.T,owner.T)
            row_scores=[]
            for i,image in enumerate(ids):
                vals={f'I2T_R{k}':float(i2t[k][i]) for k in (1,5,10)}
                vals.update({f'T2I_R{k}':float(t2i[k][i*5:i*5+5].mean()) for k in (1,5,10)})
                row_scores.append(dict(state=state,part=part,image=image,mR=np.mean(list(vals.values())),**vals))
            retrieval.extend(row_scores)
            summary[part]['mR']=float(np.mean([v['mR'] for v in row_scores]))
            summary[part]['recalls']={k:float(np.mean([v[k] for v in row_scores])) for k in vals}
            e1scores=fixed(vs,e1bank['text_features'])
            selected=defaultdict(list)
            for z,archived in zip(e1cases,prior,strict=True):
                group=e1_group(z)
                if group not in E1SETS:continue
                vi=ids.index(z['image'])
                pos,neg=e1scores[vi,et[z['positive']]],e1scores[vi,et[z['negative']]]
                margin=float(pos-neg)
                measured=dict(state=state,part=part,group=group,image=z['image'],pair=z['pair'],correct=int(margin>0),margin=margin)
                selected[group].append(measured);e1rows.append(measured)
                if state=='base_00' and part=='native':
                    ref=archived['scores']['positive']-archived['scores']['negative']
                    baseline_e1.append((abs(ref-margin),(margin>0)==(ref>0)))
            groups={}
            for g,entries in selected.items():
                pairs=defaultdict(list)
                for row in entries:pairs[row['pair']].append(row)
                groups[g]=dict(correct=sum(r['correct'] for r in entries),n=len(entries),margin=float(np.mean([r['margin'] for r in entries])))
                if g in ('relation','attribute'):
                    assert len(pairs)==18 and all(len(v)==2 for v in pairs.values())
                    groups[g]['both']=sum(all(r['correct'] for r in v) for v in pairs.values())
            assert [groups[g]['n'] for g in E1SETS]==[22,22,36,36]
            summary[part]['e1']=groups
        complete[state]=summary
    output=root/'scratch'/'baseline_check' if baseline_only else root/'results'
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('retrieval',retrieval),('detail',relation),('e1',e1rows),('behavior',trajectory)]:csvwrite(output/(name+'_by_source.csv'),rows)
    dump(output/'states.json',complete)
    baseline=complete['base_00']['native']['e1']
    assert [baseline[g]['correct'] for g in E1SETS]==[16,17,23,19] and all(bool(x[1]) for x in baseline_e1)
    dump(output/'e1_replay.json',dict(cases=len(baseline_e1),max_margin_difference=float(max(x[0] for x in baseline_e1)),all_decisions_match=True))
    if baseline_only:
        return
    # Build aligned vectors, keeping each training seed paired on the same source images.
    def aligned(state,part,metric):
        if metric=='mR':seq=[(r['image'],r['mR']) for r in retrieval if r['state']==state and r['part']==part]
        elif metric in ('both','pairacc','margin'):seq=[(r['image'],r[metric]) for r in relation if r['state']==state]
        else:seq=[(r['image'],r[metric]) for r in trajectory if r['state']==state and r['part']==part]
        seq.sort()
        return [r[0] for r in seq],np.asarray([r[1] for r in seq],dtype=np.float64)
    pairs={}
    comparisons=[]
    for step in CHECKS:
        route=lambda arm:[(f'{s}_{arm}_{step:02d}',f'{s}_off_{step:02d}' if arm=='on' else 'base_00') for s in SEEDS]
        for i,s in enumerate(SEEDS):
            comparisons.extend([(f'{s}_on-off_{step}',[route('on')[i]]),(f'{s}_on-base_{step}',[(f'{s}_on_{step:02d}','base_00')])])
        comparisons.extend([(f'mean_on-off_{step}',route('on')),(f'mean_on-base_{step}',[(f'{s}_on_{step:02d}','base_00') for s in SEEDS])])
    for arm in ('off','on'):
        items=[(f'{seed}_{arm}_24_force',f'{seed}_{arm}_24') for seed in SEEDS]
        for seed,item in zip(SEEDS,items):comparisons.append((f'{seed}_{arm}_force-natural',[item]))
        comparisons.append((f'mean_{arm}_force-natural',items))
    comparisons.append(('mean_on-off_force',[(f'{s}_on_24_force',f'{s}_off_24_force') for s in SEEDS]))
    for name,items in comparisons:
        result={}
        for part,m in [('native','mR'),('generic','mR'),('native','eos'),('native','format'),('native','tool'),('detail','both'),('detail','pairacc'),('detail','margin')]:
            if part=='generic' and 'force' in name:continue
            ds=[]
            for a,b in items:
                ai,x=aligned(a,part,m);bi,y=aligned(b,part,m);assert ai==bi and len(set(ai))==len(ai)
                ds.append(x-y)
            result[part+'_'+m]=scalar_ci(np.mean(ds,axis=0))
        pairs[name]=result
    dump(output/'paired_contrasts.json',pairs)
    base=complete['base_00'];decisions={}
    for seed in SEEDS:
        on=complete[seed+'_on_24'];off=complete[seed+'_off_24']
        fm,om,bm=[x['native']['mR'] for x in (on,off,base)]
        fp,op,bp=[x['detail']['both'] for x in (on,off,base)]
        decisions[seed]=dict(lower_eos=on['native']['eos']<off['native']['eos'],keeps_history=fm>=bm and fp>=bp,
                             improves_matched=fm>=om and fp>=op and (fm>om or fp>op))
    decisions['promote']=all(v['keeps_history'] and v['improves_matched'] for v in decisions.values())
    decisions['consistent_lower_eos']=all(decisions[s]['lower_eos'] for s in SEEDS)
    dump(output/'decision.json',decisions)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-only',action='store_true')
    summarize(parser.parse_args().baseline_only)

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
CHECKS = (8,16)
ARMS = ('single','separate','joint')


def scalar_ci(values):
    return ci(np.asarray(values,dtype=np.float64))


def study_states():
    return ['base_00']+[f'{seed}_{arm}_{step:02d}' for seed in SEEDS for arm in ARMS for step in CHECKS]


def e1_group(row):
    if row['cohort']=='previous' and row['dimension']=='quantity' and row['template'] in ('A','B'):
        return 'quantity_'+row['template']
    if row['cohort']=='extension':
        return 'relation' if row['template']=='shadow_direction' else row['dimension']
    return ''


def details(scores, rows, meta, texts, state, part, words):
    index={(r['image'],r['view']):i for i,r in enumerate(rows)}
    result=[]
    for r in meta:
        margins={}
        for word in words:
            pos,neg=[texts.index(x) for x in r[word]]
            for view,sign in [('upright',1),('rotated',-1)]:
                ix=index[r['image'],view]
                margins[word,view]=float(sign*(scores[ix,pos]-scores[ix,neg]))
        result.append(dict(state=state,part=part,image=r['image'],
               **{w+'_'+v:z for (w,v),z in margins.items()},
               **{'both_'+w:int(min(margins[w,'upright'],margins[w,'rotated'])>0) for w in words},
               upright=np.mean([margins[w,'upright']>0 for w in words]),
               rotated=np.mean([margins[w,'rotated']>0 for w in words]),
               weakest=np.mean([min(margins[w,'upright'],margins[w,'rotated']) for w in words]),
               both=np.mean([margins[w,'upright']>0 and margins[w,'rotated']>0 for w in words]),
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
        state_binding=read(out/'complete.json')
        assert set(state_binding['parts'])=={'native','generic','detail','trainpanel'}
        assert sha(out/'train_text.pt')==state_binding['train_text']
        train_text=torch.load(out/'train_text.pt',map_location='cpu',weights_only=True)
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
            if part in ('trainpanel','detail'):
                train=part=='trainpanel'
                words=('A','B') if train else ('C','D')
                features=train_text['features'] if train else text['relation']
                strings=train_text['phrases'] if train else text['texts']
                cases=read(root/'inputs/train.json') if train else details_meta
                assert [(r['image'],v) for r in cases for v in ('upright','rotated')]==[(r['image'],r['view']) for r in rs]
                rows=details(fixed(vs,features),rs,cases,strings,state,part,words)
                relation.extend(rows)
                metrics={k:float(np.mean([v[k] for v in rows])) for k in ('pairacc','margin','weakest','upright','rotated')}
                summary[part].update(**metrics,both=int(round(2*sum(v['both'] for v in rows))),out_of=2*len(cases),
                                     both_by_word={w:int(sum(v['both_'+w] for v in rows)) for w in words})
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
        elif part in ('detail','trainpanel'):seq=[(r['image'],r[metric]) for r in relation if r['state']==state and r['part']==part]
        else:seq=[(r['image'],r[metric]) for r in trajectory if r['state']==state and r['part']==part]
        seq.sort()
        return [r[0] for r in seq],np.asarray([r[1] for r in seq],dtype=np.float64)
    pairs={}
    comparisons=[]
    for step in CHECKS:
        for left,right in [('joint','separate'),('joint','single'),('separate','single'),('joint','base'),('separate','base'),('single','base')]:
            items=[]
            for seed in SEEDS:
                pair=(f'{seed}_{left}_{step:02d}',f'{seed}_{right}_{step:02d}' if right!='base' else 'base_00')
                comparisons.append((f'{seed}_{left}-{right}_{step}',[pair]))
                items.append(pair)
            comparisons.append((f'mean_{left}-{right}_{step}',items))
    for name,items in comparisons:
        result={}
        for part,m in [('native','mR'),('generic','mR'),('native','eos'),('native','format'),('native','tool'),('detail','both'),('detail','pairacc'),('detail','weakest'),('detail','margin'),('trainpanel','both'),('trainpanel','weakest')]:
            ds=[]
            for a,b in items:
                ai,x=aligned(a,part,m);bi,y=aligned(b,part,m);assert ai==bi and len(set(ai))==len(ai)
                ds.append(x-y)
            result[part+'_'+m]=scalar_ci(np.mean(ds,axis=0))
        pairs[name]=result
    dump(output/'paired_contrasts.json',pairs)
    base=complete['base_00'];decisions={}
    for seed in SEEDS:
        candidate=complete[seed+'_joint_16'];control=complete[seed+'_separate_16']
        fm,om,bm=[x['native']['mR'] for x in (candidate,control,base)]
        fp,op,bp=[x['detail']['both'] for x in (candidate,control,base)]
        decisions[seed]=dict(retain_historical_mR=fm>=bm,retain_historical_direction=fp>=bp,
                             improve_matched_direction=fp>op,retain_matched_mR=fm>=om)
    decisions['promote']=all(all(decisions[seed].values()) for seed in SEEDS)
    dump(output/'decision.json',decisions)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-only',action='store_true')
    summarize(parser.parse_args().baseline_only)

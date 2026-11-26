"""Cross-fit a two-outcome candidate reward, then evaluate on disjoint archived panels."""
import collections
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import torch

from 候选复验 import ROOT, PRIOR_RUN, OLD, RFPO, FORM, PRIOR, best, previous_banks, previous_panel, read, rr, summarize
from panels import bank as load_bank
sys.path.insert(0, str(ROOT.parent/'E3-2_UCM有效图组定向短训_20261124/code'))
from verify_screen import unpack

ARMS = ('ordinary','wording','rank_only','calibrated')
FEATURES = ('A margin','B margin','smaller margin','A correct','B correct','both correct',
            'mean RR','lowest RR','mean positive similarity','lowest positive similarity',
            'mean positive minus top negative','lowest positive minus top negative','mean positive minus top eight negatives')


def external(folder, skip):
    rows = read(folder/'inputs/panel.json')
    by_pair = collections.defaultdict(list)
    for row in rows:
        by_pair[row['pair']].append(row)
    rows = [r for group in by_pair.values() if all(r['rgb_sha256'] not in skip for r in group) for r in group]
    bank = load_bank(folder)
    for seed in 'ab':
        batches = [unpack(folder/'records', tag) for tag in (seed,'extra_'+seed)]
        for r in rows:
            first, second = [batch[r['image']] for batch in batches]
            assert first[0]['rgb'] == second[0]['rgb'] == r['rgb_sha256']
            assert first[0]['adapter'] == second[0]['adapter']
            assert first[0]['query'] == second[0]['query'] == r['scene_query']
            visual = torch.cat([z[1]['visual'] for z in (first,second)]).double().numpy()
            assert visual.shape == (16,768)
            yield r, seed, 'scene', bank, visual


def packets(stream):
    for row, seed, view, bank, visual in stream:
        vis = visual[:16]
        ix = bank['maps'][row['image']]
        pos = ix['native']
        alts = sorted(set(pos)-{pos[0]})
        if len(alts)<2:
            continue
        ids=np.array([ix[k] for k in 'ABCDE'])
        # If a literal scoring phrase recurs in the evaluation pair this isn't a new wording.
        assert set(ids[:3].ravel()).isdisjoint(ids[3:].ravel())
        hard_scores = vis @ bank['hard'][ids.ravel()].T
        h = hard_scores[:,::2] - hard_scores[:,1::2]
        native = vis @ bank['native'].T
        odd = ~bank['even'].copy()
        even = bank['even'].copy()
        odd[pos] = False
        even[pos] = False
        alt_scores = native[:,alts]
        fit = np.array([[rr(v[j],v[odd]) for j in alts] for v in native],dtype=object)
        check = np.array([[float(rr(v[j],v[even])) for j in alts] for v in native])
        ranks = np.array([[1 + np.sum(v[odd]>v[j]) + .5*np.sum(v[odd]==v[j]) for j in alts] for v in native])
        top = np.partition(native[:,odd],-8,axis=1)[:,-8:]
        for held in range(len(alts)):
            cols = [j for j in range(len(alts)) if j!=held]
            ordinary = fit[:,cols].mean(axis=1)
            visible = alt_scores[:,cols]
            mean_pos, low_pos = visible.mean(1),visible.min(1)
            features = np.column_stack((h[:,0],h[:,1],h[:,:2].min(1),h[:,0]>0,h[:,1]>0,
                                        (h[:,:2]>0).all(1),ordinary,fit[:,cols].min(1),
                                        mean_pos,low_pos,mean_pos-top.max(1),low_pos-top.max(1),mean_pos-top.mean(1))).astype(float)
            for block,start in enumerate((0,8)):
                sl = slice(start,start+8)
                x = features[sl]
                y = np.column_stack((h[sl,2],check[sl,held]))
                baseline = best(ordinary[sl],range(8))
                phrase = h[sl,:2]
                diffs = phrase[:,None,:]-phrase[None,:,:]
                pct = ((diffs>0)+0.5*(diffs==0)).mean(1)
                gain = (pct-pct[baseline].mean(0)).min(1)
                mids = ranks[sl][:,cols]
                ok = ((phrase>0).all(1) & (phrase>phrase[baseline].mean(0)).all(1)
                      & (mids<=mids[baseline].mean(0)+1).all(1))
                wording = best(list(zip(gain,ordinary[sl])),np.flatnonzero(ok)) or baseline
                yield dict(image=row['image'], pair=row['pair'], rgb=row['rgb_sha256'],seed=seed,view=view,
                           held=held,block=block,weight=1/len(alts),x=x-x.mean(axis=0),y=y-y.mean(axis=0),
                           fine=(h[sl,3:]>0).mean(1),mrr=check[sl,held],ordinary=ordinary[sl],
                           baseline=baseline,wording=wording)


def fit_model(data):
    x,y=[np.concatenate([r[k] for r in data]) for k in ('x','y')]
    weights=np.repeat([r['weight'] for r in data],8)
    weights/=weights.sum()
    scale_x,scale_y=[np.sqrt(weights@(v*v)) for v in (x,y)]
    scale_x[scale_x<1e-12]=1.
    scale_y[scale_y<1e-12]=1.
    x=x/scale_x
    y=y/scale_y
    weighted=x.T*weights
    coef=np.linalg.solve(weighted@x+.1*np.eye(x.shape[1]),weighted@y)
    return dict(features=FEATURES, scale_x=scale_x.tolist(),scale_y=scale_y.tolist(),coef=coef.tolist())


def evaluate(data, model):
    weights=np.array(model['coef'])/np.array(model['scale_x'])[:,None]
    for r in data:
        x=r['x'];ref=r['baseline']
        # Two predicted gains are in units of the corresponding training-outcome SD.
        pred=x@weights
        gains=pred-pred[ref].mean(0)
        eligible=np.flatnonzero((gains>1e-12).all(1))
        joint=best(list(zip(gains.min(1),r['ordinary'])),eligible) or ref
        rank=best(list(zip(np.round(pred[:,1],12),r['ordinary'])),range(8))
        top={'ordinary':ref,'wording':r['wording'],'rank_only':rank,'calibrated':joint}
        fine, ordinary = r['fine'],r['mrr']
        good=(fine>fine[ref].mean()+1e-12)&(ordinary>ordinary[ref].mean()+1e-12)
        outcomes={}
        for arm,ws in top.items():
            df=fine[ws,None]-fine[ref]; dr=ordinary[ws,None]-ordinary[ref]
            changed=ws!=ref
            win=float(((df>0)&(dr>0)).mean()) if changed else 0.
            loss=float(((df<0)&(dr<0)).mean()) if changed else 0.
            outcomes[arm]=dict(top=ws,detail=float(fine[ws].mean()),mrr=float(ordinary[ws].mean()),net=win-loss,
                               win=win,loss=loss,changed=changed,oracle_hit=float(good[ws].mean()))
        yield ({k:r[k] for k in ('image','pair','rgb','seed','view','held','block')}
               |dict(support=0,oracle=bool(good.any()),outcomes=outcomes))


def report(rows, tag):
    with gzip.open(ROOT/('校准_'+tag+'.jsonl.gz'),'wt') as f:
        for r in rows:
            f.write(json.dumps(r)+'\n')
    groups=collections.defaultdict(list)
    for r in rows:
        groups[r['view']].append(r)
    result={k:summarize(v,ARMS,('ordinary','rank_only','wording')) for k,v in groups.items()}
    for view,block in result.items():
        del block['support_images']
        block['selected_opportunities']={arm:sum(r['outcomes'][arm]['oracle_hit'] for r in groups[view]) for arm in ARMS}
    return result


def main():
    torch.set_num_threads(1)
    base=read(OLD/'inputs/real_roster.json')
    hold_keys={r['rgb_sha256'] for r in read(PRIOR_RUN/'inputs/holdout.json')}
    dev=[r for r in base if r['rgb_sha256'] not in hold_keys]
    held=[r for r in base if r['rgb_sha256'] in hold_keys]
    previous=previous_banks()
    training=list(packets(previous_panel(dev,previous)))
    names=sorted({r['pair'] for r in training})
    np.random.default_rng(261126).shuffle(names)
    assignment={pair:i%5 for i,pair in enumerate(names)}
    cv=[]
    for fold in range(5):
        model=fit_model([r for r in training if assignment[r['pair']]!=fold])
        cv.extend(evaluate([r for r in training if assignment[r['pair']]==fold],model))
    out={'dev_cv':report(cv,'开发交叉预测')}
    model=fit_model(training)
    (ROOT/'锁定评分参数.json').write_text(json.dumps(dict(model=model,folds=assignment),indent=2)+'\n')
    out['hold_first']=report(list(evaluate((r for r in packets(previous_panel(held,previous)) if r['view']=='first'),model)),'78首句')
    skip={r['rgb_sha256'] for r in base+read(PRIOR/'reward_confirm/inputs/panel.json')+read(RFPO/'inputs/panel.json')+read(FORM/'inputs/ucm.json')}
    merged=[]
    for folder,total in ((ROOT.parent/'E3_UCM多数配文新图复核_20261125',148),(PRIOR/'guard',122)):
        panel=read(folder/'inputs/panel.json')
        assert len(panel)==total and not skip.intersection(r['rgb_sha256'] for r in panel)
        rows=list(evaluate(packets(external(folder,skip)), model))
        out[str(total)]=report(rows,str(total))
        merged.extend(rows)
        skip.update(r['rgb_sha256'] for r in panel)
    assert not {r['pair'] for r in merged}.intersection(assignment)
    assert len({r['pair'] for r in merged})==out['148']['scene']['pairs']+out['122']['scene']['pairs']
    out['pooled']=summarize(merged,ARMS,('ordinary','rank_only','wording'))
    del out['pooled']['support_images']
    def passed(stats):
        d=stats['against']['ordinary']['calibrated']
        return min(d['detail']['seeds'])>0 and all(d[k]['mean']>=0 for k in ('mrr','net'))
    out['train']=all(passed(stat) for stat in [out['hold_first']['first'],out['148']['scene'],out['122']['scene']]) and out['pooled']['against']['ordinary']['calibrated']['detail']['interval'][0]>0 and out['pooled']['against']['rank_only']['calibrated']['detail']['mean']>0
    (ROOT/'校准结果.json').write_text(json.dumps(out,indent=2)+'\n')
    print('train',out['train'])


if __name__=='__main__':
    main()

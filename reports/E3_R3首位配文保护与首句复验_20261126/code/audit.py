"""Verify paired endpoint changes, protection coverage, and the source of rank changes."""
import gzip
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

STAGE=Path(__file__).resolve().parents[1]
ARMS=('calibrated','guard','margin_guard','rank_priority','rank_only')


def read_rows(tag):
    with gzip.open(STAGE/'results'/('priority_'+tag+'.jsonl.gz'),'rt') as f:
        return [json.loads(line) for line in f]


def audit(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r['image'],r['seed'],r['pair'])].append(r)
    out={'images':len({r['rgb'] for r in rows}), 'pairs':len({r['pair'] for r in rows}), 'folds':len(rows)}
    images=sorted({r['image'] for r in rows})
    def paired_mean(fn):
        v=np.array([[np.mean([fn(r) for r in g]) for (n,s,p),g in sorted(groups.items()) if s==seed] for seed in 'ab'])
        assert v.shape==(2,len(images))
        return float(v.mean())
    out['visible_first_when_actual_first']=paired_mean(lambda r:r['ceiling'] and r['protected']>0)/paired_mean(lambda r:r['ceiling'])
    out['arms']={}
    for arm in ARMS:
        def level(r,k):return r['outcomes'][arm][k]
        ceiling=paired_mean(lambda r:r['ceiling'])
        old_lost=paired_mean(lambda r:r['outcomes']['calibrated']['drop_first'])
        catch=paired_mean(lambda r:r['outcomes']['calibrated']['drop_first'] if level(r,'intervened') else 0)
        delta=paired_mean(lambda r:level(r,'mrr')-r['outcomes']['ordinary']['mrr'])
        cost=paired_mean(lambda r:level(r,'ceiling_cost'))
        recovered=paired_mean(lambda r:level(r,'mrr')-r['outcomes']['calibrated']['mrr'] if r['ceiling'] else 0.)
        rest=paired_mean(lambda r:level(r,'mrr')-r['outcomes']['calibrated']['mrr'] if not r['ceiling'] else 0.)
        out['arms'][arm]=dict(delta=delta,first_cost=cost,other_contribution=delta-cost,
               drop_rate_if_original_first=paired_mean(lambda r:level(r,'drop_first'))/ceiling,
               share_of_old_drops_touched=catch/old_lost if old_lost else None,
               mrr_recovered_on_first=recovered,mrr_change_elsewhere=rest,
               changes_from_old=sum(level(r,'intervened') for r in rows),
               changes_from_guard=sum(level(r,'top')!=r['outcomes']['guard']['top'] for r in rows))
        assert abs((recovered+rest)-paired_mean(lambda r:level(r,'mrr')-r['outcomes']['calibrated']['mrr']))<1e-12
    # Bootstrap whole natural pairs, carrying both seeds and all caption folds together.
    by_pair=defaultdict(list)
    for (image,seed,pair),entries in groups.items():
        by_pair[pair].append(np.mean([[float(r['ceiling']),
                 *(r['outcomes'][arm]['drop_first'] for arm in ARMS)] for r in entries],axis=0))
    packed=np.array([np.sum(block,axis=0) for pair,block in sorted(by_pair.items())])
    rng=np.random.default_rng(261126)
    weights=rng.multinomial(len(packed),np.ones(len(packed))/len(packed),size=20000)
    totals=weights@packed
    assert (totals[:,0]>0).all()
    drops=totals[:,1:]/totals[:,0,None]
    for col,arm in enumerate(ARMS):
        out['arms'][arm]['drop_rate_interval']=np.quantile(drops[:,col],[.025,.975]).tolist()
        out['arms'][arm]['drop_rate_delta_from_old_interval']=np.quantile(drops[:,col]-drops[:,0],[.025,.975]).tolist()
        out['arms'][arm]['images_with_old_drop']=len({r['rgb'] for r in rows if r['outcomes']['calibrated']['drop_first']>0})
        out['arms'][arm]['images_with_drop']=len({r['rgb'] for r in rows if r['outcomes'][arm]['drop_first']>0})
    return out


def example(rows,roster,bank):
    # Hash order fixes one interpretable rescue if present.
    import hashlib
    choices=[r for r in rows if r['ceiling'] and r['outcomes']['calibrated']['mrr']<1
             and r['outcomes']['guard']['mrr']==1 and r['outcomes']['guard']['detail']>=r['outcomes']['calibrated']['detail']]
    if not choices:return None
    def identifier(r):return '|'.join(str(r[k]) for k in ('image','seed','block','held'))
    row=min(choices,key=lambda r:hashlib.sha256(identifier(r).encode()).hexdigest())
    meta=next(r for r in roster if r['image']==row['image'])
    ids=bank['maps'][row['image']]['native']
    caption_id=sorted(set(ids)-{ids[0]})[row['held']]
    return dict(image=row['image'],seed=row['seed'],block=row['block'],input=meta['orig_query'],
                held=meta['captions'][ids.index(caption_id)],old=row['outcomes']['calibrated'],guard=row['outcomes']['guard'])


def main():
    from guard import PRIOR,load_bank,read
    tags=['first78','scene148','scene122','first98']
    rows={k:read_rows(k) for k in tags}
    results={k:audit(v) for k,v in rows.items()}
    from guard import summarize
    first=rows['first78']+rows['first98']
    assert not {r['rgb'] for r in rows['first78']} & {r['rgb'] for r in rows['first98']}
    assert len({r['pair'] for r in first})==88
    arms=('ordinary',)+ARMS
    combined=summarize(first,arms,('ordinary','calibrated'))
    del combined['support_images']
    (STAGE/'results/两批首句事后合并.json').write_text(json.dumps(combined,indent=2)+'\n')
    results['first_combined']=audit(first)
    results['scene_combined']=audit(rows['scene148']+rows['scene122'])
    for tag in tags:
        ref=read(STAGE/'results'/('priority_'+tag+'.json'))
        for arm in ARMS:
            assert abs(results[tag]['arms'][arm]['delta']-ref['against']['ordinary'][arm]['mrr']['mean'])<1e-12
    results['example']=example(rows['first98'],read(STAGE/'inputs/real_roster.json'),load_bank(PRIOR/'reward_confirm'))
    (STAGE/'results/核对与归因.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

STAGE = Path(__file__).resolve().parents[1]
LAST = STAGE.parent/'E3_R3跨措辞与相似图对复验_20261126'
sys.path.insert(0,str(LAST))
from 校准复验 import external, load_bank, packets
from 候选复验 import PRIOR, OLD, PRIOR_RUN, best, previous_banks, previous_panel, read, rr, summarize, sha

ARMS=('ordinary','calibrated','guard','margin_guard')
METRICS=('detail','mrr','net')


def with_gaps(stream):
    for row,seed,view,bank,visual in stream:
        ix=bank['maps'][row['image']]['native']
        alts=sorted(set(ix)-{ix[0]})
        if len(alts)<2:
            continue
        neg=~bank['even'].copy()
        neg[ix]=False
        sims=visual[:16]@bank['native'].T
        gaps=sims[:,alts]-sims[:,neg].max(1,keepdims=True)
        for packet in packets([(row,seed,view,bank,visual)]):
            start=8*packet['block']
            cols=[c for c in range(len(alts)) if c!=packet['held']]
            packet['gaps']=gaps[start:start+8,cols]
            yield packet


def choose(packet, weights):
    ref=packet['baseline'];gaps=packet['gaps']
    keep=(gaps[ref]>0).all(0)
    visible=gaps[:,keep]
    # A positive gap means there are zero tied or higher negatives.
    safe=(visible>0).all(1)
    buffer=(visible>=visible[ref].mean(0)-1e-12).all(1)
    pred=packet['x']@weights
    gains=pred-pred[ref].mean(0)
    eligible=(gains>1e-12).all(1)
    keys=list(zip(gains.min(1),packet['ordinary']))
    def select(mask):
        return best(keys,np.flatnonzero(mask)) or ref
    tops=dict(ordinary=ref,calibrated=select(eligible),guard=select(eligible&safe),
              margin_guard=select(eligible&safe&buffer))
    for arm in ('guard','margin_guard'):
        if tops[arm]!=ref:
            assert safe[tops[arm]].all()
    return tops,int(keep.sum())


def measure(packet,selected,visible_first):
    ref=selected['ordinary'];mrr=packet['mrr'];fine=packet['fine']
    old=selected['calibrated']
    ceiling=bool(np.all(mrr[ref]==1))
    out={}
    for arm,ws in selected.items():
        df=fine[ws,None]-fine[ref];dr=mrr[ws,None]-mrr[ref]
        changed=ws!=ref
        win=float(((df>0)&(dr>0)).mean()) if changed else 0.
        loss=float(((df<0)&(dr<0)).mean()) if changed else 0.
        out[arm]=dict(top=ws,detail=float(fine[ws].mean()),mrr=float(mrr[ws].mean()),
                      net=win-loss,win=win,loss=loss,changed=changed,intervened=ws!=old,
                      drop_first=float((mrr[ws]<1).mean()) if ceiling else 0.,
                      ceiling_cost=float(mrr[ws].mean()-1) if ceiling else 0.)
    return {k:packet[k] for k in ('image','rgb','pair','seed','view','block','held')} | dict(
        outcomes=out,protected=visible_first,ceiling=ceiling,support=0,
        oracle=bool(((mrr>mrr[ref].mean()+1e-12)&(fine>fine[ref].mean()+1e-12)).any()))


def detailed(records):
    summary=summarize(records,ARMS,('ordinary','calibrated'))
    del summary['support_images']
    groups=defaultdict(list)
    for r in records:
        groups[r['seed'],r['rgb']].append(r)
    def mean(fn):
        return float(np.mean([np.mean([fn(r) for r in group]) for group in groups.values()]))
    ceiling=mean(lambda r:r['ceiling'])
    summary.update(protected_folds=sum(r['protected']>0 for r in records),
                   ceiling_folds=sum(r['ceiling'] for r in records),
                   first_also_protected=sum(r['ceiling'] and r['protected']>0 for r in records))
    for arm in ARMS:
        def value(r,key):return r['outcomes'][arm][key]
        changed=[r for r in records if value(r,'intervened')]
        summary['totals'][arm].update(interventions=len(changed),intervention_images=len({r['rgb'] for r in changed}),
              drop_rate_given_first=mean(lambda r:value(r,'drop_first'))/ceiling if ceiling else None,
              mrr_contribution_when_first=mean(lambda r:value(r,'ceiling_cost')))
    return summary


def save(tag,records):
    with gzip.open(STAGE/'results'/(tag+'.jsonl.gz'),'wt') as f:
        for row in records:f.write(json.dumps(row)+'\n')
    summary=detailed(records)
    (STAGE/'results'/(tag+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def real_stream():
    rows=read(STAGE/'inputs/real_roster.json')
    bank=load_bank(PRIOR/'reward_confirm')
    locked=read(STAGE/'inputs/real_lock.json')
    assert sha(STAGE/'inputs/real_roster.json')==locked['roster']
    historical=read(PRIOR/'reward_confirm/records/provenance.json')
    for seed in 'ab':
        base=STAGE/'records/real'/seed
        raw_manifest=read(base/'raw/complete.json')
        assert raw_manifest['adapter']==historical['adapters'][seed]
        features=base/'features'
        manifest={k:v for i in range(2) for k,v in read(features/('complete_'+str(i)+'.json'))['files'].items()}
        assert set(manifest)=={r['image'] for r in rows}==set(raw_manifest['files'])
        for row in rows:
            name=row['image'];path=features/(name+'.pt')
            rawpath=base/'raw'/(name+'.pt')
            assert sha(rawpath)==raw_manifest['files'][name] and sha(path)==manifest[name]
            raw=torch.load(rawpath,map_location='cpu',weights_only=True)
            feat=torch.load(path,map_location='cpu',weights_only=True)
            assert row['rgb_sha256']==raw['rgb']==feat['rgb']
            assert raw['query']==row['orig_query']==row['captions'][0] and raw['feedback']==row['feedback']
            assert raw['rngs']==[row['rng_'+seed]+100*j for j in (0,1)]
            assert feat['adapter']==raw_manifest['adapter']==raw['adapter'] and feat['source']==raw_manifest['files'][name]
            assert feat['visual'].shape==(17,768) and len(raw['paths'])==16
            yield row,seed,'first',bank,feat['visual'].double().numpy()


def run(tag):
    (STAGE/'results').mkdir(exist_ok=True)
    assert sha(LAST/'锁定评分参数.json')==read(STAGE/'inputs/real_lock.json')['calibration']
    model=read(LAST/'锁定评分参数.json')['model']
    weights=np.array(model['coef'])/np.array(model['scale_x'])[:,None]
    if tag=='first98':
        stream=real_stream()
    elif tag=='first78':
        hold={r['rgb_sha256'] for r in read(PRIOR_RUN/'inputs/holdout.json')}
        rows=[r for r in read(OLD/'inputs/real_roster.json') if r['rgb_sha256'] in hold]
        stream=(v for v in previous_panel(rows,previous_banks()) if v[2]=='first')
    else:
        folder=STAGE.parent/'E3_UCM多数配文新图复核_20261125' if tag=='scene148' else PRIOR/'guard'
        stream=external(folder,set())
    records=[measure(p,*choose(p,weights)) for p in with_gaps(stream)]
    summary=save(tag,records)
    if tag!='first98':
        alias={'first78':('hold_first','first'),'scene148':('148','scene'),'scene122':('122','scene')}[tag]
        old=read(LAST/'校准结果.json')[alias[0]][alias[1]]
        assert summary['folds']==old['folds'] and summary['images']==old['images']
        for arm in ('ordinary','calibrated'):
            for k in METRICS:
                assert abs(summary['levels'][arm][k]-old['levels'][arm][k])<1e-12
    print(tag,summary['images'],summary['totals']['guard']['interventions'],flush=True)


def merged():
    rows=[]
    for tag in ('scene148','scene122'):
        with gzip.open(STAGE/'results'/(tag+'.jsonl.gz'),'rt') as f:rows.extend(json.loads(s) for s in f)
    pooled=detailed(rows)
    (STAGE/'results/pooled_scene.json').write_text(json.dumps(pooled,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(1)
    for tag in sys.argv[1:]:
        if tag=='pool':merged()
        else:run(tag)

"""Joint paired endpoint and post-tool sampling contrasts across the two seed-matched runs."""
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from combine import aligned
from metrics import ci
ROOT=Path(__file__).resolve().parents[1]
ARMS=['R0','R1','R2','first_R2','old_R2']
SEEDS=['.','repeat']
def read(path):return json.loads(path.read_bytes())
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def merged(ds):
    z=ci(np.mean(ds,axis=0))
    z['by_seed']=[float(v.mean()) for v in ds]
    return z

def validation():
    values=defaultdict(lambda:defaultdict(list))
    for sub in SEEDS:
        for file,g,k in ([('retrieval_by_image.csv','',m) for m in ('mR','I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10')]+
                        [('detail_by_case.csv',g,'correct') for g in ('quantity_A','quantity_B','attribute','relation')]):
            v={a:aligned(ROOT/sub,file,g,k,a) for a in ARMS}
            for a,b in [('R2','R0'),('R2','R1'),('R2','first_R2'),('R2','old_R2'),('first_R2','old_R2'),('R1','R0')]:
                assert v[a].keys()==v[b].keys()
                values[a+'-'+b][g or k].append(100*np.array([v[a][i]-v[b][i] for i in sorted(v[a])]))
    write(ROOT/'results/paired_validation.json',{pair:{m:merged(d) for m,d in v.items()} for pair,v in values.items()})

def panel():
    values=defaultdict(lambda:defaultdict(list));levels={}
    for sub in SEEDS:
        pooled={}
        for arm in ARMS:
            f=ROOT/sub/'panel'/(arm+'.csv')
            info=read(f.with_suffix('.json'))
            assert hashlib.sha256(f.read_bytes()).hexdigest()==info['sha'] and info['n']==192
            with f.open(encoding='utf-8-sig') as fp:
                rows=list(csv.DictReader(fp))
            assert len({(x['image'],x['slot']) for x in rows})==192
            g=defaultdict(list)
            for x in rows:g[x['pair']].append(x)
            assert len(g)==12 and all(len(x)==16 for x in g.values())
            stats={m:np.array([np.mean([float(x[m]) for x in g[key]]) for key in sorted(g)]) for m in ('margin','mrr')}
            pooled[arm]=stats
            levels.setdefault(sub,{})[arm]=dict(**{m:float(v.mean()) for m,v in stats.items()},
                immediate_end=sum(x['second_eos_only']=='True' for x in rows),
                third_round=sum(x['third_round']=='True' for x in rows))
        for a,b in [('R2','R0'),('R2','R1'),('R2','first_R2'),('R2','old_R2')]:
            for m in ('margin','mrr'):
                values[a+'-'+b][m].append(pooled[a][m]-pooled[b][m])
    write(ROOT/'results/paired_panel.json',dict(levels=levels,comparisons={k:{m:merged(d) for m,d in v.items()} for k,v in values.items()}))

def training_audit():
    result={}
    for sub in SEEDS:
        loc=ROOT/sub/'records'
        x,y=[read(loc/a/'steps.json') for a in ('R1','R2')]
        assert x[0]['trajectories']==y[0]['trajectories'] and x[0]['proxy_rank']==y[0]['proxy_rank']
        for arm in ('R1','R2','first_R2'):
            steps=read(loc/arm/'steps.json')
            assert len(steps)==24 and [z['image'] for z in steps]==[z['image'] for z in x]
            for z in steps:
                raw=1/(1+np.asarray(z['proxy_rank']))
                scores=np.array(z['retrieval'])
                if arm=='R1':assert np.allclose(raw,scores)
                else:
                    mar=np.array(z['native_margin']);o=np.argsort(mar,kind='stable')
                    for m in np.unique(mar):
                        ix=mar[o]==m
                        assert np.allclose(scores[o[ix]],np.sort(raw)[ix].mean())
                if arm!='first_R2':assert np.ptp(z['format'])==0
                total=scores+np.array(z['format'])
                assert np.allclose((total-total.mean())/(total.std(ddof=1)+1e-8),z['advantage'])
            result.setdefault(sub,{})[arm]=dict(groups=len(steps),nonzero=sum(z['norm']>0 for z in steps),
                immediate_end=sum(t['ended'] and len(t['ids'])==1 for z in steps for t in z['trajectories']),
                varied=sum(np.ptp(z['retrieval'])>1e-12 for z in steps).item(),
                replay_error=float(np.mean([z['replay_mean_abs_logp_difference'] for z in steps])))
    write(ROOT/'results/training_checks.json',result)
if __name__=='__main__':
    validation();panel();training_audit()

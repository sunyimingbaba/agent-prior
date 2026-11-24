"""Freeze one global scale using only the two reward components on the discovery panel."""
import csv,json,sys,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from analyze import write_csv,dump,boot
from learning_signal import reward,advantage
def groups(path):
    blocks=defaultdict(list)
    with (path/'candidates.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):blocks[r['arm'],r['image']].append(r)
    return blocks
def col(rr,k):return np.array([float(r[k]) for r in rr])
def lock(root):
    ratio=[]
    for rr in groups(root/'results').values():
        n,a=[np.ptp(col(rr,k)) for k in ('native_margin','A')]
        if min(n,a)>1e-8:ratio.append(n/a)
    dump(root/'inputs/continuous_lock.json',dict(kappa=float(np.median(ratio)),groups=len(ratio),
         protocol_sha=hashlib.sha256((root/'连续分差复核口径.md').read_bytes()).hexdigest()))
def run(root,dest):
    spec=json.loads((root/'inputs/continuous_lock.json').read_text());k=spec['kappa']
    assert spec['protocol_sha']==hashlib.sha256((root/'连续分差复核口径.md').read_bytes()).hexdigest()
    rows=[];counts=defaultdict(lambda:defaultdict(int));delta=defaultdict(list)
    for (arm,name),rr in groups(dest).items():
        r=rr[0];m,h,old=[col(rr,key) for key in ('native_margin','A','proxy_rank')]
        raw=reward(m,old);new=reward(m+k*h,old);act=advantage(new)
        i=np.flatnonzero(raw==max(raw));j=np.flatnonzero(new==max(new))
        counts[arm]['flat']+=int(not np.any(act));counts[arm]['changed']+=int(set(i)!=set(j));counts[arm]['active_to_flat']+=int(np.any(advantage(raw)) and not np.any(act))
        record=dict(arm=arm,image=name,pair=r['pair'],scene=r['scene'],slots=','.join(map(str,j)))
        for key in ['B','C']+(['D'] if 'D' in r else [])+['MRR']:
            v=col(rr,key);v=v if key=='MRR' else v>0
            record[key]=float(v[j].mean());record['diff_'+key]=float(v[j].mean()-v[i].mean())
        if 'D' in r:record['diff_BCD']=float(np.mean([record['diff_'+p] for p in 'BCD']))
        rows.append(record)
    levels={};byarm={a:[r for r in rows if r['arm']==a] for a in ('a','b')}
    for arm,rr in byarm.items():levels[arm]={k:float(np.mean([r[k] for r in rr])) for k in ('B','C','D','MRR') if k in rr[0]}
    diffs={}
    for metric in [k for k in rows[0] if k.startswith('diff_')]:
        assert [r['image'] for r in byarm['a']]==[r['image'] for r in byarm['b']]
        x=np.array([[r[metric] for r in rr] for rr in byarm.values()]);p=[r['pair'] for r in byarm['a']]
        diffs[metric[5:]]=dict(pairs=boot(x.mean(0),p),seeds=x.mean(1).tolist(),scenes=boot(x.mean(0),[r['scene'] for r in byarm['a']]))
    dump(dest/'continuous.json',dict(kappa=k,levels=levels,differences=diffs,groups={k:dict(v) for k,v in counts.items()}))
    write_csv(dest/'continuous_choices.csv',rows)
if __name__=='__main__':
    root=Path(sys.argv[1])
    if sys.argv[2]=='lock':lock(root)
    else:run(root,Path(sys.argv[2]))

from pathlib import Path
import shutil
import torch
from local import ROOT,read,load,sha,dump,feature_path
def main():
    dest=ROOT/'reachable';before=read(ROOT/'inputs/lock.json');plans=read(ROOT/'inputs/train.json')
    trace=read(ROOT/'inputs/teacher_trace.json');hs={}
    for name in ('train.json','holdout.json','teacher_trace.json'):
        shutil.copy2(ROOT/'inputs'/name,dest/'inputs'/name)
    for s in 'ab':
        ts=load(ROOT/'inputs'/('targets_'+s+'.pt'));rows={r['image']:r for r in plans[s]}
        for proof in (r for r in trace if r['seed']==s):
            n,view=proof['image'],proof['view'];f=feature_path(rows[n],s,view);assert sha(f)==proof['file']
            v=load(f)['visual'][proof['winner']].float();g=ts[n,view]
            g['ordinary_target']=g['ordinary_dir']@v;g['teacher']=g['hard_dir']@v
        f=dest/'inputs'/('targets_'+s+'.pt');torch.save(ts,f);hs[s]=sha(f)
    dump(dest/'inputs/lock.json',dict(**{k:v for k,v in before.items() if k not in ('protocol','teacher_files')},
        protocol=sha(dest/'实验口径.md'),teacher_files=hs,original_lock=sha(ROOT/'inputs/lock.json'),feasibility=sha(ROOT/'results/feasibility.json')))
if __name__=='__main__':main()

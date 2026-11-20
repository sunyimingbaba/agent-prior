"""Verify raw vectors, rollout routing and native archives; reproduce per-sample totals from CPU arrays."""
import csv,hashlib
from pathlib import Path
import numpy as np
import torch
from shared import ROOT,read,dump,sha
def main():
    known=read(ROOT/'inputs/retained.json');plan=read(ROOT/'inputs/protocol.json')
    for file,h in read(ROOT/'inputs/lock.json').items():assert sha(ROOT/file)==h,file
    acc={};greedy={};mismatch={}
    cases={r['image']:r for r in known}
    for split in ('train','val'):
        out=ROOT/split;buf=torch.load(out/'features.pt',map_location='cpu',weights_only=True)
        ix=read(out/'index.json')
        for file,h in read(out/'binding.json')['files'].items():assert sha(out/file)==h,file
        with (out/'candidates.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
        native=buf['visual'].double().numpy()@buf['text'].double().numpy().T
        for i,z in enumerate(rows):
            r=cases[z['image']];mode=z['mode'];k=int(z['k']);tag=split+'/'+mode
            stem=ROOT/'samples'/mode/Path(z['image']).stem/('greedy.pt' if k==-1 else f'sample{k}.pt')
            h=torch.load(stem,weights_only=True,map_location='cpu');rec=read(stem.with_suffix('.json'))
            assert rec['sha256']==sha(stem) and rec['vector_sha256']==z['vector']==hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()
            assert rec['query']==r['orig_query' if mode=='original' else 'scene_query']
            assert len(rec['events'])==rec['used']+rec['filtered'] and sum(len(q['ids']) for q in rec['rounds'])==int(z['tokens'])
            if mode=='original' and k==-1:
                greedy[split]=greedy.get(split,0)+int(rec['history']['bit_equal'])
                if not rec['history']['bit_equal']:mismatch.setdefault(split,[]).append(z['image'])
            for w in 'ABCD':
                p,n=[ix['texts'].index(s) for s in r[w]]
                expected=(native[i,p]-native[i,n])*(-1 if mode=='rotated' else 1)
                assert abs(expected-float(z[w]))<1e-12
            if mode!='rotated':
                assert abs(float(z['old'])-float(z['fmt'])-1/(1+int(z['rank0'])))<1e-12
            acc[tag]=acc.get(tag,0)+1
        key=[(z['image'],z['mode'],z['k']) for z in rows];assert len(key)==len(set(key))
    train_ids={r['image'] for r in known if r['split']=='train'}
    assert not train_ids&{r['image'] for r in known if r['split']=='val'}
    source_ids=len(read(ROOT/'inputs/selected.json'))
    dump(ROOT/'verification.json',dict(raw_paths_verified=acc,scored=sum(acc.values()),photos=len(known),screened_for_scoring=source_ids,
         greedy_matches=greedy,greedy_mismatch=mismatch,training_decision=read(ROOT/'train/decision.json')['pass_gate'],
         rank_and_margin_recomputed=True))
if __name__=='__main__':main()

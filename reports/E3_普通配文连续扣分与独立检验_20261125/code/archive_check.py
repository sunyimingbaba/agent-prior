"""Check the downloaded raw samples, greedy identities, feature sources and frozen checkpoints."""
from pathlib import Path
import tarfile,hashlib,io,sys
import torch
from compare import OUT,SOURCE,read,sha,dump,load
def archives():
    idx=read(OUT/'archives/index.json');evidence={};hist=read(SOURCE/'inputs/lock.json')
    for name,meta in idx.items():
        f=OUT/'archives'/name;assert sha(f)==meta['sha'];typ,tag,s=name.split('_')[:3];s=s.split('.')[0]
        if typ=='ordinary':rows=read(SOURCE/'inputs'/(tag+'.json'))
        else:rows=read(OUT/'inputs'/('covered_extra.json' if typ=='covered' else tag+'.json'))
        data={r['image']:r for r in rows}
        with tarfile.open(f) as tf:
            import json
            raw=tf.extractfile('complete.json').read();assert hashlib.sha256(raw).hexdigest()==meta['source_manifest']
            cp=json.loads(raw);assert cp['adapter']==meta['adapter'] and set(cp['files'])==set(data)
            slots=0
            for n,r in data.items():
                blob=tf.extractfile(n+'.pt').read();assert hashlib.sha256(blob).hexdigest()==cp['files'][n]
                z=torch.load(io.BytesIO(blob),weights_only=True);assert z['rgb']==r['rgb_sha256'] and z['image']==n and z['adapter']==cp['adapter']
                if typ=='ordinary':
                    assert z['query']==r['orig_query'] and z['prior'].shape==(3584,);slots+=1
                else:
                    assert z['query']==r['scene_query'] and z['hidden'].shape==(17,3584) and len(z['paths'])==16 and torch.isfinite(z['hidden']).all()
                    folder=OUT/'records/covered/extra'/s if typ=='covered' else OUT/'records'/tag/s
                    feat=load(folder/'features'/(n+'.pt'));assert feat['source']==cp['files'][n] and feat['adapter']==cp['adapter']
                    slots+=16
            evidence[name]=dict(images=len(rows),candidate_or_single_count=slots,adapter=cp['adapter'])
    return evidence
def main():
    a=archives();lock=read(OUT/'inputs/head_nomination.json');models={}
    for s in 'ab':
        for arm in ('ordinary','hard'):
            folder=OUT/'head'/s/arm;done=read(folder/'done.json');steps=read(folder/'steps.json')
            assert len(steps)==done['steps'] and all(r['norm']>=0 for r in steps)
            cp=folder/('e%d.pt'%lock['epoch']);assert sha(cp)==lock['checkpoints'][s][arm]
            models[s+'_'+arm]=dict(steps=len(steps),positive_grads=sum(i['norm']>0 for i in steps),sha=sha(cp),epoch=lock['epoch'])
    dump(OUT/'results/integrity.json',dict(archives=a,models=models,
        donor_file=sha(OUT/'inputs/donors.json'),candidate_proof=sha(OUT/'results/multi_audit.json')))
if __name__=='__main__':main()

"""Verify the local external checkpoint mirror and all test-image generation records."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import torch
from compare import OUT,load,read,sha,dump
from single_scores import RFPO,FORM

def main(stage,dest):
    base=Path(dest)
    idx=read(OUT/'archives'/(stage+'_index.json'))
    rows={k:{r['image']:r for r in read(p)} for k,p in [('primary',RFPO/'inputs/panel.json'),('official',FORM/'inputs/ucm.json')]}
    results={}
    for arm,z in idx.items():
        assert z['adapter_sha']==sha(base/arm/'adapter/adapter_model.safetensors') and z['optimizer_sha']==sha(base/arm/'resume.pt')
        archive=OUT/'archives'/z['archive']; assert sha(archive)==z['archive_sha']
        ab=OUT/stage/arm/'evaluate'
        packs={sub:load(ab/(sub+'.pt')) for sub in rows}
        ids={sub:{name:i for i,name in enumerate(pack['images'])} for sub,pack in packs.items()}
        seen={s:set() for s in rows};manifest={}
        with tarfile.open(archive,'r|gz') as tf:
            for m in tf:
                sub,name=m.name.split('/')
                b=tf.extractfile(m).read()
                if name=='complete.json':
                    manifest[sub]=json.loads(b)
                    assert manifest[sub]==read(ab/sub/'complete.json')
                    continue
                data=torch.load(io.BytesIO(b),weights_only=True)
                n=data['image'];r=rows[sub][n]
                assert sha256(b)==manifest[sub]['files'][n] and data['adapter']==z['adapter_sha'] and data['rgb']==r['rgb_sha256']
                assert data['query']==r['scene_query' if sub=='primary' else 'orig_query']
                assert torch.equal(data['v'],packs[sub]['features'][ids[sub][n]]) and data['prior'].shape==(3584,)
                assert n not in seen[sub];seen[sub].add(n)
        assert all(v==set(rows[k]) for k,v in seen.items())
        results[arm]=dict(adapter=z['adapter_sha'],resume=z['optimizer_sha'],archive=z['archive_sha'],groups={k:len(v) for k,v in seen.items()})
    dump(OUT/'results'/(stage+'_backup.json'),dict(checkpoints=str(base),verified=results))

def sha256(b):return hashlib.sha256(b).hexdigest()
if __name__=='__main__':main(*sys.argv[1:])

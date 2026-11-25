"""Snapshot the terminal policy, training state, per-image decisions and one-generation evidence."""
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(mode):
    arms={'exact_policy':('ordinary','joint'),'policy':('R2','old','new'),'mined_policy':('selected_ordinary','selected_joint','uniform_joint')}[mode]
    read=lambda p:json.loads(p.read_bytes())
    index={}
    for seed in 'ab':
        for arm in arms:
            key=seed+'_'+arm
            src=ROOT/mode/key;record=read(src/'complete.json');logs=src/'steps.json'
            ad=src/'adapter'/'adapter_model.safetensors';resume=src/'resume.pt'
            assert sha(logs)==record['logs'] and sha(ad)==record['final'] and resume.exists()
            dest=ROOT/'archives'/(mode+'_'+key+'.tar.gz');temp=dest.with_suffix('.part')
            count={}
            with tarfile.open(temp,'w:gz',compresslevel=3) as tf:
                for sub,n in [('primary',126),('official',210)]:
                    folder=src/'evaluate'/sub
                    manifest=folder/'complete.json'
                    cp=read(manifest);assert len(cp['files'])==n and sha(folder.with_suffix('.pt'))==cp['features']
                    count[sub]=n
                    tf.add(manifest,arcname=sub+'/complete.json')
                    for name,expected in cp['files'].items():
                        p=folder/(name+'.pt');assert sha(p)==expected
                        tf.add(p,arcname=sub+'/'+p.name)
            temp.replace(dest)
            index[key]=dict(files=count,archive=dest.name,archive_sha=sha(dest),adapter_sha=sha(ad),optimizer_sha=sha(resume),steps_sha=sha(logs))
    (ROOT/'archives'/(mode+'_index.json')).write_text(json.dumps(index,indent=2)+'\n')

if __name__=='__main__':main(sys.argv[1])

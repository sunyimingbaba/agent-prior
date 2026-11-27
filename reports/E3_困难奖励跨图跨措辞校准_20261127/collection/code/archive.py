"""Package provenance-verified CPU and GPU output and rehash every archive member."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(p):return json.loads(p.read_bytes())

def material():
    groups={}
    for name in ('train98','pool270'):
        generation=read(ROOT/name/'generation.json')
        files={f'{name}/raw/{key}.pt':value for key,value in generation['files'].items()}
        groups[name+'_states']=files
    files={}
    for n in (0,1):
        proof=read(ROOT/'pool270'/f'scoring_{n}.json')
        for k,h in proof['files'].items():
            path=f'pool270/scores/{k}.json'
            assert path not in files
            files[path]=h
            files[f'pool270/readouts/{k}.pt']=proof['projections'][k]
            r=read(ROOT/path)
            assert r['source']==groups['pool270_states'][f'pool270/raw/{k}.pt'] and r['projection']==proof['projections'][k]
    assert [len(groups[k]) for k in groups]==[98,238] and len(files)==476
    groups['pool270_scores']=files
    return groups

def main(check):
    groups=material();folder=ROOT/'archive';folder.mkdir(exist_ok=True)
    index={}
    for k,files in groups.items():
        path=folder/(k+'.tar.gz')
        if not check:
            tmp=path.with_suffix('.part')
            with tarfile.open(tmp,'w:gz',compresslevel=3) as tf:
                for name,digest in sorted(files.items()):
                    assert sha(ROOT/name)==digest
                    tf.add(ROOT/name,arcname=name)
            tmp.replace(path)
        with tarfile.open(path,'r|gz') as tf:
            seen={}
            for m in tf:
                assert m.isfile() and m.name not in seen
                seen[m.name]=hashlib.file_digest(tf.extractfile(m),'sha256').hexdigest()
                assert seen[m.name]==sha(ROOT/m.name)
            assert seen==files
        index[k]=dict(file=path.name,bytes=path.stat().st_size,members=len(files),sha256=sha(path))
    p=folder/'index.json'
    if check:assert read(p)==index
    else:p.write_text(json.dumps(index,indent=2)+'\n')
    print(index)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');main(p.parse_args().check)

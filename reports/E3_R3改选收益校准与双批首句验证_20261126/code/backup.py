"""Archive byte-verified raw trajectories and reader outputs from both seeds."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    dest=ROOT/'archives';dest.mkdir(exist_ok=True)
    roster=json.loads((ROOT/'inputs/real_roster.json').read_text())
    names={r['image'] for r in roster}
    index={}
    for seed in 'ab':
        base=ROOT/'records/real'/seed
        path=dest/(seed+'.tar.gz')
        with tarfile.open(path,'w:gz',compresslevel=3) as tar:
            for part in ('raw','features'):
                folder=base/part
                manifests=sorted(folder.glob('complete*.json'))
                hashes={k:h for f in manifests for k,h in json.loads(f.read_text())['files'].items()}
                assert set(hashes)==names
                for f in manifests:tar.add(f,arcname=part+'/'+f.name)
                for name in sorted(hashes):
                    f=folder/(name+'.pt');assert digest(f)==hashes[name]
                    tar.add(f,arcname=part+'/'+f.name)
        # Read the generated archive in full and verify the raw trajectories and projected features for every image.
        with tarfile.open(path,'r:gz') as tar:
            for part in ('raw','features'):
                mats=[m for m in tar.getmembers() if m.name.startswith(part+'/complete')]
                hashes={k:h for m in mats for k,h in json.load(tar.extractfile(m))['files'].items()}
                assert set(hashes)==names
                for name,expected in hashes.items():
                    stream=tar.extractfile(part+'/'+name+'.pt')
                    assert hashlib.sha256(stream.read()).hexdigest()==expected
        index[seed]=dict(images=len(names),archive_sha256=digest(path),raw_manifest=digest(base/'raw/complete.json'),
                         size=path.stat().st_size,adapter=json.loads((base/'raw/complete.json').read_text())['adapter'])
    (dest/'index.json').write_text(json.dumps(index,indent=2)+'\n')


if __name__=='__main__':main()

"""Archive each first-caption group with the manifest produced by the sampler."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    target = ROOT/'archives'
    target.mkdir(exist_ok=True)
    index = {}
    for s in 'ab':
        raw = ROOT/'records/real'/s/'raw'
        manifest = raw/'complete.json'
        cp = json.loads(manifest.read_bytes())
        assert len(cp['files']) == 282
        archive = target/('real_all_'+s+'.tar.gz')
        tmp = target/(archive.name+'.part')
        with tarfile.open(tmp,'w:gz',compresslevel=3) as tf:
            tf.add(manifest,arcname=manifest.name)
            for name,checksum in cp['files'].items():
                p=raw/(name+'.pt')
                assert sha(p)==checksum
                tf.add(p,arcname=p.name)
        tmp.replace(archive)
        index[archive.name]=dict(groups=len(cp['files']),sha=sha(archive),manifest=sha(manifest),adapter=cp['adapter'])
    (target/'real_index.json').write_text(json.dumps(index,indent=2)+'\n')

if __name__=='__main__':
    main()

"""Keep full trajectory and feature evidence with an explicit checksum per image."""
from pathlib import Path
import json,hashlib,tarfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    entries={};dest=ROOT/'archives';dest.mkdir(exist_ok=True)
    for mode,tags in [('candidate',('fresh','extension')),('ordinary',('first','second','third'))]:
        for tag in tags:
            for s in 'ab':
                base=ROOT/'records'/tag/s if mode=='candidate' else ROOT/'records/ordinary'/tag/s
                folder=base/'raw' if mode=='candidate' else base
                p=folder/'complete.json';cp=json.loads(p.read_bytes());paths=[]
                for n,expected in cp['files'].items():
                    f=folder/(n+'.pt');assert sha(f)==expected;paths.append(f)
                a=dest/(mode+'_'+tag+'_'+s+'.tar.gz')
                if not a.exists():
                    temp=a.with_suffix('.part')
                    with tarfile.open(temp,'w:gz',compresslevel=3) as tf:
                        for f in [p]+paths:tf.add(f,arcname=f.name)
                    temp.replace(a)
                entries[a.name]=dict(files=len(paths),sha=sha(a),source_manifest=sha(p),adapter=cp['adapter'])
    (dest/'index.json').write_text(json.dumps(entries,indent=2)+'\n')
if __name__=='__main__':main()

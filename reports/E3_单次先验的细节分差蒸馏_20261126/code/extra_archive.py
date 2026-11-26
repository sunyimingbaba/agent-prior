from pathlib import Path
import hashlib,json,tarfile,sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    idx={};dest=ROOT/'archives'
    for d in sorted((ROOT/'cross').iterdir()):
        cp=json.loads((d/'complete.json').read_bytes());archive=dest/('cross_'+d.name+'.tar.gz');tmp=archive.with_suffix('.part')
        assert sha(d/'features.pt')==cp['features'] and len(cp['images'])==448
        with tarfile.open(tmp,'w:gz',compresslevel=3) as tf:
            tf.add(d/'complete.json',arcname='complete.json')
            for n,h in cp['images'].items():
                p=d/(n+'.pt');assert sha(p)==h;tf.add(p,arcname=p.name)
        tmp.replace(archive);idx[d.name]=sha(archive)
    assert len(idx)==(2 if sys.argv[1:]==["reachable"] else 4)
    log=dest/'all_execution_logs.tar.gz'
    with tarfile.open(log,'w:gz') as tf:tf.add(ROOT/'logs',arcname='logs')
    (dest/'extra_index.json').write_text(json.dumps(dict(cross=idx,logs=sha(log)),indent=2)+'\n')
if __name__=='__main__':main()

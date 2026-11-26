from pathlib import Path
import hashlib,json,tarfile,sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    dest=ROOT/'archives';dest.mkdir(exist_ok=True);index={}
    for folder in sorted((ROOT/'evaluate').iterdir()):
        archive=dest/(folder.name+'.tar.gz');tmp=archive.with_suffix('.part')
        with tarfile.open(tmp,'w:gz',compresslevel=3) as tf:
            manifest={}
            for source in sorted(p for p in folder.iterdir() if p.is_dir()):
                cp=json.loads((source/'complete.json').read_bytes());assert sha(source.with_suffix('.pt'))==cp['features']
                manifest[source.name]=len(cp['images']);tf.add(source/'complete.json',arcname=source.name+'/complete.json')
                for name,h in cp['images'].items():
                    f=source/(name+'.pt');assert sha(f)==h;tf.add(f,arcname=source.name+'/'+f.name)
        tmp.replace(archive);index[folder.name]=dict(sha=sha(archive),images=manifest)
    assert len(index)==8
    log=dest/'execution_logs.tar.gz'
    with tarfile.open(log,'w:gz') as t:t.add(ROOT/'logs',arcname='logs')
    (dest/'index.json').write_text(json.dumps(dict(groups=index,logs=sha(log)),indent=2)+'\n')
if __name__=='__main__':main()

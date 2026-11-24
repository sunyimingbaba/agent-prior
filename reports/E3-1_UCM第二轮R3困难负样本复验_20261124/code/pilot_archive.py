import tarfile
from base import RUN,read,sha,dump
def main():
    root=RUN/'pilot';out=root/'traces';out.mkdir(exist_ok=True)
    index=read(root/'vectors/index.json');assert len(index)==12
    hashes={}
    for folder in sorted((root/'greedy').iterdir()):
        dest=out/(folder.name+'.tar.gz')
        with tarfile.open(dest,'w:gz',compresslevel=3) as tar:
            for subset in sorted(folder.iterdir()):
                proof=read(subset/'complete.json')
                for im,h in proof['files'].items():assert sha(subset/(im+'.pt'))==h
            tar.add(folder,arcname=folder.name)
        hashes[dest.name]=sha(dest)
    assert len(hashes)==6
    dump(root/'archives.json',dict(traces=hashes,vectors=index,train=sha(root/'train.json'),contexts={s:sha(root/(s+'_contexts.pt')) for s in ('a','b')},
            pilot_protocol=sha(RUN/'短训口径.md')))
if __name__=='__main__':main()

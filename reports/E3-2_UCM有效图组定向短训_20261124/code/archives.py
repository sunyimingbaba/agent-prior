"""Package each verified evaluation state as soon as it is complete."""
import tarfile,time
from common import ROOT,OLD,ARMS,read,dump,sha
def wait(p):
    while not p.is_file():time.sleep(2)
def main():
    pil=ROOT/'pilot';out=pil/'traces';out.mkdir(exist_ok=True);traces={}
    for seed in ('a','b'):
        for arm in ('start',*ARMS):
            tag=seed+'_'+arm;folder=pil/'greedy'/tag
            for sub in (('fresh',) if arm=='start' else ('fresh','val')):
                p=folder/sub/'complete.json';wait(p)
                for nm,h in read(p)['files'].items():assert h==sha(folder/sub/(nm+'.pt'))
            dest=out/(tag+'.tar.gz');temp=dest.with_suffix('.part')
            with tarfile.open(temp,'w:gz',compresslevel=3) as tar:tar.add(folder,arcname=tag)
            temp.replace(dest);traces[dest.name]=sha(dest)
    index=pil/'vectors/index.json';wait(index)
    dump(pil/'archives.json',dict(traces=traces,vectors=read(index),contexts={s:sha(ROOT/'records'/(s+'_contexts.pt')) for s in ('a','b')},
        rosters=sha(ROOT/'inputs/rosters.json'),tools=sha(ROOT/'records/tools.json'),fresh_text=sha(ROOT/'records/fresh_text.pt'),screen=sha(ROOT/'screen/records/extra_provenance.json'),protocol=sha(ROOT/'实验口径.md')))
if __name__=='__main__':main()

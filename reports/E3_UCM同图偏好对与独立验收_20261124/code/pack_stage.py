import sys,tarfile,time
from env_pair import ROOT,read,dump,sha,weight
def main(stage,tags):
    dest=ROOT/'archives';dest.mkdir(exist_ok=True);index={}
    for tag in tags:
        seed,arm=tag.split('_',1);sig=weight(seed) if arm=='start' else read(ROOT/'pilot'/tag/'complete.json')['final']
        subs=('official','blind') if stage=='confirm' else ('val','fresh')
        df=dest/(stage+'_'+tag+'.tar.gz');tmp=df.with_suffix('.part')
        with tarfile.open(tmp,'w:gz',compresslevel=3) as tar:
            for sub in subs:
                directory=ROOT/'pilot/greedy'/tag/sub;f=directory/'complete.json'
                while not f.is_file():time.sleep(2)
                v=read(f);assert v['adapter']==sig
                for nm,h in v['files'].items():assert sha(directory/(nm+'.pt'))==h
                tar.add(directory,arcname=tag+'/'+sub)
        tmp.replace(df);index[df.name]=sha(df)
    dump(dest/(stage+'_index.json'),index)
if __name__=='__main__':main(sys.argv[1],sys.argv[2:])

"""Apply the same frozen candidate selector, using each actual-caption condition's own greedy anchor."""
import tempfile
from pathlib import Path
import numpy as np
import multi_view as mv
from compare import OUT,read,dump,sha,boot
def main():
    lock=read(OUT/'inputs/multi_nomination.json');plan=read(OUT/'inputs/real_lock.json')
    assert plan['reward']==sha(OUT/'inputs/multi_nomination.json') and plan['roster']==sha(OUT/'inputs/real_roster.json') and lock['files']==mv.signature()
    roster=read(OUT/'inputs/real_roster.json')
    donors=mv.donor_bank()[0];arch=read(OUT/'inputs/donors.json');assert all([list(r) for r in ds]==arch[str(k)] for k,ds in donors.items())
    ss={};allrs=[];drop={};m=lock['method']
    for tag in ('fresh','extension','saved','extra'):
        source=OUT/'records' if tag in ('fresh','extension') else OUT/'records/covered/bank'
        cases=[r for r in roster if r['tag']==tag]
        with tempfile.TemporaryDirectory() as tmp:
            scratch=Path(tmp);(scratch/'inputs').mkdir();(scratch/'records').mkdir()
            dump(scratch/'inputs/fresh.json',cases)
            for n in ('text.pt','read_complete.json'):(scratch/'records'/n).symlink_to(source/n)
            for s in 'ab':
                rawdir=OUT/'records/real'/s/'features';dest=scratch/'records/fresh'/s/'features';dest.mkdir(parents=True)
                head={n:v for j in range(2) for n,v in read(rawdir/('complete_'+str(j)+'.json'))['files'].items()}
                assert set(head)=={r['image'] for r in roster}
                for c in cases:
                    n=c['image'];(dest/(n+'.pt')).symlink_to(rawdir/(n+'.pt'))
                for i in range(2):dump(dest/('complete_'+str(i)+'.json'),dict(files={r['image']:head[r['image']] for r in cases[i::2]}))
            mv.OUT=scratch
            try:fs,drop[tag]=mv.folds('fresh')
            finally:mv.OUT=OUT
            for f in fs:f['tag']=tag
            rs=mv.collect(fs,[lock['weights']],{'all':lock['model']});allrs.extend(rs)
            dump(OUT/'records'/('real_'+tag+'.json'),rs)
    for g,tags in [('first138',('fresh','extension')),('other144',('saved','extra')),('all',('fresh','extension','saved','extra'))]+[(t,(t,)) for t in ('fresh','extension','saved','extra')]:
        rs=[r for r in allrs if r['tag'] in tags];ss[g]=mv.summary(rs,[m,'random'])
        ss[g]['contrasts']={k:mv.contrasts(rs,m,k) for k in ('old','R2','random','anchor')}
        scene={}
        for t in tags:
            p=OUT/'records'/((('multi_'+t) if t in ('fresh','extension') else ('covered_'+t+'_images'))+'.json')
            scene.update({(r['image'],r['seed']):r for r in read(p)})
        ss[g]['interaction']={}
        for i,key in enumerate(('joint','mrr','detail')):
            x=[(r['means'][m][i]-r['means']['old'][i])-(scene[r['image'],r['seed']]['means'][m][i]-scene[r['image'],r['seed']]['means']['old'][i]) for r in rs]
            ss[g]['interaction'][key]=boot(x,np.ones(len(rs)),[r['tag']+'|'+r['pair'] for r in rs])
    z=ss['all']['contrasts']['old'];ok=all(ss[t]['contrasts']['old']['joint']['mean']>0 for t in ('first138','other144')) and z['joint']['interval'][0]>0 and z['mrr']['interval'][0]>-.01 and z['detail']['mean']>=0
    dump(OUT/'results/real_confirmation.json',dict(eligible=ok,plan=sha(OUT/'inputs/real_lock.json'),nomination=plan['reward'],excluded=drop,**ss))
    print('real input',ok,z,flush=True)
if __name__=='__main__':main()

"""Evaluate the already frozen reward on held-out photographs of covered attributes."""
from pathlib import Path
from collections import defaultdict
import tempfile
import sys,torch,numpy as np
import multi_view as mv
from compare import OUT,SOURCE,OLD,PRIOR,read,dump,sha,load,boot
from panels import question_panels

def make_text(rows):
    # Use the exact stored C/D/E strings. Each feature comes from the same frozen text reader.
    roots=[OLD,PRIOR/'guard',PRIOR/'reward_confirm']
    lookup={};diff=0;native=None
    for p in roots:
        f=p/'records/text.pt';assert sha(f)==read(p/'records/read_complete.json')['text']
        b=load(f)
        if native is None:native=b
        else:assert torch.equal(native['native'],b['native']) and torch.equal(native['native_tokens'],b['native_tokens'])
        for i,t in enumerate(b['texts']):
            if t in lookup:
                old,ot=lookup[t];assert torch.equal(ot,b['tokens'][i]);diff=max(diff,float((old-b['hard'][i]).abs().max()))
            else:lookup[t]=(b['hard'][i],b['tokens'][i])
    assert diff<2e-5,diff
    strings=sorted({x for r in rows for k in 'ABCDE' for x in r[k]});ids={x:i for i,x in enumerate(strings)}
    native_ids=read(OUT/'inputs/native_index.json')['columns'];maps={}
    for r in rows:
        maps[r['image']]=dict(native=[native_ids[t] for t in r['captions']],**{k:[ids[t] for t in r[k]] for k in 'ABCDE'})
    dest=OUT/'records/covered/bank';dest.mkdir(exist_ok=True,parents=True);f=dest/'text.pt'
    torch.save(dict(native=native['native'],native_tokens=native['native_tokens'],texts=strings,hard=torch.stack([lookup[t][0] for t in strings]),
                    tokens=torch.stack([lookup[t][1] for t in strings]),maps=maps),f)
    dump(dest/'read_complete.json',dict(text=sha(f),max_text_component_difference=diff,source={str(p.name):sha(p/'records/text.pt') for p in roots}))
def saved(rows):
    eligible={r['image']:r for r in rows};out=OUT/'records/covered/saved';proof=defaultdict(dict)
    for tag in ('development','repeat'):
        for c,s,mode,b,v,rk,_,_,_ in question_panels(tag):
            n=c['image']
            if mode!='scene' or n not in eligible:continue
            assert c['rgb']==eligible[n]['rgb_sha256'] and v.shape==(17,768)
            dst=out/s/'features';dst.mkdir(exist_ok=True,parents=True);p=dst/(n+'.pt')
            torch.save(dict(visual=torch.from_numpy(v).float(),ranks=rk.tolist(),rgb=c['rgb']),p)
            proof[s][n]=sha(p)
    for s in 'ab':
        assert set(proof[s])==set(eligible)
        keys=sorted(eligible)
        for i in range(2):dump(out/s/'features'/('complete_'+str(i)+'.json'),dict(files={n:proof[s][n] for n in keys[i::2]}))
def setup():
    rs=read(OUT/'inputs/covered_saved.json');make_text(rs+read(OUT/'inputs/covered_extra.json'));saved(rs)
def score():
    lock=read(OUT/'inputs/multi_nomination.json');plan=read(OUT/'inputs/covered_lock.json')
    assert lock['files']==mv.signature() and plan['source']==sha(OUT/'inputs/multi_nomination.json')
    sup={r['attribute'] for tag in ('first','second','third') for r in read(SOURCE/'inputs'/(tag+'.json'))}
    # Cache and check the original training-only donors before directing the frozen evaluator to the replication inputs.
    by,names=mv.donor_bank();archived=read(OUT/'inputs/donors.json')
    assert all([list(row) for row in ds]==archived[str(k)] for k,ds in by.items())
    allrs=[];ss={};drop={}
    with tempfile.TemporaryDirectory() as tmp:
        scratch=Path(tmp);(scratch/'inputs').mkdir();(scratch/'records').mkdir()
        for f in ('text.pt','read_complete.json'):(scratch/'records'/f).symlink_to(OUT/'records/covered/bank'/f)
        for alias,tag in [('fresh','saved'),('extension','extra')]:
            p=OUT/'inputs'/('covered_'+tag+'.json')
            assert sha(p)==plan['files'][tag] and all(r['attribute'] in sup for r in read(p))
            (scratch/'inputs'/(alias+'.json')).symlink_to(p);(scratch/'records'/alias).symlink_to(OUT/'records/covered'/tag)
            mv.OUT=scratch
            try:fs,excluded=mv.folds(alias)
            finally:mv.OUT=OUT
            for f in fs:f['tag']=tag
            rs=mv.collect(fs,[lock['weights']],{'all':lock['model']});allrs.extend(rs);drop[tag]=excluded
            dump(OUT/'records'/('covered_'+tag+'_images.json'),rs)
            ss[tag]=mv.summary(rs,['random',lock['method']]);ss[tag]['change']=mv.contrasts(rs,lock['method'],'old')
    m=lock['method'];ss['pooled']=mv.summary(allrs,['random',m])
    ss['pooled']['change']=mv.contrasts(allrs,m,'old');ss['pooled']['change_R2']=mv.contrasts(allrs,m,'R2')
    u=ss['pooled']['change'];ok=all(ss[t]['change']['joint']['mean']>0 for t in ('saved','extra')) and u['joint']['interval'][0]>0 and u['mrr']['interval'][0]>-.01 and u['detail']['mean']>=0
    dump(OUT/'results/covered_confirmation.json',dict(nomination=plan['source'],plan=sha(OUT/'inputs/covered_lock.json'),eligible=ok,excluded=drop,**ss))
    print('eligible',ok,'both',u,flush=True)
if __name__=='__main__':{'setup':setup,'score':score}[sys.argv[1]]()

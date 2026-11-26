import sys
import numpy as np
from local import ROOT,FORM,load,read,dump,sha
from transfer_scores import counts,summary,scalar_values
def main(mode="base"):
    dest=ROOT if mode=="base" else ROOT/mode
    arms=("retained","parent","teacher") if mode=="base" else ("retained","parent","teacher","reachable_teacher")
    strict=mode=="strict_rfpo"
    if strict:arms=("retained","parent","strict_coldordinary","strict_warmstrict","strict_coldstrict")
    target=arms[-1]
    rows=read(FORM/'inputs/rsitmd.json');names=[r['image'] for r in rows]
    gal=load(FORM/'results/rsitmd_text.pt');historical=read(FORM/'results/paired_results.json')['rsitmd'];out={};by=[]
    for mode in ('native','alternatives'):
        raw={}
        for s in 'ab':
            for a in arms:
                if a=='retained':
                    z=load(FORM/'results'/('rsitmd_'+s+'_R2.pt'));assert z['manifest']==sha(FORM/'inputs/rsitmd.json')
                else:
                    stage=ROOT/'strict_rfpo' if a.startswith('strict_') else ROOT/'reachable' if a.startswith('reachable_') else ROOT
                    p=stage/'cross'/(s+'_'+a.removeprefix('reachable_').removeprefix('strict_'))/'features.pt';q=read(p.with_name('complete.json'));assert sha(p)==q['features'] and q['protocol']==sha(stage/'跨库普通检索复核.md')
                    z=load(p);assert z['images']==names
                ids,arr,ds=counts(z['features'],gal,mode)
                if raw:
                    assert np.array_equal(ids,ref_ids) and np.array_equal(ds,ref_ds)
                else:ref_ids,ref_ds=ids,ds
                raw[s,a]=arr
                if a=='retained':
                    v=scalar_values(arr.sum(0),np.asarray(len(ds)),np.asarray(ds.sum()))[-1]
                    assert abs(100*v-historical['all_'+mode]['levels'][s+'_R2']['mR'])<1e-9
                by.extend(dict(seed=s,arm=a,mode=mode,image=names[i],text_count=int(d),hits=x.tolist()) for i,d,x in zip(ids,ds,arr))
        out[mode]=dict(images=len(ids),captions=int(ds.sum()),levels={},teacher_minus={})
        for a in arms:
            out[mode]['levels'][a]=summary(np.mean([raw[s,a] for s in 'ab'],0),ds)
            if a!=target:
                diff=np.stack([raw[s,target]-raw[s,a] for s in 'ab'])
                out[mode]['teacher_minus'][a]=dict(scores=summary(diff.mean(0),ds),
                    seeds=[float(scalar_values(x.sum(0),np.asarray(len(ds)),np.asarray(ds.sum()))[-1]) for x in diff])
            if strict and a!='strict_warmstrict':
                diff=np.stack([raw[s,'strict_warmstrict']-raw[s,a] for s in 'ab'])
                out[mode].setdefault('warm_minus',{})[a]=dict(scores=summary(diff.mean(0),ds),
                    seeds=[float(scalar_values(x.sum(0),np.asarray(len(ds)),np.asarray(ds.sum()))[-1]) for x in diff])
    dump(dest/'results/cross_scores.json',out);dump(dest/'results/cross_images.json',by)
if __name__=='__main__':main(*sys.argv[1:])

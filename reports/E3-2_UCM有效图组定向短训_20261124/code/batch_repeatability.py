"""Post hoc sampling-frequency diagnostic, evaluated on the other eight draws from the same start policy."""
from collections import Counter
import numpy as np,torch
from verify_screen import unpack
from score import OLD,ROOT,load,read,dump,write_csv
def main():
    torch.set_num_threads(2);out={};raw=[]
    for seed in ('a','b'):
        folds=[]
        for sub in ('','confirm'):
            prefix=OLD/sub;path=prefix/'records';bank=load(path/'text.pt');ext=ROOT/'screen/records' if sub else path
            batches=(unpack(path,seed),unpack(ext,'extra_'+seed));h=bank['hard'].double().numpy()
            for c in read(prefix/'inputs/panel.json'):
                nm=c['image'];m=bank['maps'][nm];counts=[];is_post=batches[0][nm][0]['post_tool']
                if not is_post:continue
                for rec in batches:
                    s,v=rec[nm];assert s['post_tool']==is_post;x=v['visual'].double().numpy()
                    a=x@(h[m['A'][0]]-h[m['A'][1]]);b=x@(h[m['B'][0]]-h[m['B'][1]])
                    good=int(np.sum((a>0)&(b>0)));bad=int(np.sum((a<0)&(b<0)))
                    counts.append([good,bad,8-good-bad])
                folds.append((nm,counts));raw.append(dict(seed=seed,image=nm,attribute=c['attribute'],**{f'{label}_{i+1}':v for i,trip in enumerate(counts) for label,v in zip(('good','bad','other'),trip)}))
        rs={}
        for i in (0,1):
            pool=[counts[1-i] for nm,counts in folds if min(counts[i][:2])>0]
            rs[str(i+1)]=dict(selected=len(pool),repeat_mixed=sum(min(c[:2])>0 for c in pool),repeat_all_good=sum(c[0]==8 for c in pool),repeat_all_bad=sum(c[1]==8 for c in pool),good=int(np.array(pool)[:,0].sum()),bad=int(np.array(pool)[:,1].sum()),other=int(np.array(pool)[:,2].sum()))
        union=sum(any(min(c[:2])>0 for c in pair) for nm,pair in folds)
        both=sum(all(min(c[:2])>0 for c in pair) for nm,pair in folds)
        combined=sum(all(sum(c[i] for c in pair)>0 for i in (0,1)) for nm,pair in folds)
        assert combined==len(read(ROOT/'inputs/rosters.json')[seed]['eligible'])
        out[seed]=dict(first_to_second=rs['1'],second_to_first=rs['2'],ever_mixed_in_eight=union,both_batches_mixed=both,only_after_pooling=combined-union)
    write_csv(ROOT/'results/screen_batch_counts.csv',raw)
    dump(ROOT/'results/batch_repeatability.json',out)
if __name__=='__main__':main()

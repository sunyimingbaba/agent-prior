"""Resample whole attributes, keeping both pairs, seed repetitions and prompt arms together."""
import numpy as np
from analyze import ROOT,read,dump
FIRST=ROOT.parent/'E3_UCM中性观察问题候选对照_20261125'
def main():
    batches=[read(p/'results/per_image.json') for p in (FIRST,ROOT)]
    attrs=sorted({x['attribute'] for x in batches[0]})
    assert len(attrs)==27
    for key in ('image','pair'):assert not {x[key] for x in batches[0]}&{x[key] for x in batches[1]}
    def mat(mode,key):
        # Within each cell lie two sides from each of two disjoint natural pairs.
        data=np.array([[[x[key] for batch in batches for x in batch if x['seed']==seed and x['attribute']==attr and x['mode']==mode] for attr in attrs] for seed in 'ab'],float)
        assert data.shape==(2,27,4)
        return data
    draws=np.random.default_rng(20261125).integers(27,size=(20000,27));ones=np.ones((2,27,4))
    def estimate(num,den):
        top=num.sum(axis=(0,2))[draws].sum(1);bottom=den.sum(axis=(0,2))[draws].sum(1);valid=bottom>0
        ci=np.quantile(top[valid]/bottom[valid],[.025,.975]).tolist() if valid.any() else [None,None]
        return dict(mean=float(num.sum()/den.sum()) if den.sum() else None,interval=ci,
            seeds=[float(a.sum()/b.sum()) if b.sum() else None for a,b in zip(num,den)],defined=int(valid.sum()))
    out={}
    for key in ['available','good_draws','preserved','R2','R3','R2_preserved','R3_preserved','eligible','greedy_detail','greedy_rr','best_detail','R2_detail','R3_detail','R2_rr','R3_rr','with_tool','unique']:
        scene=mat('scene',key);question=mat('question',key)
        out[key]={k:estimate(v,ones) for k,v in [('scene',scene),('question',question),('change',question-scene)]}
    for m in ['scene','question']:
        den=mat(m,'available')
        out['conditional_'+m]={key:estimate(mat(m,key),den) for key in ('R2','R3','increment')}
    repeat=read(ROOT/'results/summary.json')
    agreed=repeat['advance_to_training'] and all(row['interval'][0] is not None and row['interval'][0]>0
        for row in [out['available']['change'],out['conditional_question']['increment']])
    dump(ROOT/'results/pooled.json',dict(unique_images=108,pairs=54,attributes=27,sampled=6912,greedy=432,metrics=out,advance_to_training=bool(agreed)))
if __name__=='__main__':main()

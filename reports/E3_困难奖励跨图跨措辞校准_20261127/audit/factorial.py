"""Separate the effect of changing the written claim from switching its source context."""
from collections import defaultdict
import json

import numpy as np

from independent import OUT,STAGE,infer,load,pool,stats


def success(x):
    return np.where(x>0,1.,np.where(x==0,0.5,0.))


def main():
    bank,cohort,_=pool()
    grouped=defaultdict(dict)
    d=STAGE/"collection/pool270/raw"
    for r in cohort:
        trace=load(d/(r["image"]+".pt"))
        assert trace["rgb"]==r["rgb_sha256"] and trace["side"]==r["side"]
        projected=infer(trace["states"],bank[r["attribute"]],side=0).reshape(2,2)
        # Both contexts below always use side zero's word first and its opposite second.
        if r["side"]:
            projected=projected[:,::-1]
        key=(r['cohort'],r['pair'])
        grouped[key][r["side"]]=projected
    summaries=[]
    for pair,vals in grouped.items():
        a,b=vals[0],vals[1]
        word=np.column_stack((a[:,0]-a[:,1],b[:,0]-b[:,1]))
        context=a-b
        # Rows denote D/E. Columns index the context or the fixed word.
        m={}
        for w,(j,cs) in enumerate(zip(word,context)):
            label="DE"[w]
            m["word_acc_"+label]=success(j).mean()
            m["context_acc_"+label]=success(cs).mean()
            m["word_gain_"+label]=j.mean()
            m["context_gain_"+label]=cs.mean()
            m["opposite_word_preference_"+label]=float(j[0]*j[1]<0)
            m["interaction_"+label]=j[0]-j[1]
        m.update(context_acc=success(context).mean(),word_acc=success(word).mean(),
                 acc_difference=success(word).mean()-success(context).mean(),
                 context_gain=context.mean(),word_gain=word.mean())
        summaries.append((pair,m))
    out=stats(summaries);out["images"]=len(cohort);out["data_unit"]="one mirrored pair"
    (OUT/"同字替换上下文.json").write_text(json.dumps(out,indent=2)+"\n")
    print(out)


if __name__=="__main__":main()

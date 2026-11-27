"""Train context-oriented contrasts and evaluate the prespecified symmetric two-phrase diagnostic."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent))
from independent import OLD,REFERENCE,STAGE,load,norm,pool,read,stats
from factorial import success


def fitted_axes():
    rows=read(REFERENCE/"fresh98/inputs/rows.json")
    groups=defaultdict(dict)
    values=defaultdict(list)
    deltas=defaultdict(list)
    for row in rows:
        mat=norm(load(OLD/"raw"/(row["image"]+".pt"))["states"].double().numpy())
        side=row["side"]
        canonical=mat if side==0 else mat[::-1]
        groups[row["attribute"],row["pair"]][side]=canonical
        values[row["attribute"]].extend(canonical)
    for (attr,pair),sides in groups.items():
        deltas[attr].append((sides[0]-sides[1]).mean(axis=0))
    return ({k:norm(np.mean(v,axis=0)) for k,v in deltas.items()},
            {k:np.mean(v,axis=0) for k,v in values.items()})


def main():
    word_axis,rows,_=pool()
    axis,center=fitted_axes()
    assert set(axis)==set(center)==set(word_axis)
    allgroups=defaultdict(dict)
    for row in rows:
        source=load(STAGE/"collection/pool270/raw"/(row["image"]+".pt"))
        assert source['texts']==["<answer>"+t+"</answer>" for k in ("D","E") for t in row[k]]
        hidden=norm(source["states"].double().numpy()).reshape(2,2,-1)
        attr=row["attribute"];s=row["side"]
        # Reverse the two phrases in records from side one. From here on the model
        # receives the same canonical pair of sentences for either side.
        h=hidden[:,::1 if not s else -1,:]
        q=h@axis[attr]
        lex=h@word_axis[attr]
        projected=(h.mean(axis=1)-center[attr])@axis[attr]
        allgroups[row["cohort"],row["pair"]][s]=(q,lex,projected)
    report=[]
    for pair,images in allgroups.items():
        (q0,l0,s0),(q1,l1,s1)=images[0],images[1]
        context=success(q0-q1)
        word=np.column_stack((success(q0[:,0]-q0[:,1]),success(q1[:,0]-q1[:,1])))
        side=np.column_stack((success(s0),success(-s1)))
        word_truth=success(np.column_stack((l0[:,0]-l0[:,1],l1[:,0]-l1[:,1])))
        truth=np.column_stack((success(np.sign(s0)*(l0[:,0]-l0[:,1])),success(np.sign(s1)*(l1[:,1]-l1[:,0]))))
        m={}
        for name,a in [('same_words_context_acc',context),('same_context_words_acc',word),('two_phrase_side_acc',side),
                       ('predicted_side_truth_acc',truth),('oracle_side_truth_acc',word_truth),
                       ('context_acc_minus_word_acc',context-word),('predicted_minus_oracle',truth-word_truth)]:
            m[name]=a.mean()
            for idx,label in enumerate("DE"):m[name+'_'+label]=a[idx].mean()
        report.append((pair,m))
    obj=stats(report);obj["images"]=len(rows);obj["train_images"]=98
    (OUT/"结果.json").write_text(json.dumps(obj,indent=2)+"\n")
    print({k:v for k,v in obj["metrics"].items() if k.split('_')[-1] not in ('D','E')})


if __name__=="__main__":main()

"""Re-score the locked top candidates on F/G, preserving each tied set and the original image pairing."""
from collections import defaultdict
import gzip
import json
import numpy as np
import torch
from data import ROOT,read,sha,stream,summarize

ARMS=('ordinary','old','old3','switch','local3')


def load(tag,prefix=''):
    with gzip.open(ROOT/'results'/(prefix+tag+'.jsonl.gz'),'rt') as f:
        return {(r['image'],r['seed'],r['block'],r['held']):r for r in (json.loads(line) for line in f)}


def main():
    pack=torch.load(ROOT/'inputs/fresh_text.pt',map_location='cpu',weights_only=True)
    assert pack['source']==sha(ROOT/'inputs/独立措辞.json')
    mapping=read(ROOT/'inputs/独立措辞.json')
    words={w:i for i,w in enumerate(pack['words'])}
    text=pack['features'].double().numpy()
    output={};all_rows=[]
    for tag in ('v122','v148'):
        records=load(tag);local=load(tag,'neighbor_');assert records.keys()==local.keys()
        rows=[]
        for r,seed,_,bank,v in stream(tag):
            if len(set(bank['maps'][r['image']]['native']))<3:continue
            wording=mapping[r['attribute']]
            cols=np.array([[words[x] for x in group] for group in wording])
            scores=v[:16]@text[cols.ravel()].T
            side=r['side'];right=(scores.reshape(16,2,2)[:,:,side]>scores.reshape(16,2,2)[:,:,1-side]).astype(float)
            held_count=len(set(bank['maps'][r['image']]['native']))-1
            for block in (0,1):
                for held in range(held_count):
                    key=(r['image'],seed,block,held)
                    original=records[key];q={k:v for k,v in original.items() if k!='outcomes'}
                    q['outcomes']={}
                    for arm in ARMS:
                        source=local[key] if arm=='local3' else original
                        o=source['outcomes'][arm]
                        correct=right[np.array(o['top'])+8*block].mean(0)
                        q['outcomes'][arm]={**o,'detail':float(correct.mean()),'F':float(correct[0]),'G':float(correct[1])}
                    rows.append(q)
        def summary(group):
            # MRR is unchanged; net from D/E is deliberately omitted.
            result=summarize(group,ARMS,('ordinary','old','old3'))
            return dict(images=result['images'],pairs=result['pairs'],
                        levels={a:z['detail'] for a,z in result['levels'].items()},
                        deltas={ref:{a:z['detail'] for a,z in vals.items()} for ref,vals in result['against'].items()})
        output[tag]=summary(rows);all_rows+=rows
    output['pooled']=summary(all_rows)
    (ROOT/'results/补充措辞.json').write_text(json.dumps(output,indent=2)+'\n')

if __name__=='__main__':
    torch.set_num_threads(1)
    main()

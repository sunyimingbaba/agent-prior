import hashlib
import io
import json
import tarfile
from collections import Counter
import numpy as np
import torch
from study import STAGE, PREV, FORM, RFPO, read, dump, load, sha
from dual_scores import features


def main():
    stage=STAGE/'reference';index=read(stage/'archives/index.json')
    assert sha(stage/'archives/execution_logs.tar.gz')==index['logs']
    prim=read(RFPO/'inputs/panel.json');off=read(FORM/'inputs/ucm.json')
    repeat=read(STAGE/'inputs/repeat.json');held=read(STAGE/'inputs/holdout.json')
    rows={'primary':prim,'official':off,'hold_scene':held,'hold_first':held,'repeat_scene':repeat,'repeat_first':repeat}
    prior={(r['panel'],r['seed'],r['image']):r for r in read(STAGE/'results/caption_outcomes.json') if r['arm']=='retained'}
    fresh={(r['panel'],r['seed'],r['image']):r for r in read(stage/'caption_outcomes.json')}
    assert prior.keys()==fresh.keys()
    form_old={(r['tag'],r['seed'],r['image']):r['values'] for r in read(STAGE/'results/formal_images.json') if r['arm']=='retained'}
    form_new={(r['tag'],r['seed'],r['image']):r['values'] for r in read(stage/'formal_images.json')}
    assert form_old.keys()==form_new.keys()
    out={}
    for seed in 'ab':
        cp=index['groups'][seed+'_retained']
        assert sha(stage/'archives'/(seed+'_retained.tar.gz'))==cp['sha']
        expected=read(PREV/'strict_rfpo/inputs/lock.json')['starts'][seed]['R2']
        dest=stage/'evaluate'/(seed+'_retained');seen=set();mats={};manifests={};idxmap={}
        out[seed]={}
        for tag,roster in rows.items():
            a=features(tag,'retained',seed,roster);b=features(tag,'retained',seed,roster,True)
            mats[tag]=b;idxmap[tag]={r['image']:i for i,r in enumerate(roster)}
            diff=(a-b).double()
            linked=[k for k in prior if k[:2]==(tag,seed)]
            c=sum(sum(v!=w for v,w in zip(prior[k]['CDE'],fresh[k]['CDE'])) for k in linked)
            p=sum(sum(v!=w for v,w in zip(prior[k]['mrr_by_caption'],fresh[k]['mrr_by_caption'])) for k in linked)
            out[seed][tag]=dict(images=len(roster),exact_vectors=int(torch.all(diff==0,dim=1).sum()),max_coord=float(diff.abs().max()),
                mean_L2=float(diff.norm(dim=1).mean()),decision_images=len(linked),changed_detail=c,changed_caption_ranks=p)
            if tag in ('primary','official'):
                out[seed][tag]['changed_formal_image_readouts']=sum(form_old[tag,seed,r['image']]!=form_new[tag,seed,r['image']] for r in roster)
        with tarfile.open(stage/'archives'/(seed+'_retained.tar.gz'),'r|gz') as tf:
            for entry in tf:
                tag,key=entry.name.split('/');buf=tf.extractfile(entry).read()
                if key=='complete.json':
                    manifests[tag]=json.loads(buf);assert manifests[tag]==read(dest/tag/key);continue
                z=torch.load(io.BytesIO(buf),weights_only=True);name=z['image'];i=idxmap[tag][name];r=rows[tag][i]
                assert z['adapter']==expected and z['rgb']==r['rgb_sha256'] and z['query']==r['scene_query' if tag.endswith('scene') or tag=='primary' else 'orig_query']
                assert hashlib.sha256(buf).hexdigest()==manifests[tag]['images'][name] and torch.equal(z['v'],mats[tag][i])
                assert (tag,name) not in seen;seen.add((tag,name))
        assert len(seen)==sum(len(l) for l in rows.values())
    dump(stage/'baseline_check.json',out)


if __name__=='__main__':
    main()

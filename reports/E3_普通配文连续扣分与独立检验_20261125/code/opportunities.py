"""Separate availability of a jointly improving candidate from the frozen reward's ability to identify one."""
import sys
import numpy as np
from compare import OUT, SOURCE, read, dump, boot
from single_scores import FORM, RFPO

TAGS=('fresh','extension','saved','extra')

def evaluate(condition):
    m=read(OUT/'inputs/multi_nomination.json')['method']
    data={}
    for tag in TAGS:
        if condition=='first_caption': name='real_'+tag
        elif tag in ('fresh','extension'): name='multi_'+tag
        else: name='covered_'+tag+'_images'
        data[tag]=read(OUT/'records'/(name+'.json'))
    out={}
    for group,tags in [('all',TAGS)]+[(t,(t,)) for t in TAGS]:
        rows=[r for t in tags for r in data[t]]
        labels=[r['tag']+'|'+r['pair'] for r in rows]
        opportunity=np.array([r['means']['opportunity'] for r in rows])
        z={a:np.array([r['means'][a][0] for r in rows]) for a in ('old','R2',m,'random')}
        assert all(np.all(q<=opportunity+1e-12) for q in z.values())
        out[group]=dict(images=len(rows)//2,availability=boot(opportunity,np.ones(len(rows)),labels),
            hit_if_available={a:boot(v,opportunity,labels) for a,v in z.items()},
            new_minus_old_if_available=boot(z[m]-z['old'],opportunity,labels))
    return out

def main():
    from collections import Counter
    rows=read(OUT/'inputs/real_roster.json')
    sets={t:{r['rgb_sha256'] for r in rows if r['tag']==t} for t in TAGS}
    panels={tag:{r['rgb_sha256'] for r in read(SOURCE/'inputs'/(tag+'.json'))} for tag in ('first','second','third')}
    train={r['rgb'] for r in read(SOURCE/'inputs/training_roster.json')['rows']}
    h={r['rgb_sha256'] for r in read(RFPO/'inputs/panel.json')}
    official={r['rgb_sha256'] for r in read(FORM/'inputs/ucm.json')}
    assert sum(map(len,sets.values()))==len(set.union(*sets.values()))==282
    for v in sets.values():
        assert all(v.isdisjoint(s) for s in (*panels.values(),train,h,official))
    assert train.isdisjoint(h|official)
    overlap=dict(candidate_photos=282,raw_development_photos=sum(map(len,panels.values())),rfpo_train=len(train),held_detail=len(h),official=len(official),
        held_detail_also_official=len(h&official),train_to_evaluation=0,candidates_to_rfpo_train=0,candidates_to_evaluation=0,
        candidate_splits=dict(Counter(r['split'] for r in rows)))
    dump(OUT/'results/photo_split.json',overlap)
    for condition in sys.argv[1:]:
        assert condition in ('scene','first_caption')
        dump(OUT/'results'/('opportunity_'+condition+'.json'),evaluate(condition))

if __name__=='__main__':main()

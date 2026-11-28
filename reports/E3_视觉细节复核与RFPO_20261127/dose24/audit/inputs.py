"""Independently verify the training roster and both retrieval banks used on 98 images."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json

import torch

DOSE = Path(__file__).resolve().parents[1]
STAGE = DOSE.parent
REPORTS = STAGE.parent
FORMAL = REPORTS / 'E3-2_UCM与RSITMD独立复验_20261124'
FINE = REPORTS / 'E3_UCM多描述奖励同预算RFPO_20261124'
HISTORY = REPORTS / 'E3_双收益回答对的偏好训练_20261127/validation98/fresh98/inputs'
WORDINGS = REPORTS / 'E3_R3改选收益校准与双批首句验证_20261126/inputs'
PANEL = DOSE / 'extra98'

def read(p):
    return json.loads(p.read_bytes())

def load(p):
    return torch.load(p, map_location='cpu', weights_only=True)

def digest(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def output(name, obj):
    (DOSE/'audit'/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')

def pixels(rows):
    ids = {row['rgb_sha256'] for row in rows}
    assert len(ids)==len(rows)
    return ids

def main():
    train=read(DOSE/'inputs/roster.json')['rows']
    assert train==sorted(train,key=lambda r:r['rgb_sha256'])
    original_train=read(STAGE/'training_signal/third/inputs/roster.json')
    assert [(r['image'],r['rgb_sha256']) for r in train]==[(r['image'],r['rgb_sha256']) for r in original_train]
    groups=read(STAGE/'training_signal/third/inputs/groups.json')
    group_ids=defaultdict(list)
    for g in groups:
        group_ids[g['pair']].append((g['selected'],g['side'],g['image']))
    assert len(group_ids)==12
    assert {g['image'] for g in groups}=={r['image'] for r in train}
    assert all(len(z)==2 and {u[0] for u in z}=={True,False} and {u[1] for u in z}=={0,1} for z in group_ids.values())

    rows=read(PANEL/'inputs/rows.json');features=load(PANEL/'inputs/texts.pt')
    official=read(FORMAL/'inputs/ucm.json');extra=read(STAGE/'extra164/inputs/rows.json')
    sets={'train24':pixels(train),'panel98':pixels(rows),'official210':pixels(official),'extra164':pixels(extra)}
    intersect={a+'_'+b:sorted(vals & sets[b]) for a,vals in sets.items() for b in sets if a<b}
    assert not any(intersect.values())
    assert list(map(len,sets.values()))==[24,98,210,164]
    archived=read(HISTORY/'rows.json')
    assert archived==[{k:r[k] for k in src} for r,src in zip(rows,archived,strict=True)]
    lock=read(PANEL/'inputs/manifest.json')
    assert all(digest(REPORTS/n)==signature for n,signature in lock['sources'].items())
    assert digest(PANEL/'inputs/rows.json')==lock['rows'] and digest(PANEL/'inputs/texts.pt')==lock['texts']
    assert digest(PANEL/'事前口径.md')==lock['protocol']
    bank=next(REPORTS.glob('E3-1_奖励一致性*/空间关系候选*/native'))
    native=load(bank/'features.pt')['features']; nidx=read(bank/'index.json')['columns']
    hbank=load(REPORTS/'E3_UCM同图偏好对与独立验收_20261124/reward_confirm/records/text.pt')
    words=read(WORDINGS/'独立措辞.json');fg=load(WORDINGS/'fresh_text.pt')
    fresh=dict(zip(fg['words'],fg['features'],strict=True)); seen_caps=set();seen_det=set();pairs=defaultdict(list)
    for r in rows:
        assert r['orig_query']==r['captions'][0]
        for cap,j in zip(r['captions'],r['panel_positive_ids'],strict=True):
            assert torch.equal(features['captions'][j],native[nidx[cap]])
            seen_caps.add(j)
        labels=hbank['maps'][r['image']]
        for wording,cols in zip('DE',(r['panel_de_ids'][:2],r['panel_de_ids'][2:])):
            assert [hbank['texts'][j] for j in labels[wording]]==r[wording]
            for j,k in zip(labels[wording],cols,strict=True):
                assert torch.equal(features['details'][k],hbank['hard'][j])
                seen_det.add(k)
        for phr,cols in zip(words[r['attribute']],(r['panel_fg_ids'][:2],r['panel_fg_ids'][2:]),strict=True):
            for side,j in zip((r['side'],1-r['side']),cols,strict=True):
                assert torch.equal(fresh[phr[side]],features['details'][j])
                seen_det.add(j)
        pairs[r['pair']].append(r)
    assert len(pairs)==49 and all(sorted(r['side'] for r in z)==[0,1] for z in pairs.values())
    assert all({r['attribute'] for r in z}=={z[0]['attribute']} and all(z[0][k]==z[1][k][::-1] for k in 'DE') for z in pairs.values())
    assert seen_caps==set(range(len(features['captions']))) and seen_det==set(range(len(features['details'])))
    output('名单像素与措辞向量核查.json',dict(panel_pairs=len(pairs),training_pairs=len(group_ids),
            sizes={a:len(v) for a,v in sets.items()},pixel_intersections=intersect,distinct_caption_tokens=len(seen_caps),
            checked_caption_assignments=sum(len(r['captions']) for r in rows),distinct_detail_slots=len(seen_det),
            queries_with_alternative=sum(len(set(r['panel_positive_ids']))>1 for r in rows),manifest=digest(PANEL/'inputs/manifest.json')))

if __name__=='__main__':
    main()

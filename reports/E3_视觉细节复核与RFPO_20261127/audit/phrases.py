"""Verify withholding and label orientation using the original encoded text rows."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import torch
ROOT=Path(__file__).resolve().parents[1]
REP=ROOT.parent
FOR=REP/'E3_困难奖励跨图跨措辞校准_20261127'
POOL=REP/'E3_回填后双收益训练图筛选_20261127'
FRESH=REP/'E3_R3改选收益校准与双批首句验证_20261126/inputs'
def read(p):return json.loads(p.read_bytes())
def load(p):return torch.load(p,weights_only=True,map_location='cpu')

def main():
    banks={
        'v122':load(REP/'E3_UCM同图偏好对与独立验收_20261124/guard/records/text.pt'),
        'v148':load(REP/'E3_UCM多数配文新图复核_20261125/records/text.pt'),
        'external':load(REP/'E3_UCM同图偏好对与独立验收_20261124/reward_confirm/records/text.pt')}
    panels={'train':read(POOL/'scoring/locks/training.json')['rows'],'dev':read(FOR/'inputs/test_panel.json'),
            'external':read(FOR/'followup_pool/inputs/rows.json'),'formal126':read(REP/'E3_UCM多描述奖励同预算RFPO_20261124/inputs/panel.json')}
    banks['formal126']=load(REP/'E3_UCM多描述奖励同预算RFPO_20261124/records/text.pt')
    fg=load(FRESH/'fresh_text.pt');fg_labels=read(FRESH/'独立措辞.json')
    fw={t:tuple(v.tolist()) for t,v in zip(fg['words'],fg['tokens'])}
    held=defaultdict(dict);cwords=defaultdict(dict)
    side_labels=defaultdict(set);by_type={};vectors=load(ROOT/'visual/inputs/c_vectors.pt')
    for label,rows in panels.items():
        attrs=set();pix=set()
        for row in rows:
            key=row['attribute'];side=row['side'];name=row['image']
            assert row['rgb_sha256'] not in pix;pix.add(row['rgb_sha256']);attrs.add(key)
            source=label if label in ['formal126','external'] else row['cohort']
            bank=banks[source];words=bank['text' if source=='formal126' else 'texts']
            for k in 'CDE':
                tokens=[tuple(bank['tokens'][words.index(t)].tolist()) for t in row[k]]
                assert tokens[0]!=tokens[1]
                dest=cwords if k=='C' and label!='formal126' else held if k in 'DE' else None
                if dest is not None:
                    dest[label].update(zip(tokens,row[k]))
                if k=='C' and label!='formal126':
                    feat=bank['hard'][[words.index(t) for t in row[k]]]
                    assert torch.equal(vectors[name],feat) and not torch.equal(feat[0],feat[1])
            if key in fg_labels:
                for pair in fg_labels[key]:
                    assert fw[pair[0]]!=fw[pair[1]]
                    held[label].update((fw[t],t) for t in pair)
            if label!='formal126':
                side_labels[key].add(tuple(tuple(row[k][::1 if side==0 else -1]) for k in 'ABCDE'))
        by_type[label]={'images':len(rows),'attributes':len(attrs),'held_unique_tokens':len(held[label])}
    duplicates=[]
    for source,x in cwords.items():
        by_type[source]['C_unique_tokens']=len(x)
        for label,y in held.items():
            duplicates.extend((source,label,x[t],y[t]) for t in sorted(x.keys()&y.keys()))
    assert all(len(v)==1 for v in side_labels.values())
    out={'panels':by_type,'C_intersections_with_all_DEFG':duplicates,'canonical_ordered_properties':len(side_labels),
         'orientation':{a:{**dict(zip('ABCDE',next(iter(v)))),'FG':fg_labels.get(a)} for a,v in sorted(side_labels.items())}}
    (ROOT/'results/C措辞与方向独立核查.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(by_type,len(duplicates),len(side_labels))
if __name__=='__main__':main()

"""Adjudicate score-blind votes, bind all scored texts, and form reciprocal count controls."""
import csv
import hashlib
import itertools
import json
import re
from functools import lru_cache
from pathlib import Path

root=Path(__file__).resolve().parents[1]
def load(p):
    return json.loads((root/'inputs'/p).read_bytes())
def tsv(path):
    with path.open(encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle,delimiter='\t'))
def save(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
images={r['image']:r for r in load('candidates.json')}
cases=[]
def add(image,cohort,dimension,template,pair,positive,negative,source,subtype):
    r=images[image]
    cases.append(dict(image=image,split='val',cohort=cohort,dimension=dimension,template=template,
                      pair=pair,positive=positive,negative=negative,generic=r['generic'],scene=r['block'],
                      source=source,subtype=subtype))
previous=load('previous_val.json')
for r in previous:
    n=r['image']
    pair=':'.join(sorted([n,r.get('matched_image',n)]))
    add(n,'previous',r['dimension'],r['wording'],pair,r['positive'],r['negative'],r['source'],'previous')

candidates=tsv(root/'inputs/extension.tsv')+tsv(root/'inputs/extra_pairs.tsv')
blind={r['template']:r for f in ('blind.tsv','extra_blind.tsv') for r in tsv(root/'review'/f)}
votes={r['template'].strip():r for r in tsv(root/'review/blind_decisions.tsv')}
assert {r['template'] for r in candidates}==set(votes)==set(blind)
decisions=[]
for r in candidates:
    key=r['template']
    vote=votes[key]
    b=blind[key]
    text1=b['text1']
    expected=r['a'] if text1==r['positive_a'] else r['b']
    assert text1 in (r['positive_a'],r['positive_b'])
    agrees=vote['text1_image'].strip().split('.')[0]==expected
    decision=vote['decision'].strip().lower()
    assert decision in ('keep','attribute_only','exclude')
    # Object orientation applies as an attribute; ambiguous pairs are rejected in both dimensions.
    keep=decision!='exclude' and agrees
    orientation=decision=='attribute_only'
    dimension='attribute' if orientation else r['dimension']
    pair=':'.join(sorted((r['a']+'.tif',r['b']+'.tif')))
    decisions.append(dict(**r,vote=decision,expected_text1=expected,observed_text1=vote['text1_image'],
                          accepted=keep,dimension_final=dimension,reason=vote['reason']))
    if keep:
        for side,other in [('a','b'),('b','a')]:
            add(r[side]+'.tif','extension',dimension,key,pair,r['positive_'+side],r['positive_'+other],
                r['source_note'],'orientation' if orientation else 'natural')
nums='zero one two three four five six seven eight nine ten'.split()
oldcounts={w:{i['image']:i for i in previous if i['wording']==w} for w in ('A','B')}
for block in sorted({r['block'] for r in images.values()}):
    names=sorted((n for n in oldcounts['A'] if images[n]['block']==block),key=lambda n:int(n.split('.')[0]))
    counts=[nums.index(oldcounts['A'][n]['positive'].split()[-1][:-1]) for n in names]
    @lru_cache(None)
    def match(ids):
        if not ids:
            return ()
        a,*rest=ids
        choices=[match(tuple(rest))]
        for b in rest:
            if counts[a]!=counts[b]:
                choices.append(((a,b),)+match(tuple(i for i in rest if i!=b)))
        return min(choices,key=lambda x:(-len(x),sum(abs(counts[i]-counts[j]) for i,j in x),x))
    for a,b in match(tuple(range(len(names)))):
        left,right=names[a],names[b]
        for w,items in oldcounts.items():
            for n,other in [(left,right),(right,left)]:
                add(n,'quantity_reversed','quantity',w,':'.join([left,right]),items[n]['positive'],items[other]['positive'],
                    'confirmed_quantity_reciprocal','reciprocal')
key=lambda r:(r['cohort'],r['template'],r['image'])
assert len({key(r) for r in cases})==len(cases)
assert all(len({r['positive'],r['negative'],r['generic']})==3 for r in cases)
save(root/'inputs/adjudication.json',decisions)
save(root/'inputs/cases.json',cases)
bound=[p for p in (root/'inputs').iterdir() if p.is_file() and p.name!='lock.json']
bound += [root/'实验口径.md',root/'code/lock.py',root/'review/blind_decisions.tsv',root/'review/blind.tsv',root/'review/extra_blind.tsv']
save(root/'inputs/lock.json',{str(p.relative_to(root)):digest(p) for p in sorted(bound)})
print(len(cases),'items;',len({r['image'] for r in cases}),'images')

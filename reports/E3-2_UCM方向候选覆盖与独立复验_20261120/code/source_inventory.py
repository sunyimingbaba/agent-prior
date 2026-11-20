"""Check that new questions are independent of prior labeled panels, including identifiable crop overlaps."""
from pathlib import Path
from shared import ROOT,read,sha,dump
REPORTS=ROOT.parent
prev=REPORTS/'E3-2_UCM双视图方位奖励_20261120'
old=REPORTS/'E3-2_UCM同起点三奖励短训_20261120'
e1=REPORTS/'E1_UCM试验_20261110/UCM验证交付_20261120'
def main():
    files=[old/'inputs/detail_proposals.json',e1/'inputs/candidates.json',e1/'inputs/previous_val.json',e1/'inputs/cases.json']
    blocked=set().union(*({r['image'] for r in read(p)} for p in files))
    val=read(prev/'inputs/validation.json')
    pool=[r for r in val if r['fresh'] and r['image'] not in blocked]
    assert pool==read(ROOT/'inputs/pool.json')
    selected=read(ROOT/'inputs/fresh.json');train=read(ROOT/'inputs/train.json')
    assert train==read(prev/'inputs/train.json')
    assert sha(ROOT/'inputs/previous_coverage.csv')==sha(prev/'results/candidate_coverage_by_source.csv')
    for col in ('image','rgb_sha256'):
        assert {r[col] for r in train}.isdisjoint(r[col] for r in selected)
    assert all(r['image'] in {v['image'] for v in pool} for r in selected)
    bindings=files+[prev/'inputs/validation.json',prev/'inputs/train.json',prev/'inputs/detail.json',prev/'eval/base_00/complete.json',
                    prev/'results/states.json',prev/'environment.json',prev/'results/candidate_coverage_by_source.csv']
    dump(ROOT/'inputs/source_receipt.json',dict(previous_commit='5106f0643b2f3c37dccc5a1fa7e5d16547f926d5',files={str(p.relative_to(REPORTS)):sha(p) for p in bindings},
        pool=len(pool),selected=len(selected),train=len(train),selected_scenes=len({r['block'] for r in selected}),old_validation_gallery=len(val),
        old_labels_overlap=0,train_hash_overlap=0,proposal_registration=read(ROOT/'review/画面重叠筛查.json')))
if __name__=='__main__':main()

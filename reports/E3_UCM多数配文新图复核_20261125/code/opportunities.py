"""Descriptive opportunity and ceiling counts using the saved, fully paired folds."""
import gzip
import json
from collections import Counter
from lock import ROOT, dump, read, sha

def main():
    src = ROOT/'records/folds.jsonl.gz'
    assert sha(src) == read(ROOT/'results/audit.json')['fold_sha256']
    rows = [json.loads(s) for s in gzip.open(src,'rt')]
    counts = Counter()
    changed_images = set()
    hit_changes = []
    for r in rows:
        if r['anchor_detail'] == 1 or r['anchor_mrr'] == 1:
            assert not r['good']
            counts['anchor_at_ceiling'] += 1
        else:
            detail = max(r['slot_detail']) > r['anchor_detail']
            ordinary = max(r['slot_mrr']) > r['anchor_mrr']
            if r['good']:
                assert detail and ordinary
                category = 'same_candidate_improves_both'
            elif detail and ordinary:
                category = 'different_candidates_improve_each'
            elif detail:
                category = 'detail_only'
            elif ordinary:
                category = 'ordinary_only'
            else:
                category = 'neither'
            counts[category] += 1
        if r['changed']:
            changed_images.add(r['image'])
        if r['baseline']['hit'] != r['vote']['hit']:
            hit_changes.append({k:r[k] for k in ('image','seed','held','anchor_detail','anchor_mrr','baseline','vote')})
    assert sum(counts.values()) == len(rows)
    dump(ROOT/'results/opportunities.json', dict(raw_fold_counts=counts, changed_winner_images=sorted(changed_images),
        changed_joint_folds=hit_changes,images_with_opportunities=sorted({r['image'] for r in rows if r['good']})))

if __name__ == '__main__':
    main()

"""Compare the locked selectors with a second implementation and the original scoring code."""
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from lock import CAP, PRIOR, REP, ROOT, dump, read, sha

sys.path.insert(0, str(CAP / 'code'))
from panels import bank, load
from run_stage import boot
from verify import rr, winners
from vote import K, one

sys.path.insert(0, str(REP / 'E3-2_UCM有效图组定向短训_20261124/code'))
from verify_screen import unpack


def indices(mask):
    return np.flatnonzero(mask).tolist()


def image_folds(row, seed, b, visual, ranks):
    lookup = b['maps'][row['image']]
    native = visual @ b['native'].T
    hard = visual @ b['hard'].T
    first = lookup['native'][0]
    positive = [first] + sorted(set(lookup['native']) - {first}, key=b['hashes'].__getitem__)
    allowed = np.ones(len(b['native']), dtype=bool)
    allowed[positive] = False
    train = native[:, allowed & b['even']]
    test = native[:, allowed & ~b['even']]
    assert train.shape[1] + test.shape[1] + len(positive) == len(allowed)
    margin = native[:, first] - train.max(axis=1)
    r2 = winners(margin[:16], ranks)
    delta = hard[:, lookup['A'][0]] - hard[:, lookup['A'][1]]
    _, dense = np.unique((margin + K * delta)[:16], return_inverse=True)
    correct = np.stack([hard[:, lookup[k][0]] > hard[:, lookup[k][1]] for k in 'CDE'], axis=1)
    detail = correct.mean(axis=1)

    records = []
    for held in positive[1:]:
        visible = [p for p in positive if p != held]
        # Compute exact rational ranks of only the visible positives on the scoring side.
        train_rr = [[rr(native[i, p], train[i]) for p in visible] for i in range(16)]
        floors = [sum(train_rr[i][j] for i in r2) / len(r2) for j in range(len(visible))]
        base_pass, vote_pass, conflicts = [], [], []
        for row_rr in train_rr:
            old = sum(row_rr[:2]) >= sum(floors[:2])
            yes = sum(v > f for v, f in zip(row_rr, floors))
            no = sum(v < f for v, f in zip(row_rr, floors))
            base_pass.append(old)
            vote_pass.append(True if 2 * yes > len(visible) else False if 2 * no > len(visible) else old)
            conflicts.append(bool(yes and no))
        target = np.array([rr(native[i, held], test[i]) for i in range(17)])
        good = (detail[:16] > detail[16]) & (target[:16] > target[16])
        rec = dict(image=row['image'], pair=row['pair'], seed=seed, held=held, visible=visible,
                   conflict=any(conflicts), conflicting_slots=indices(conflicts), good=indices(good),
                   r2=r2, anchor_mrr=float(target[16]), anchor_detail=float(detail[16]),
                   slot_mrr=target[:16].astype(float).tolist(), slot_detail=detail[:16].tolist(),
                   flipped_slots=int(np.count_nonzero(np.asarray(base_pass) != vote_pass)))
        for method, passed in [('baseline', base_pass), ('vote', vote_pass)]:
            win = winners(dense + 17 * np.asarray(passed), ranks)
            hit = float(np.mean(good[win]))
            blocked = bool(np.any(good) and any(passed) and not (good & passed).any())
            rec[method] = dict(passed=indices(passed), winners=win, detail=float(np.mean(detail[win])),
                               mrr=float(target[win].mean()), hit=hit,
                               blocked_miss=(1-hit) * blocked, other_miss=(1-hit) * (bool(good.any()) and not blocked))
        rec['changed'] = rec['baseline']['winners'] != rec['vote']['winners']
        records.append(rec)

    # The previously locked, independently implemented vote rule recomputes the same endpoints.
    published, _ = one(row, b, visual, ranks, 'CDE')
    assert abs(published['opportunity'] - np.mean([bool(r['good']) for r in records])) < 1e-12
    for method in ('baseline', 'vote'):
        for k, old in [('detail', 'detail'), ('mrr', 'mrr'), ('hit', 'joint')]:
            assert abs(np.mean([r[method][k] for r in records]) - published[method][old]) < 1e-12
    return records, correct[16].tolist()


def score_panel():
    torch.set_num_threads(2)
    lock = read(ROOT/'inputs/lock.json')
    panel = read(ROOT/'inputs/panel.json')
    assert sha(ROOT/'实验口径.md') == lock['protocol'] and sha(ROOT/'inputs/panel.json') == lock['panel']
    assert sha(CAP/'code/vote.py') == lock['vote']
    b = bank(ROOT)
    folder = ROOT/'records'
    tools = read(folder/'tools.json')
    assert tools == read(PRIOR/'confirm/tools.json')
    for prefix in ('', 'extra_'):
        info = read(folder/(prefix+'provenance.json'))
        assert info['tools'] == sha(folder/'tools.json') and info['text'] == sha(folder/'text.pt')
    with (PRIOR/'results/blind.csv').open(encoding='utf-8-sig') as f:
        previous = {(r['tag'], r['image'], r['metric']): float(r['score']) for r in csv.DictReader(f)}
    all_rows = []
    excluded = []
    counts = {}
    for seed in 'ab':
        a, extra = [unpack(folder, tag) for tag in (seed, 'extra_'+seed)]
        anchor_path = PRIOR/'confirm'/(seed+'_start_blind.pt')
        assert sha(anchor_path) == lock['anchors'][seed]
        anchor = load(anchor_path)
        assert anchor['images'] == [r['image'] for r in panel]
        total_post = 0
        for i, row in enumerate(panel):
            image = row['image']
            s, vs = a[image]
            e, ve = extra[image]
            assert s['rgb'] == e['rgb'] == row['rgb_sha256'] and s['adapter'] == e['adapter'] == anchor['adapter']
            assert s['query'] == e['query'] == row['scene_query'] and s['first'] == e['first']
            assert s['tool_calls'] == e['tool_calls'] and s['post_tool'] == e['post_tool']
            assert s['seed'] == 1141000 + row['row'] and e['seed'] == 2242000 + row['row']
            total_post += s['post_tool']
            visual = torch.cat([vs['visual'], ve['visual'], anchor['features'][i:i+1]]).double().numpy()
            ranks = np.array(vs['proxy_rank'] + ve['proxy_rank'])
            assert visual.shape == (17,768) and len(ranks) == 16 and np.isfinite(visual).all()
            assert np.allclose(np.linalg.norm(visual, axis=1), 1, atol=1e-4)
            if not s['post_tool']:
                assert np.all(visual[:16] == visual[0])
            if len(set(b['maps'][image]['native'])) < 3:
                if seed == 'a':
                    excluded.append(image)
                continue
            folds, anchor_correct = image_folds(row, seed, b, visual, ranks)
            for k, c in zip('CDE', anchor_correct):
                assert c == previous[seed+'_start', image, k]
            all_rows.extend(folds)
        counts[seed] = dict(images=len(panel), slots=16*len(panel), after_tool=total_post)
    return all_rows, dict(generation=counts, excluded=excluded, candidates=32*len(panel), included_images=len(panel)-len(excluded))


def summarize(rows, mask, denominators):
    chosen = [r for r in rows if mask(r)]
    if not chosen:
        return dict(images=0, folds=0)
    labels = [r['pair'] for r in rows]
    weight = np.array([float(mask(r)) / denominators[(r['image'], r['seed'])] for r in rows])
    opportunity = np.array([bool(r['good']) for r in rows])
    def calc(numerator, denominator):
        z = boot(weight * numerator, weight * denominator, labels)
        return {k:z[k] for k in ('mean','interval','defined_draws')}
    out = dict(images=len({r['image'] for r in chosen}), pairs=len({r['pair'] for r in chosen}),
               folds=len(chosen), opportunity_folds=sum(bool(r['good']) for r in chosen),
               changes=sum(r['changed'] for r in chosen), changed_slots=sum(r['flipped_slots'] for r in chosen),
               unique_opportunity_images=len({r['image'] for r in chosen if r['good']}),
               opportunity=calc(opportunity, 1),
               random_hit_given_opportunity=calc(np.array([len(r['good']) / 16 for r in rows]), opportunity))
    for metric in ('hit','mrr','detail','blocked_miss','other_miss'):
        den = opportunity if metric in ('hit','blocked_miss','other_miss') else 1
        a, b = [np.array([r[method][metric] for r in rows]) for method in ('baseline','vote')]
        out[metric] = dict(baseline=calc(a, den), vote=calc(b, den), change=calc(b-a, den))
    return out


def main():
    rows, audit = score_panel()
    denominators = Counter((r['image'],r['seed']) for r in rows)
    bad = {r['image'] for r in rows if r['conflict']}
    pairs = defaultdict(set)
    for row in rows:
        pairs[row['pair']].add(row['image'])
    groups = {'all': lambda r: True, 'a': lambda r: r['seed']=='a', 'b': lambda r: r['seed']=='b',
              'disagree': lambda r: r['conflict'], 'agree': lambda r: not r['conflict'],
              'ever_disagree_image': lambda r: r['image'] in bad, 'never_disagree_image': lambda r: r['image'] not in bad,
              'complete_pairs_only': lambda r: len(pairs[r['pair']])==2}
    result = {name:summarize(rows, pick, denominators) for name,pick in groups.items()}
    assert len({(r['image'],r['seed'],r['held']) for r in rows}) == len(rows)
    all_result = result['all']
    for m in ('baseline','vote'):
        assert abs(sum(all_result[k][m]['mean'] for k in ('hit','blocked_miss','other_miss')) - 1) < 1e-12
    data = ''.join(json.dumps(r, separators=(',',':'))+'\n' for r in rows).encode()
    (ROOT/'records/folds.jsonl.gz').write_bytes(gzip.compress(data, mtime=0))
    audit['folds'] = len(rows)
    audit['locked_vote_checked_image_seeds'] = len(denominators)
    audit['historical_anchor_detail_checks'] = 3 * len(denominators)
    audit['fold_sha256'] = sha(ROOT/'records/folds.jsonl.gz')
    audit['eligible_for_training'] = bool(all_result['hit']['change']['interval'][0] > 0
        and all_result['detail']['change']['mean'] > 0 and all_result['mrr']['change']['mean'] >= 0)
    dump(ROOT/'results/summary.json', result)
    dump(ROOT/'results/audit.json', audit)

if __name__ == '__main__':
    main()

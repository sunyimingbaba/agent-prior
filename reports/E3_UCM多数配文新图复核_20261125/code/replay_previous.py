"""Replay the historical 142-image panel and check all primary point estimates and intervals."""
from collections import Counter
from evaluate_new import ROOT, CAP, dump, image_folds, read, summarize
from panels import transfer_panel

def main():
    rows = []
    for c, seed, _, bank, visual, ranks, *_ in transfer_panel():
        rows.extend(image_folds(c, seed, bank, visual, ranks)[0])
    scores = summarize(rows, lambda _: True, Counter((r['image'], r['seed']) for r in rows))
    expected = read(CAP/'results/vote.json')['transfer']['metrics']
    diffs = []
    for metric in ('hit','detail','mrr'):
        for arm in ('baseline','vote','change'):
            reference = expected['joint_if_available' if metric=='hit' else metric][arm]
            observed = scores[metric][arm]
            diffs.append(abs(reference['mean']-observed['mean']))
            diffs.extend(abs(a-b) for a,b in zip(reference['interval'],observed['interval']))
    assert max(diffs) < 1e-12
    dump(ROOT/'results/reference_check.json', dict(images=scores['images'], seeds=2, folds=len(rows),
        point_and_interval_checks=len(diffs), maximum_absolute_difference=max(diffs)))

if __name__ == '__main__':
    main()

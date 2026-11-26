"""Evaluate fixed R3 selectors on matched candidate trajectories."""
import collections
from fractions import Fraction
import gzip
import json
from pathlib import Path
from functools import lru_cache
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PRIOR_RUN = ROOT.parent / 'E3_双输入同预算RFPO_20261126'
sys.path.insert(0, str(PRIOR_RUN / 'code'))
from study import OLD, FORM, RFPO, banks, digest, feature_path, load, read, rr, sha
from strict_reward import strict_rewards
from panels import PRIOR, transfer_panel

ARMS = ('ordinary', 'old_joint', 'wording', 'visual_pair')
METRICS = ('detail', 'mrr', 'net')
VIEWS = ('first', 'scene')


def previous_banks():
    result = []
    for pack in banks():
        odd = np.array([int(digest(','.join(map(str, token)))[-1], 16) % 2 == 1
                        for token in pack['native_tokens'].tolist()])
        result.append(dict(native=pack['native'].double().numpy(), hard=pack['hard'].double().numpy(),
                           maps=pack['maps'], even=~odd))
    return result


@lru_cache(None)
def saved_manifest(directory):
    return {name: checksum for j in range(2)
            for name, checksum in read(directory/('complete_'+str(j)+'.json'))['files'].items()}


def source_feature(row, seed, view):
    path = feature_path(row, seed, view)
    source = load(path)
    assert source['rgb'] == row['rgb_sha256']
    if 'adapter' not in source:
        assert row['tag'] == 'saved' and view == 'scene'
        assert sha(path) == saved_manifest(path.parent)[row['image']]
    return source


def previous_panel(rows, text_banks):
    for row in rows:
        bank = text_banks[int(row['tag'] in ('saved', 'extra'))]
        for view in VIEWS:
            for seed in 'ab':
                source = source_feature(row, seed, view)
                yield row, seed, view, bank, source['visual'].double().numpy()


def pair_directions(rows):
    pairs = collections.defaultdict(list)
    for row in rows:
        pairs[row['pair']].append(row)
    directions = collections.defaultdict(list)
    for name, (left, right) in pairs.items():
        assert left['A'] == right['A'][::-1]
        for view in VIEWS:
            for seed in 'ab':
                pos, neg = [source_feature(row, seed, view) for row in (left, right)]
                if 'adapter' in pos and 'adapter' in neg:
                    assert pos['adapter'] == neg['adapter']
                delta = (pos['visual'][-1].double() - neg['visual'][-1].double()).numpy()
                directions[view, seed, tuple(left['A'])].append((name, delta))
                directions[view, seed, tuple(right['A'])].append((name, -delta))
    return directions


def grid(row, bank, visual):
    assert visual.shape == (17, 768)
    ix = bank['maps'][row['image']]
    pos = ix['native']
    alt = sorted(set(pos) - {pos[0]})
    sim = visual @ bank['native'].T
    ranks = []
    midranks = None
    for mask in (~bank['even'], bank['even']):
        neg = mask.copy()
        neg[pos] = False
        ranks.append([[rr(v[j], v[neg]) for j in alt] for v in sim])
        if midranks is None:
            midranks = np.array([[1 + np.sum(v[neg] > v[j]) + 0.5 * np.sum(v[neg] == v[j])
                                  for j in alt] for v in sim])
    lut = np.array([ix[k] for k in 'ABCDE'])
    hard_scores = visual @ bank['hard'][lut.flatten()].T
    margins = hard_scores[:, ::2] - hard_scores[:, 1::2]
    assert np.isfinite(margins).all()
    return ranks, midranks, margins


def best(keys, eligible):
    if len(eligible) == 0:
        return []
    top = max(keys[i] for i in eligible)
    return [int(i) for i in eligible if keys[i] == top]


def select(fit, rank, hard, direction_score, cols):
    means = [sum(row[j] for j in cols) / len(cols) for row in fit]
    ordinary = best(means, range(8))
    packet = dict(rr=[[[row[j].numerator, row[j].denominator] for j in cols] for row in fit],
                  detail=(hard[:, :2] > 0).sum(1).tolist())
    rewards, _ = strict_rewards(packet)
    assert ordinary == np.flatnonzero(rewards['ordinary'] == rewards['ordinary'].max()).tolist()
    old = np.flatnonzero(rewards['joint'] == rewards['joint'].max()).tolist()
    mid = rank[:8, cols]
    anchor_rank = mid[ordinary].mean(axis=0)
    allowed = (mid <= anchor_rank + 1).all(axis=1)
    phrasing = hard[:8, :2]
    both_correct = (phrasing > 0).all(axis=1)
    # Midranks measure how many candidates each wording prefers to this one.
    diffs = phrasing[:, None, :] - phrasing[None, :, :]
    percentiles = ((diffs > 0) + 0.5 * (diffs == 0)).mean(axis=1)
    gain = (percentiles - percentiles[ordinary].mean(axis=0)).min(axis=1)
    bigger = (phrasing > phrasing[ordinary].mean(axis=0)).all(axis=1)
    eligible = np.flatnonzero(allowed & both_correct & bigger)
    wording = best(list(zip(gain, means[:8])), eligible)
    pair = []
    if direction_score is not None:
        visual_gain = direction_score[:8] - direction_score[ordinary].mean()
        pool = np.flatnonzero(allowed & both_correct & (visual_gain > 0))
        pair = best(list(zip(visual_gain, means[:8])), pool)
    return dict(ordinary=ordinary, old_joint=old, wording=wording or ordinary, visual_pair=pair or ordinary)


def evaluate(row, seed, view, bank, visual, directions):
    (fit, check), midrank, margins = grid(row, bank, visual)
    captions = len(fit[0])
    if captions < 2:
        return []
    support = [v for name, v in directions.get((view, seed, tuple(row['A'])), [])
               if name != row['pair']]
    projection = visual @ np.mean(support, axis=0) if len(support) >= 2 else None
    records = []
    for block in (0, 1):
        ids = list(range(8 * block, 8 * block + 8)) + [16]
        fine = (margins[ids[:8], 2:] > 0).mean(1)
        for held in range(captions):
            cols = [j for j in range(captions) if j != held]
            selected = select([fit[i] for i in ids], midrank[ids], margins[ids],
                              projection[ids] if projection is not None else None, cols)
            held_rr = np.array([float(check[i][held]) for i in ids[:8]])
            baseline = selected['ordinary']
            oracle = bool(((fine > fine[baseline].mean() + 1e-12)
                           & (held_rr > held_rr[baseline].mean() + 1e-12)).any())
            outcomes = {}
            for arm, winners in selected.items():
                changed = winners != baseline
                df = fine[winners, None] - fine[baseline]
                dr = held_rr[winners, None] - held_rr[baseline]
                win = float(((df > 0) & (dr > 0)).mean()) if changed else 0.
                loss = float(((df < 0) & (dr < 0)).mean()) if changed else 0.
                outcomes[arm] = dict(top=winners, detail=float(fine[winners].mean()),
                                     mrr=float(held_rr[winners].mean()), net=win-loss,
                                     win=win, loss=loss, changed=changed)
            records.append(dict(image=row['image'], rgb=row['rgb_sha256'], pair=row['pair'], seed=seed,
                                view=view, block=block, held=held, support=len(support),
                                oracle=oracle, outcomes=outcomes))
    return records


def summarize(records, arms=ARMS, references=('ordinary', 'old_joint')):
    grouped = collections.defaultdict(list)
    for row in records:
        grouped[row['seed'], row['image']].append(row)
    names = sorted(n for s, n in grouped if s == 'a')
    assert names == sorted(n for s, n in grouped if s == 'b')
    pairs = [grouped['a', n][0]['pair'] for n in names]
    unique, index = np.unique(pairs, return_inverse=True)
    rng = np.random.default_rng(261126)
    draws = rng.multinomial(len(unique), np.ones(len(unique))/len(unique), size=20000)
    denom = draws @ np.bincount(index)
    def summary(v):
        by_pair = np.bincount(index, weights=v.mean(0), minlength=len(unique))
        ci = np.quantile((draws @ by_pair)/denom, [.025, .975])
        return dict(mean=float(v.mean()), seeds=v.mean(1).tolist(), interval=ci.tolist())
    levels = {}
    totals = {}
    for arm in arms:
        levels[arm] = np.array([[[np.mean([r['outcomes'][arm][k] for r in grouped[s, n]])
                                  for n in names] for s in 'ab'] for k in METRICS])
        outcomes = [r['outcomes'][arm] for r in records]
        totals[arm] = dict(changes=sum(o['changed'] for o in outcomes),
                           wins=sum(o['win'] for o in outcomes), losses=sum(o['loss'] for o in outcomes))
    comparators = {}
    for ref in references:
        comparators[ref] = {arm:{k:summary(d) for k,d in zip(METRICS, levels[arm]-levels[ref])}
                            for arm in arms if arm != ref}
    return dict(images=len(names), pairs=len(unique), folds=len(records), oracle=sum(r['oracle'] for r in records),
                support_images=sum(grouped['a', n][0]['support'] >= 2 for n in names), totals=totals,
                levels={arm:dict(zip(METRICS, v.mean(axis=(1,2)).tolist())) for arm,v in levels.items()},
                against=comparators)


def process(stream, directions, output):
    groups = collections.defaultdict(list)
    counts = collections.defaultdict(set)
    with gzip.open(ROOT/(output+'.jsonl.gz'), 'wt', encoding='utf-8') as f:
        for row, seed, view, bank, visual in stream:
            counts[view].add(row['image'])
            for rec in evaluate(row, seed, view, bank, visual, directions):
                f.write(json.dumps(rec) + '\n')
                groups[view].append(rec)
    result = {view:summarize(records) for view,records in groups.items()}
    for view, summary in result.items():
        summary['input_images'] = len(counts[view])
    return result


def main():
    torch.set_num_threads(1)
    rows = read(OLD/'inputs/real_roster.json')
    held_keys = {r['rgb_sha256'] for r in read(PRIOR_RUN/'inputs/holdout.json')}
    dev = [r for r in rows if r['rgb_sha256'] not in held_keys]
    held = [r for r in rows if r['rgb_sha256'] in held_keys]
    assert len(dev)==204 and len(held)==78 and not {r['pair'] for r in dev} & {r['pair'] for r in held}
    proto = pair_directions(dev)
    text_banks = previous_banks()
    result = {'development':process(previous_panel(dev, text_banks), proto, 'development')}
    arms = result['development']['first']['against']['ordinary']
    feasible = [arm for arm in ('wording','visual_pair') if all(arms[arm][m]['mean']>0 for m in METRICS)]
    nominated = max(feasible, key=lambda arm:arms[arm]['detail']['mean']) if feasible else None
    result['nomination'] = nominated
    (ROOT/'开发提名.json').write_text(json.dumps({'nomination':nominated,'summary':arms},indent=2)+'\n')
    result['hold78'] = process(previous_panel(held, text_banks), proto, 'hold78')
    skip = {r['rgb_sha256'] for r in rows + read(FORM/'inputs/ucm.json') + read(RFPO/'inputs/panel.json')}
    groups = collections.defaultdict(list)
    for row in read(PRIOR/'reward_confirm/inputs/panel.json'):
        groups[row['pair']].append(row)
    extra = [r for group in groups.values() if all(r['rgb_sha256'] not in skip for r in group) for r in group]
    keys = {r['rgb_sha256'] for r in extra}
    assert len(keys)==98 and not {r['pair'] for r in rows} & {r['pair'] for r in extra}
    stream = ((r,s,v,b,x) for r,s,v,b,x,*_ in transfer_panel() if r['rgb_sha256'] in keys)
    result['extra98'] = process(stream, proto, 'extra98')
    # Match the previous implementation's image-weighted ordinary and strict reward.
    prev = read(PRIOR_RUN/'results/candidate_transfer.json')
    extra_prev = read(PRIOR_RUN/'results/independent_ties.json')
    for tag,view,old in [('hold78',v,prev['hold_'+v]['leave_one']) for v in VIEWS]+[('extra98','scene',extra_prev['leave_one'])]:
        block = result[tag][view]
        assert block['folds']==old['groups'] and block['images']==old['images']
        for now,oldarm in [('ordinary','ordinary'),('old_joint','joint')]:
            for k,oldkey in [('detail','CDE'),('mrr','mrr')]:
                assert abs(block['levels'][now][k]-old['levels'][oldarm][oldkey]['mean'])<1e-10
    def passed(panel):
        delta = panel['against']['ordinary'][nominated]
        return min(delta['detail']['seeds'])>0 and delta['detail']['interval'][0]>0 and all(delta[m]['mean']>=0 for m in ('mrr','net'))
    result['train'] = nominated is not None and passed(result['hold78']['first']) and passed(result['extra98']['scene'])
    (ROOT/'结果.json').write_text(json.dumps(result, ensure_ascii=False,indent=2)+'\n')
    print('nominated', nominated, 'train', result['train'])


if __name__ == '__main__':
    main()

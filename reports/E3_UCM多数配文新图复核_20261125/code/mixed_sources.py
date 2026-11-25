"""Fixed-budget mixed sampling on the two earlier development panels, with original R3."""
from collections import defaultdict

import numpy as np
from evaluate_new import CAP, ROOT, boot, dump, one, read
from panels import question_panels


def pool_metrics(c, bank, visual, ranks):
    assert visual.shape == (17, 768) and ranks.shape == (16,)
    scores, _ = one(c, bank, visual, ranks, 'BC')
    return dict(opportunity=scores['opportunity'], **scores['baseline'])


def collect(tag):
    rows = []
    batches = defaultdict(dict)
    for c, seed, mode, bank, visual, ranks, *_ in question_panels(tag):
        batches[c['image'],seed][mode] = (c, bank, visual, ranks)
    assert len(batches) == 108
    for (_, seed), modes in batches.items():
        c, bank, scene, rs = modes['scene']
        q, qb, question, rq = modes['question']
        assert c == q and bank is qb and np.array_equal(scene[-1], question[-1])
        values = dict(scene=pool_metrics(c, bank, scene, rs), question=pool_metrics(c, bank, question, rq))
        allocations = []
        for first, second in [(slice(0,8),slice(8,16)), (slice(8,16),slice(0,8))]:
            v = np.vstack((scene[first], question[second], scene[-1:]))
            r = np.r_[rs[first], rq[second]]
            allocations.append(pool_metrics(c, bank, v, r))
        values['mixed'] = {k:np.mean([m[k] for m in allocations]) for k in allocations[0]}
        rows.append(dict(tag=tag,image=c['image'],pair=c['pair'],attribute=c['attribute'],seed=seed,
                         allocations=allocations,**values))
    return rows


def summarize(rows):
    labels = [r['attribute'] for r in rows]
    assert len(set(labels)) == 27
    def estimate(x,y):
        z = boot(x, y, labels)
        return dict(mean=z['mean'],interval=z['interval'])
    metrics = {}
    for k in ('opportunity','joint','joint_if_available','mrr','detail'):
        field = 'joint' if k=='joint_if_available' else k
        vals, nums, dens = {}, {}, {}
        for arm in ('scene','question','mixed'):
            nums[arm] = np.array([r[arm][field] for r in rows])
            dens[arm] = np.array([r[arm]['opportunity'] if k=='joint_if_available' else 1 for r in rows])
            vals[arm] = estimate(nums[arm], dens[arm])
        for control in ('scene','question'):
            if k != 'joint_if_available':
                vals['mixed_minus_'+control] = estimate(nums['mixed'] - nums[control], dens[control])
            else:
                # Each arm has its own opportunity denominator. Use the same attribute draws for both.
                groups = list(dict.fromkeys(labels))
                ix = [groups.index(g) for g in labels]
                draw = np.random.default_rng(20261125).integers(len(groups),size=(20000,len(groups)))
                levels = []
                for arm in ('mixed',control):
                    n = np.bincount(ix, weights=nums[arm])[draw].sum(1)
                    d = np.bincount(ix, weights=dens[arm])[draw].sum(1)
                    levels.append(np.divide(n,d,out=np.full_like(n,np.nan),where=d>0))
                diff = levels[0] - levels[1]
                vals['mixed_minus_'+control] = dict(mean=vals['mixed']['mean']-vals[control]['mean'],
                    interval=np.nanquantile(diff,[.025,.975]).tolist())
        metrics[k] = vals
    return metrics


def main():
    rows = collect('development') + collect('repeat')
    assert len({r['image'] for r in rows}) == 108
    summary = {}
    for tag in ('development','repeat','combined'):
        chosen = rows if tag=='combined' else [r for r in rows if r['tag']==tag]
        summary[tag] = summarize(chosen)
        if tag != 'combined':
            old = read(CAP/'results'/(tag+'.json'))['levels']['hash_two']
            for metric,scores in summary[tag].items():
                assert abs(scores['question']['mean'] - old[metric]['mean']) < 1e-12
    dump(ROOT/'results/mixed_sources.json',summary)
    dump(ROOT/'records/mixed_sources_images.json',rows)

if __name__ == '__main__':
    main()

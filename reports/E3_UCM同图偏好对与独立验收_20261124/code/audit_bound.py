"""Verify source weights, repeated paired exposures, hinge activation and the final adapters."""
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from dev_metrics import ROOT, read, sha, dump, load

def audit(backup):
    rows = {}
    initial = read(ROOT / 'records/environment.json')['starting']
    for seed in 'ab':
        tag = seed + '_bounded'
        folder = ROOT / 'pilot' / tag
        steps = read(folder / 'steps.json')
        checkpoint = read(folder / 'complete.json')
        source = ROOT / 'records' / (seed + '_bound_source.pt')
        reference_path = ROOT / 'records' / (seed + '_bound_ref.json')
        reference = read(reference_path)
        assert sha(backup / tag / 'adapter/adapter_model.safetensors') == checkpoint['final']
        assert reference['adapter'] == checkpoint['initial'] == initial[seed] == load(source)['adapter']
        assert reference['states'] == sha(source) and sha(reference_path) == checkpoint['ref']
        assert sha(ROOT / 'code/bound_fit.py') == checkpoint['source'] and sha(ROOT / 'code/bound_reader.py') == checkpoint['reader']
        first = [r['pair'] for r in read(ROOT / 'pilot' / (seed + '_joint') / 'steps.json')]
        order = [r['pair'] for r in steps]
        assert order[:84] == first and set(first) == set(order[84:]) and len(set(first)) == 84
        assert len(steps) == checkpoint['steps'] == 168
        refs = reference['images']
        target = np.median([abs(x) for fields in refs.values() for x in fields['scene_query']['AB']])
        assert np.isclose(target, reference['target']) and len(refs) == 168
        exposures, native, detail, cosine = Counter(), [], [], []
        for j, step in enumerate(steps, 1):
            assert step['step'] == j and len(step['records']) == 4 and np.isfinite(step['norm'])
            seen = {(entry['image'], entry['field']) for entry in step['records']}
            assert len(seen) == 4 and len({name for name, _ in seen}) == 2
            exposures.update(seen)
            for v in step['records']:
                r = refs[v['image']][v['field']]
                cap = next((k for k in (1,5,10) if r['rank'] <= k), r['rank'])
                assert r['cap'] == v['cap'] == cap and (v['native'] > 0) == (v['rank'] > cap)
                assert np.isclose(v['hard'], np.maximum(0, 50*(target-np.array(v['AB']))).mean(), atol=1e-5)
                scene = v['field'] == 'scene_query'
                assert np.isclose(v['hard_weight'], 2*checkpoint['kappa'] if scene else 0)
                assert v['cos'] > .98 and np.isfinite(v['gradnorm'])
                native.append(v['native'] > 0)
                if scene: detail.append(v['hard'] > 0)
                cosine.append(v['cos'])
        assert len(exposures) == 336 and set(exposures.values()) == {2}
        rows[seed] = dict(pairs=84, passes=2, greedy_train_trajectories=len(cosine), target=float(target),
                          native_active=sum(native), native_observations=len(native), hard_active=sum(detail),
                          hard_observations=len(detail), actual_updates=sum(r['norm']>0 for r in steps), min_replay_cos=min(cosine))
    dump(ROOT / 'results/bounded_audit.json', rows)

if __name__ == '__main__':
    audit(Path(sys.argv[1]))

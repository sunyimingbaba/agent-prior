"""Freeze the archived independent image roster and recover its original A/B templates."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT.parent
OLD = REP / 'E3-1_UCM第二轮R3困难负样本复验_20261124'
PRIOR = REP / 'E3_UCM同图偏好对与独立验收_20261124'
CAP = REP / 'E3_UCM普通描述奖励门槛复核_20261125'
sys.path.insert(0, str(OLD / 'code'))
from pairs import BANK

def read(path):
    return json.loads(path.read_bytes())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def main():
    src = PRIOR / 'confirm/inputs/panel.json'
    rows = read(src)
    original = read(PRIOR / 'confirm/inputs/lock.json')
    assert original['panel'] == sha(src)
    histories = []
    for path in sorted(REP.glob('E3_UCM中性观察问题*/inputs/lock.json')):
        histories.extend((r['image'], r['rgb']) for r in read(path)['rows'])
    for name in ('guard', 'reward_confirm'):
        histories.extend((r['image'], r['rgb_sha256']) for r in read(PRIOR / name / 'inputs/panel.json'))
    for i, key in enumerate(('image', 'rgb_sha256')):
        new = {r[key] for r in rows}
        old = {r[i] for r in histories}
        assert len(new) == len(rows) == 148 and len(old) == len(histories) == 372 and old.isdisjoint(new)
    assert set(Counter(r['pair'] for r in rows).values()) == {2}
    assert len({r['pair'] for r in rows}) == 74
    for r in rows:
        assert r['split'] == 'train'
        for letter, terms in zip('ABC', BANK[r['attribute']]):
            ordered = [terms[r['side']], terms[1-r['side']]]
            if letter == 'C':
                assert ordered == r['C']
            else:
                r[letter] = ordered
    dest = ROOT / 'inputs/panel.json'
    dump(dest, rows)
    anchors = {s: sha(PRIOR / 'confirm' / (s + '_start_blind.pt')) for s in 'ab'}
    dump(ROOT / 'inputs/lock.json', dict(panel=sha(dest), protocol=sha(ROOT/'实验口径.md'),
        train_arrow=original['train_arrow'], original_panel=sha(src), previous_images=len(histories),
        pairs=74, images=len(rows), anchors=anchors, vote=sha(CAP/'code/vote.py')))

if __name__ == '__main__':
    main()

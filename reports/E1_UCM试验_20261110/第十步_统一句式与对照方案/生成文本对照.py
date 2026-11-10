#!/usr/bin/env python3
"""Create two wording variants from the user-reviewed E1 sample manifest."""
import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
with args.source.open(encoding='utf-8-sig', newline='') as f:
    source = list(csv.DictReader(f))
assert len(source) == 100 and len({r['图像编号'] for r in source}) == 100
terms = {'飞机':'airplanes', '棒球场':'baseball fields', '储罐':'storage tanks',
         '网球场':'tennis courts', '沙坑':'sand bunkers', '高尔夫沙坑':'sand bunkers', '住宅':'houses'}
words = {1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven'}
reverse = {word: number for number, word in words.items()}
pattern = re.compile(r'\b(?:one|two|three|four|five|six|seven)\b', re.IGNORECASE)
# A stable, fixed 50/50 display order for each wording; the score depends on text, not its display position.
ordered = sorted(source, key=lambda r: hashlib.sha256(('20261110:' + r['图像编号']).encode()).hexdigest())
first_correct = {r['图像编号'] for r in ordered[:50]}
results = []
for r in source:
    image = r['图像编号']
    count = int(r['用户确认数量'])
    found = pattern.findall(r['错误描述'])
    assert len(found) == 1, (image, r['错误描述'], found)
    wrong = reverse[found[0].lower()]
    assert wrong != count and count in words
    noun = terms[r['目标']]
    wording = {
        'A_原句式': lambda n: f'The number of {noun} visible in the image is {words[n]}.',
        'B_改写句式': lambda n: f'The number of visible {noun} is {words[n]}.',
    }
    for name, render in wording.items():
        pos, neg = render(count), render(wrong)
        pa, na = pos.split(), neg.split()
        assert len(pa) == len(na) and sum(a != b for a, b in zip(pa, na)) == 1
        display1, display2 = (pos, neg) if image in first_correct else (neg, pos)
        results.append({'图像编号':image, '数据来源':r['数据来源'], '目标':r['目标'],
                        '实际数量':count, '错误数量':wrong, '句式':name,
                        '正确描述':pos, '错误描述':neg, '文本1':display1, '文本2':display2,
                        '正确文本位置':1 if image in first_correct else 2,
                        '原始第一条描述':r['原始第一条描述'], '用户确认状态':r['状态']})
        if name == 'A_原句式' and r['确认来源'] == '此前17组用户确认':
            assert pos == r['正确描述'] and neg == r['错误描述'], image
assert len(results) == 200
for name in ('A_原句式', 'B_改写句式'):
    part = [r for r in results if r['句式'] == name]
    assert len(part) == 100 and len({r['图像编号'] for r in part}) == 100
    assert collections.Counter(r['正确文本位置'] for r in part) == {1:50, 2:50}
args.output_dir.mkdir(parents=True, exist_ok=True)
with (args.output_dir/'100组两种句式_对照清单.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
a = [r for r in results if r['句式'] == 'A_原句式']
summary = {'images':len(source), 'pairs_per_wording':len(a), 'total_pairs':len(results),
           'number_distribution':dict(sorted(collections.Counter(r['实际数量'] for r in a).items())),
           'quantity_pairs':{f'{x}-{y}':n for (x,y),n in sorted(collections.Counter((r['实际数量'],r['错误数量']) for r in a).items())},
           'correct_smaller':sum(r['实际数量']<r['错误数量'] for r in a),
           'correct_larger':sum(r['实际数量']>r['错误数量'] for r in a),
           'positive_position_per_wording':{'1':50,'2':50},
           'first_17_main_wording_matches_prior':True}
(args.output_dir/'生成核验.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))

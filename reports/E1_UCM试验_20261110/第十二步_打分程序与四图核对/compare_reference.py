#!/usr/bin/env python3
"""Compare new four-image scores with the actual earlier run, keyed by text role."""
import argparse
import csv
import json
import math
from pathlib import Path

EXPECTED_IMAGES = {'1983.tif', '1981.tif', '983.tif', '988.tif'}
EXPECTED_WORDINGS = {'原句式', '改写句式'}
EXPECTED_MODELS = {'CLIP', 'AGSE'}
EXPECTED_KEYS = {(m, i, w) for m in EXPECTED_MODELS for i in EXPECTED_IMAGES for w in EXPECTED_WORDINGS}

def keyed(rows, is_reference):
    out = {}
    for r in rows:
        wording = r['wording'] if is_reference else r['wording'].split('_', 1)[-1]
        key = (r['model'], r['image'], wording)
        if key in out:
            raise ValueError(f'duplicate key: {key}')
        position = r['correct_text_position']
        if position in ('文本1', '1', 1):
            positive, negative = r['text1'], r['text2']
        elif position in ('文本2', '2', 2):
            positive, negative = r['text2'], r['text1']
        else:
            raise ValueError(f'invalid position: {position}')
        out[key] = {'positive':positive, 'negative':negative,
                    'positive_score':float(r['correct_score']), 'negative_score':float(r['wrong_score'])}
    if set(out) != EXPECTED_KEYS:
        raise ValueError(f'expected {len(EXPECTED_KEYS)} specific keys, received {len(out)}; '
                         f'missing {sorted(EXPECTED_KEYS-set(out))}; extra {sorted(set(out)-EXPECTED_KEYS)}')
    return out

def compare(reference, actual, tolerance=1e-6):
    old, new = keyed(reference, True), keyed(actual, False)
    details = []
    for key in sorted(EXPECTED_KEYS):
        r, a = old[key], new[key]
        text_matches = r['positive'] == a['positive'] and r['negative'] == a['negative']
        all_scores = [r['positive_score'], r['negative_score'], a['positive_score'], a['negative_score']]
        if not all(math.isfinite(s) for s in all_scores):
            raise ValueError(f'non-finite score: {key}')
        delta_pos = abs(r['positive_score']-a['positive_score'])
        delta_neg = abs(r['negative_score']-a['negative_score'])
        old_sign = (r['positive_score'] > r['negative_score']) - (r['positive_score'] < r['negative_score'])
        new_sign = (a['positive_score'] > a['negative_score']) - (a['positive_score'] < a['negative_score'])
        details.append({'model':key[0], 'image':key[1], 'wording':key[2], 'text_matches':text_matches,
                        'positive_abs_difference':delta_pos, 'negative_abs_difference':delta_neg,
                        'ranking_matches':old_sign==new_sign,
                        'passed':text_matches and delta_pos<=tolerance and delta_neg<=tolerance and old_sign==new_sign})
    return {'passed':all(d['passed'] for d in details), 'tolerance':tolerance,
            'pair_count':len(details), 'score_count':len(details)*2,
            'max_absolute_difference':max(max(d['positive_abs_difference'], d['negative_abs_difference']) for d in details),
            'ranking_changes':sum(not d['ranking_matches'] for d in details), 'details':details}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--actual',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tolerance',type=float,default=1e-6)
    args=parser.parse_args()
    with args.reference.open(encoding='utf-8-sig',newline='') as f: reference=list(csv.DictReader(f))
    with args.actual.open(encoding='utf-8-sig',newline='') as f: actual=list(csv.DictReader(f))
    result=compare(reference,actual,args.tolerance)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='details'},ensure_ascii=False),flush=True)
    if not result['passed']:
        raise SystemExit(1)

if __name__=='__main__':
    main()

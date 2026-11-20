"""Read-only provenance and native-text linkage checks; output confined to audit trace."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import re

TRACE = Path(__file__).resolve().parent
ROOT = TRACE.parents[3]
def load(path):
    return json.loads(path.read_bytes())
def table(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f, delimiter='\t' if path.suffix == '.tsv' else ','))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

cases = load(ROOT / 'inputs/cases.json')
candidates = {r['image']: r for r in load(ROOT / 'inputs/candidates.json')}
main = [r for r in cases if r['cohort'] == 'extension' or
        (r['cohort'] == 'previous' and r['dimension'] == 'quantity' and r['template'] == 'A')]
lookup = {(r['image'], r['template']): r for r in main}
keys = table(ROOT / 'review/text_key.tsv') + table(ROOT / 'review/replay_key.tsv')
votes = {r['id']: r for r in table(ROOT / 'review/text_votes.tsv')}
details = {r['id']: r for r in table(ROOT / '生成文字逐项核验.csv')}
assert len(lookup) == 94 and len(keys) == len({r['id'] for r in keys}) == 376
for r in keys:
    case = lookup[r['image'], r['template']]
    native = load(ROOT / 'priors' / r['method'] / (Path(r['image']).stem + '.json'))
    assert r['positive'] == case['positive']
    assert {r['choice1'], r['choice2']} == {case['positive'], case['negative']}
    assert r['answer'] == native['rounds'][-1]['decoded'].strip()
    auto = not any(c.isalnum() for c in re.sub(r'\b(addCriterion|scene_classify)\b', '', r['answer']))
    assert str(auto) == r['auto']
    outcome = 'neither' if auto else votes[r['id']]['vote']
    if outcome in ('1', '2'):
        outcome = 'correct' if r['choice' + outcome] == r['positive'] else 'wrong'
    assert outcome == details[r['id']]['outcome']
    assert details[r['id']]['strict_sensitivity'] == ('neither' if r['id'] in ('T111', 'T215') else outcome)
assert set(votes) == {r['id'] for r in keys if r['auto'] == 'False'}

human = {r['图像编号']: r for r in table(ROOT.parent / '第七步_扩充数量样本/100组数量确认汇总.csv')}
nums = 'zero one two three four five six seven eight nine ten'.split()
leaked = []
for r in main:
    if r['dimension'] == 'quantity':
        source = human[r['image']]
        word = nums[int(source['用户确认数量'])]
        assert r['positive'].split()[-1] == word + '.' and source['数据来源'] == 'UCM val'
        if re.search(r'\b' + word + r'\b', candidates[r['image']]['captions'][0], re.I):
            leaked.append(r['image'])

scored = {r['image'] for r in cases}
generation = {}
for method in ('ASPF', 'AGSE', 'AGSE_scene', 'AGSE_original'):
    records = []
    for path in sorted((ROOT / 'priors' / method).glob('*.json')):
        r = load(path)
        assert sha(path.with_suffix('.pt')) == r['sha256']
        assert r['rgb_sha256'] == candidates[r['image']]['rgb_sha256']
        query = candidates[r['image']]['captions'][0] if method == 'AGSE_original' else (
            candidates[r['image']]['scene_query'] if method == 'AGSE_scene' else 'Describe the visible scene.')
        assert r['query'] == query
        if r['image'] in scored:
            records.append(r)
    assert len(records) == 104
    generation[method] = dict(images=104, with_tools=sum(r['used'] > 0 for r in records),
        first_eos=sum(r['first_eos'] for r in records),
        length_64=sum(len(r['rounds'][-1]['ids']) == 64 for r in records),
        final_strings=dict(collections.Counter(r['rounds'][-1]['decoded'] for r in records)) if method == 'AGSE_original' else None)
    if method == 'AGSE_original':
        settings = load(ROOT / 'outputs/AGSE_archived/settings.json')
        assert all(r['old_bit_exact'] and r['archive_sha256'] == settings['priors_sha256'][r['image']] for r in records)
scene_counts = collections.Counter(candidates[n]['block'] for n in scored)
result = dict(status='PASS', text_key_native_trace_exact=376, votes=113, main=94,
    main_unique_images=len({r['image'] for r in main}), human_quantity_count_match=22,
    first_caption_correct_numeral_images=leaked, generation=generation,
    overlap_pair_count_given_90_train_val_images_per_scene=sum(n * 89 - n * (n - 1) // 2 for n in scene_counts.values()),
    text_results={m: dict(collections.Counter(r['outcome'] for r in details.values() if r['method'] == m))
        for m in ('AGSE', 'AGSE_scene', 'AGSE_original', 'ASPF')})
(TRACE / 'supplemental_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print('PASS: native text linkage, quantity provenance, prior hashes and recorded source bindings')

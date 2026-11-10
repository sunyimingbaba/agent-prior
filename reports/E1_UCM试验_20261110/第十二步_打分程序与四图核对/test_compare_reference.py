"""Check that role-aware historical comparison catches changed scores and text."""
import copy
import csv
import importlib.util
from pathlib import Path

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('compare_reference', here/'compare_reference.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
reference_path = here.parent/'第六步_数量词句式对照结果'/'result.csv'
with reference_path.open(encoding='utf-8-sig',newline='') as f:
    reference = list(csv.DictReader(f))
actual = copy.deepcopy(reference)
for r in actual:
    r['wording'] = ('A_' if r['wording'] == '原句式' else 'B_') + r['wording']
    r['text1'], r['text2'] = r['text2'], r['text1']
    r['correct_text_position'] = '2' if r['correct_text_position'] == '文本1' else '1'
assert mod.compare(reference,actual)['passed']
changed = copy.deepcopy(actual)
changed[0]['correct_score'] = str(float(changed[0]['correct_score']) + 0.001)
assert not mod.compare(reference,changed)['passed']
changed = copy.deepcopy(actual)
changed[0]['text1'] += ' changed'
assert not mod.compare(reference,changed)['passed']
try:
    mod.compare(reference,actual[:-1])
    raise AssertionError('missing row not detected')
except ValueError:
    pass
print('PASS: swapped display order, changed score, changed text, and missing row')

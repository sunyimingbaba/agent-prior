"""Check input validation rejects meaningful changes before scoring."""
import copy
import csv
import importlib.util
from pathlib import Path

here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('score_wording',here/'score_wording.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
with (here.parent/'第十步_统一句式与对照方案'/'100组两种句式_对照清单.csv').open(encoding='utf-8-sig',newline='') as f:cases=list(csv.DictReader(f))
with (here.parent/'第十一步_模型与缓存核查'/'100图像与先验核查.csv').open(encoding='utf-8-sig',newline='') as f:audit=list(csv.DictReader(f))
selected, lookup=mod.validate_cases(cases,audit,'smoke')
assert len(selected)==8 and len(lookup)==100
assert len(mod.validate_cases(cases,audit,'all')[0])==200
for column in ['原始第一条描述','文本1']:
    bad=copy.deepcopy(cases);bad[0][column]+=' altered'
    try:
        mod.validate_cases(bad,audit,'smoke')
        raise AssertionError(f'altered {column} not detected')
    except ValueError:
        pass
try:
    mod.validate_cases(cases[:-1],audit,'all')
    raise AssertionError('missing text pair not detected')
except ValueError:
    pass
print('PASS: 4-image scope, 100-image scope, caption/text mismatch, and missing row')

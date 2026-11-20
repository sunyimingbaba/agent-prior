"""Link blinded final-answer evidence to verified inputs; never infer text from hidden states."""
from pathlib import Path
import json,csv,hashlib
from collections import Counter,defaultdict
from dimensions import report_dimension
root=Path(__file__).resolve().parents[1]
def tsv(p):
    with (root/'review'/p).open() as f:
        return list(csv.DictReader(f,delimiter='\t'))
def save(name,data):
    with (root/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]))
        w.writeheader();w.writerows(data)
votes={v['id']:v for v in tsv('text_votes.tsv')}
keys=tsv('text_key.tsv')+tsv('replay_key.tsv')
assert set(votes)=={r['id'] for r in keys if r['auto']=='False'} and len(keys)==376
labeled=[]
for r in keys:
    if r['auto']=='True':
        outcome='neither'
        evidence='无可判别文字'
    else:
        v=votes[r['id']]
        outcome=v['vote']
        evidence=v['evidence']
        if outcome in ('1','2'):
            outcome='correct' if r['choice'+outcome]==r['positive'] else 'wrong'
    labeled.append(dict(id=r['id'],method=r['method'],dimension=report_dimension(r['dimension'],r['template']),image=r['image'],template=r['template'],
                        outcome=outcome,strict_sensitivity='neither' if r['id'] in ('T111','T215') else outcome,ambiguous=int('ambiguous:' in evidence.lower()),evidence=evidence))
save('生成文字逐项核验.csv',labeled)
totals=[]
for m in sorted({r['method'] for r in labeled}):
    for dim in ('all','quantity','attribute','relation'):
        items=[r for r in labeled if r['method']==m and dim in ('all',r['dimension'])]
        c=Counter(r['outcome'] for r in items)
        totals.append(dict(method=m,dimension=dim,n=len(items),correct=c['correct'],wrong=c['wrong'],neither=c['neither'],
                           both=c['both'],strict_correct=sum(r['strict_sensitivity']=='correct' for r in items),
                           strict_wrong=sum(r['strict_sensitivity']=='wrong' for r in items),
                           strict_neither=sum(r['strict_sensitivity']=='neither' for r in items),ambiguous=sum(r['ambiguous'] for r in items)))
save('生成文字判断.csv',totals)
inputs={r['image']:r for r in json.loads((root/'inputs/candidates.json').read_bytes())}
settings=json.loads((root/'outputs/AGSE_archived/settings.json').read_bytes())
files=[]
for file in sorted((root/'priors/AGSE_original').glob('*.json')):
    r=json.loads(file.read_bytes())
    assert r['query']==inputs[r['image']]['captions'][0] and r['rgb_sha256']==inputs[r['image']]['rgb_sha256']
    assert r['archive_sha256']==settings['priors_sha256'][r['image']]
    assert hashlib.sha256(file.with_suffix('.pt').read_bytes()).hexdigest()==r['sha256']
    last=r['rounds'][-1]['decoded']
    files.append(dict(image=r['image'],query=r['query'],identical=r['old_bit_exact'],archive_cos=r['archive_cos'],
                      archive_sha256=r['archive_sha256'],replay_sha256=r['sha256'],rounds=len(r['rounds']),
                      first_eos=r['first_eos'],tools=r['used'],final_text=last))
assert len(files)==104 and all(r['identical'] for r in files)
save('历史先验逐图回放.csv',files)
print(totals[::4])
native=[]
for method in ('ASPF','AGSE','AGSE_scene','AGSE_original'):
    traces=[json.loads((root/'priors'/method/Path(n).with_suffix('.json')).read_bytes()) for n in settings['priors_sha256']]
    final=[r['rounds'][-1] for r in traces]
    native.append(dict(method=method,images=len(traces),first_eos=sum(r['first_eos'] for r in traces),
                       with_tool=sum(r['used']>0 for r in traces),last_visible=sum(bool(t['decoded'].strip()) for t in final),
                       last_contains_addCriterion=sum('addCriterion' in t['decoded'] for t in final),
                       last_native_length_64=sum(len(t['ids'])==64 for t in final)))
save('四种输入生成核对.csv',native)

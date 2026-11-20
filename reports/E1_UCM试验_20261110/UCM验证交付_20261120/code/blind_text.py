"""Blind semantic review of actual final decoded answers from the prespecified main sample."""
import json,csv,random,re,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
cases=json.loads((root/'inputs/cases.json').read_bytes())
cases=[r for r in cases if (r['cohort']=='previous' and r['dimension']=='quantity' and r['template']=='A') or r['cohort']=='extension']
original='--original' in sys.argv
prefix='replay_' if original else 'text_'
rng=random.Random(8491 if original else 14687)
rows=[]
for r in cases:
    for m in (('AGSE_original',) if original else ('ASPF','AGSE','AGSE_scene')):
        trace=json.loads((root/'priors'/m/Path(r['image']).with_suffix('.json')).read_bytes())
        decoded=trace['rounds'][-1]['decoded'].strip()
        texts=[r['positive'],r['negative']]
        rng.shuffle(texts)
        # An empty answer or only protocol debris supplies no detail from either candidate.
        auto=not any(c.isalnum() for c in re.sub(r'\b(addCriterion|scene_classify)\b','',decoded))
        rows.append(dict(image=r['image'],method=m,template=r['template'],dimension=r['dimension'],positive=r['positive'],
                         answer=decoded,choice1=texts[0],choice2=texts[1],auto=auto))
rng.shuffle(rows)
for k,r in enumerate(rows):r['id']=('R' if original else 'T')+('%03d'%k)
def write(name, data,fields):
    with (root/'review'/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',delimiter='\t')
        w.writeheader();w.writerows(data)
write(prefix+'key.tsv',rows,list(rows[0]))
write(prefix+'blind.tsv',(r for r in rows if not r['auto']),['id','answer','choice1','choice2'])
print('all',len(rows),'for reading',sum(not r['auto'] for r in rows))

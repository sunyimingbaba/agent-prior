from pathlib import Path
from shared import ROOT,read,dump,sha
import csv
def main():
    rows=read(ROOT/'inputs/detail_proposals.json')
    audit={z['image']:z for z in read(ROOT/'review/detail_audit.json')}
    assert set(audit)=={z['image'] for z in rows}
    with (ROOT/'review/画面重叠候选.csv').open(encoding='utf-8-sig') as f:assert not list(csv.DictReader(f))
    retained=[]
    for r in rows:
        check=audit[r['image']]
        if not check['keep']:continue
        cf=check.get('cf') or {}
        if cf:
            for w in ('C','D'):
                replaced=[]
                for s in r[w]:
                    for typ in ['subject','reference']:
                        a=r[typ];b=cf.get(typ) or a
                        s=s.replace(a,b).replace(a.capitalize(),b.capitalize())
                    replaced.append(s)
                r[w]=replaced
            for typ in ['subject','reference']:
                if cf.get(typ):r[typ]=cf[typ]
        retained.append(r)
    assert all(z['fresh'] for z in retained)
    dump(ROOT/'inputs/detail.json',retained)
    names=['inputs/'+s+'.json' for s in ['train_lock','protocol','secondary_eval','validation','detail_proposals','detail']]+['review/detail_audit.json','review/画面重叠候选.csv','review/画面重叠筛查.json']
    dump(ROOT/'inputs/eval_lock.json',{name:sha(ROOT/name) for name in names})
    print('retained',len(retained),[r['image'] for r in retained])
if __name__=='__main__':main()

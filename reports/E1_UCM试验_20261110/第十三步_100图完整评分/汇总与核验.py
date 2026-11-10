#!/usr/bin/env python3
"""Verify and summarize the completed E1 pairwise scoring run."""
import argparse, collections, csv, importlib.util, json, math
from pathlib import Path

p=argparse.ArgumentParser()
for key in ['results','cases','audit','historical-four','historical-seventeen','comparator','output-dir']:
    p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
results=read(a.results);cases=read(a.cases);audit=read(a.audit)
assert len(cases)==200 and len(audit)==100 and len(results)==400
case={(x['图像编号'],x['句式']):x for x in cases}
assert len(case)==200
annotations={x['图像编号']:x for x in audit};assert len(annotations)==100
expected={(m,n,w) for m in ['CLIP','AGSE'] for n,w in case}
indexed={}
for r in results:
    key=(r['model'],r['image'],r['wording'])
    assert key not in indexed and key in expected
    ref=case[(r['image'],r['wording'])]
    for x,y in [('text1','文本1'),('text2','文本2'),('correct_text_position','正确文本位置'),('original_first_caption','原始第一条描述'),('split','数据来源'),('actual_count','实际数量'),('incorrect_count','错误数量')]:
        assert r[x]==ref[y],(key,x)
    s1=float(r['score1']);s2=float(r['score2']);pos=int(r['correct_text_position'])
    assert all(math.isfinite(x) for x in (s1,s2,float(r['margin'])))
    cp=(s1,s2)[pos-1];cn=(s1,s2)[2-pos]
    assert abs(cp-float(r['correct_score']))<1e-12 and abs(cn-float(r['wrong_score']))<1e-12
    assert abs((cp-cn)-float(r['margin']))<1e-12
    assert (r['correct']=='True')==(cp>cn) and (r['tie']=='True')==(cp==cn)
    indexed[key]=r
assert set(indexed)==expected

spec=importlib.util.spec_from_file_location('historical_compare',a.comparator);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
oldfour=read(a.historical_four)
four=[r for r in results if r['image'] in mod.EXPECTED_IMAGES]
four_comparison=mod.compare(oldfour,four)
assert four_comparison['passed'],four_comparison
old17=read(a.historical_seventeen);assert len(old17)==17
max17=0.0
for old in old17:
    for method in ['CLIP','AGSE']:
        r=indexed[(method,old['image'],'A_原句式')]
        assert r['original_first_caption']==old['original_first_caption']
        expected_pos=old['positive'];expected_neg=old['negative'];pos=int(r['correct_text_position'])
        assert (r['text1'],r['text2'])[pos-1]==expected_pos and (r['text1'],r['text2'])[2-pos]==expected_neg
        d=max(abs(float(r['correct_score'])-float(old[method.lower()+'_positive'])),abs(float(r['wrong_score'])-float(old[method.lower()+'_negative'])))
        max17=max(max17,d)
        assert d<=1e-6,(method,old['image'],d)

def stats(rows):
    n=len(rows);c=sum(r['correct']=='True' for r in rows);t=sum(r['tie']=='True' for r in rows)
    return {'n':n,'correct':c,'ties':t,'pairacc':c/n if n else None,'mean_margin':sum(float(r['margin']) for r in rows)/n if n else None}

summary={'overall':{},'grouped':[], 'wording_change':{}, 'method_comparison':{},
         'checks':{'results_complete':True,'cases_match':True,'historical_four':{k:v for k,v in four_comparison.items() if k!='details'},
                   'historical_seventeen_max_difference':max17,'historical_seventeen_checked_model_image_pairs':34}}
for method in ['CLIP','AGSE']:
    for wording in ['A_原句式','B_改写句式']:
        rows=[r for r in results if r['model']==method and r['wording']==wording]
        assert len(rows)==100
        summary['overall'][f'{method}_{wording}']=stats(rows)
        groupers={
          '数据来源':lambda r:r['split'],
          '目标类别':lambda r:'沙坑' if r['object']=='高尔夫沙坑' else r['object'],
          '实际数量':lambda r:r['actual_count'],
          '数量方向':lambda r:'正确数量较小' if int(r['actual_count'])<int(r['incorrect_count']) else '正确数量较大',
          '第一条描述明确含对应数词':lambda r:'是' if annotations[r['image']]['第一条描述出现对应英文数词']=='True' else '未检出',
        }
        for group,fn in groupers.items():
            values=sorted({fn(r) for r in rows})
            for value in values:
                summary['grouped'].append({'model':method,'wording':wording,'group':group,'value':value,**stats([r for r in rows if fn(r)==value])})
    switches=[]
    for name in sorted(annotations):
        left=indexed[(method,name,'A_原句式')];right=indexed[(method,name,'B_改写句式')]
        def sign(r):
            x=float(r['margin']);return (x>0)-(x<0)
        switches.append((sign(left),sign(right)))
    summary['wording_change'][method]={'images':len(switches),'ranking_changed':sum(x!=y for x,y in switches),
                                      'A_correct_B_incorrect':sum(x==1 and y==-1 for x,y in switches),
                                      'A_incorrect_B_correct':sum(x==-1 and y==1 for x,y in switches),
                                      'ties_in_either':sum(x==0 or y==0 for x,y in switches)}
for w in ['A_原句式','B_改写句式']:
    combos=collections.Counter((indexed[('CLIP',name,w)]['correct']=='True',indexed[('AGSE',name,w)]['correct']=='True') for name in annotations)
    summary['method_comparison'][w]={'both_correct':combos[(True,True)],'clip_only':combos[(True,False)],
                                      'agse_only':combos[(False,True)],'both_incorrect_or_tie':combos[(False,False)]}
(a.output_dir/'结果汇总.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (a.output_dir/'分组统计.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary['grouped'][0]));w.writeheader();w.writerows(summary['grouped'])
print(json.dumps({k:v for k,v in summary.items() if k!='grouped'},ensure_ascii=False,indent=2))

#!/usr/bin/env python3
import csv,json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent
scores=json.loads((p/'打分结果.json').read_text());generation=json.loads((p/'先验生成记录.json').read_text());settings=json.loads((p/'生成设置.json').read_text())
assert len(scores)==36 and len(generation)==12 and settings['count']==12
by={(x['image'],x['condition'],x['wording']):x for x in scores};assert len(by)==36
images=sorted({x['image'] for x in scores});assert len(images)==6
words=('A_原句式','B_改写句式');result=[]
for image in images:
 for wording in words:
  original=by[image,'original',wording];neutral=by[image,'neutral',wording];historic=by[image,'historic',wording]
  assert original['correct_text']==neutral['correct_text']==historic['correct_text']
  assert original['wrong_text']==neutral['wrong_text']==historic['wrong_text']
  assert all(abs(x['margin']-(x['correct_score']-x['wrong_score']))<1e-12 for x in (original,neutral,historic))
  assert historic['historical_score_max_abs_diff']<1e-6
  result.append({'图像':image,'句式':wording,'正确描述':original['correct_text'],'错误描述':original['wrong_text'],'历史判断':historic['correct'],'原始描述判断':original['correct'],'中性描述判断':neutral['correct'],'原始描述分差':original['margin'],'中性描述分差':neutral['margin'],'中性减原始分差':neutral['margin']-original['margin'],'判断是否改变':original['correct']!=neutral['correct']})
with (p/'六图逐项对照.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
summary={'images':len(images),'comparisons':len(result),'changed':sum(x['判断是否改变'] for x in result),'mean_margin_change':sum(x['中性减原始分差'] for x in result)/len(result),'max_abs_margin_change':max(abs(x['中性减原始分差']) for x in result),'historical_score_max_abs_diff':max(x['historical_score_max_abs_diff'] for x in scores if x['condition']=='historic'),'addCriterion_traces':sum(x['contains_addCriterion'] for x in generation),'tool_calls':sum(x['tool_used'] for x in generation),'by_wording':{w:{c:sum(by[i,c,w]['correct'] for i in images) for c in ('historic','original','neutral')} for w in words}}
(p/'统计核验.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))

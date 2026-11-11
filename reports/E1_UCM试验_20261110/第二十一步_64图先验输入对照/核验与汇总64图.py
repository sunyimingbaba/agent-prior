#!/usr/bin/env python3
"""Summarize image-paired E1 results; keep six exploratory images identifiable."""
import csv,json,statistics
from collections import Counter
from pathlib import Path
p=Path(__file__).resolve().parent
scores=json.loads((p/'64图评分.json').read_text());new=json.loads((p/'58图生成记录.json').read_text());six=json.loads((p.parent/'第十七步_六图先验输入对照/先验生成记录.json').read_text())
assert len(scores)==384 and len(new)==116 and len(six)==12
assert len({(x['image'],x['condition'],x['wording']) for x in scores})==384
pairs={(x['image'],x['condition'],x['wording']):x for x in scores};names=sorted({x['image'] for x in scores});assert len(names)==64
words=['A_原句式','B_改写句式'];results=[]
for name in names:
 for word in words:
  a=pairs[name,'original',word];b=pairs[name,'neutral',word];old=pairs[name,'historic',word]
  assert a['correct_text']==b['correct_text']==old['correct_text'] and a['wrong_text']==b['wrong_text']==old['wrong_text']
  for x in (a,b,old):
   assert abs(x['correct_score']-x['wrong_score']-x['margin'])<1e-10
   assert x['correct']==(x['correct_score']>x['wrong_score']) and x['tie']==(x['correct_score']==x['wrong_score'])
  results.append({'图像':name,'来源':a['split'],'样本阶段':a['cohort'],'数量组':a['group'],'真实数量':a['actual_count'],'错误数量':a['wrong_count'],'句式':word,'正确描述':a['correct_text'],'错误描述':a['wrong_text'],'历史先验判断':old['correct'],'原始描述判断':a['correct'],'弱化数量判断':b['correct'],'原始描述分差':a['margin'],'弱化数量分差':b['margin'],'分差变化':b['margin']-a['margin'],'排序改变':a['correct']!=b['correct'],'平分':a['tie'] or b['tie']})
assert len(results)==128
fields=list(results[0])
with (p/'逐图配对结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
def summary(items):
 out={'images':len({x['图像'] for x in items}),'observations':len(items),'tie_observations':sum(x['平分'] for x in items)}
 for word in words:
  r=[x for x in items if x['句式']==word];n=len(r)
  if not n:continue
  out[word]={'n':n,'original_correct':sum(x['原始描述判断'] for x in r),'neutral_correct':sum(x['弱化数量判断'] for x in r),'historic_correct':sum(x['历史先验判断'] for x in r),'original_right_neutral_wrong':sum(x['原始描述判断'] and not x['弱化数量判断'] for x in r),'original_wrong_neutral_right':sum(not x['原始描述判断'] and x['弱化数量判断'] for x in r),'unchanged':sum(not x['排序改变'] for x in r),'mean_margin_delta':sum(x['分差变化'] for x in r)/n,'median_margin_delta':statistics.median(x['分差变化'] for x in r)}
 return out
summarys={'all64':summary(results),'new58':summary([x for x in results if x['样本阶段']=='new_58']),'explore6':summary([x for x in results if x['样本阶段']=='exploratory_6']),'train':summary([x for x in results if x['来源']=='UCM train']),'val':summary([x for x in results if x['来源']=='UCM val']),'includes_one':summary([x for x in results if x['数量组']=='includes_one']),'both_above_one':summary([x for x in results if x['数量组']=='both_above_one'])}
records=new+six;assert len({(r['image'],r['condition']) for r in records})==128
summarys['generation']={'count':len(records),'contains_addCriterion':sum(x['contains_addCriterion'] for x in records),'tool_calls':sum(x['tool_used'] for x in records),'tool_calls_below_029':sum(sum(event['confidence']<0.29 for event in x['tool_events']) for x in records),'by_condition':{cond:{'records':sum(x['condition']==cond for x in records),'called':sum(x['tool_used']>0 for x in records if x['condition']==cond),'low_confidence_events':sum(sum(e['confidence']<0.29 for e in x['tool_events']) for x in records if x['condition']==cond)} for cond in ['original','neutral']}}
genmap={(r['image'],r['condition']):r for r in records}
summarys['tool_changes']=[{'image':n,'original':genmap[n,'original']['tool_used'],'neutral':genmap[n,'neutral']['tool_used']} for n in names if bool(genmap[n,'original']['tool_used'])!=bool(genmap[n,'neutral']['tool_used'])]
summarys['ranking_changes']=[{'image':x['图像'],'wording':x['句式'],'original_correct':x['原始描述判断'],'neutral_correct':x['弱化数量判断'],'original_margin':x['原始描述分差'],'neutral_margin':x['弱化数量分差']} for x in results if x['排序改变']]
summarys['historic_max_abs_score_error']=max(x.get('historical_score_max_abs_diff',0) for x in scores if x['condition']=='historic')
(p/'汇总与核验.json').write_text(json.dumps(summarys,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(summarys,ensure_ascii=False,indent=2))

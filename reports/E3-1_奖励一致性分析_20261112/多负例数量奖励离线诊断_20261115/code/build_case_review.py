from pathlib import Path
import json
import csv
from common import ROOT,read,dump,sha
parent=ROOT.parent/read(ROOT/'plan.json')['parent_directory']/'原始运行记录';scores=read(ROOT/'candidate_scores.json');raw={x['id']:x for x in read(ROOT/'source_data/原始运行记录/scoring/all_scores.json')};selected=[]
for x in scores:
    if x['arm']=='after' and x['image']=='2035.tif' and x['role']=='candidate':
        trace=parent/raw[x['id']]['trace_file']
        if not trace.exists():trace=ROOT/'case_traces'/(x['id']+'.json')
        assert sha(trace)==raw[x['id']]['trace_sha256'];tr=read(trace)
        saved=ROOT/'case_traces'/(x['id']+'.json');saved.parent.mkdir(exist_ok=True)
        if saved.exists():assert saved.read_bytes()==trace.read_bytes()
        else:saved.write_bytes(trace.read_bytes())
        selected.append({k:x[k] for k in ['id','image','actual_count','A_pair_margin','A_hard_margin','B_pair_correct','B_hard_correct','B_argmax_count']}|{'first_text':tr['rounds'][0]['generated_text'],'final_text':tr['rounds'][-1]['generated_text'],'trace_sha256':raw[x['id']]['trace_sha256'],'text_fact_accuracy':'[待确认]','trace':tr})
assert len(selected)==4
dump(ROOT/'decisive_case_2035.json',selected);print('DECISIVE_CASE_FOUR_TRACES_HASHED')

per=read(ROOT/'per_image_selections.json');lookup={(x['arm'],x['image'],x['selector']):x for x in per};changes=[]
for x in per:
    if x['selector']!='hard_total':continue
    old=lookup[x['arm'],x['image'],'pair_total']
    if x['selected_ids']!=old['selected_ids']:
        changes.append({'arm':x['arm'],'cohort':x['cohort'],'image':x['image'],'object':x['object'],'actual_count':x['actual_count'],'original_selected':old['selected_ids'][0],'hard_selected':x['selected_ids'][0],'B_hard_before':old['B_hard_correct'],'B_hard_after':x['B_hard_correct'],'B_hard_delta':x['B_hard_correct']-old['B_hard_correct'],'B_margin_delta':x['B_hard_margin']-old['B_hard_margin']})
with (ROOT/'selector_changes.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(changes[0]),lineterminator='\n');w.writeheader();w.writerows(changes)

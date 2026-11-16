"""Preserve a deterministic representative for every observed protocol stop category."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
rows=json.loads((r/'generation_v2/records.json').read_text());seen=set();w=[]
for x in rows:
 key=x['group'],x['stop_reason'],x['strict_protocol_usable']
 if key in seen:continue
 seen.add(key);t=json.loads((r/x['trace_file']).read_text());w.append({'record':x,'prompt_roles':[q['message_roles'] for q in t['rounds']],'generated_texts':[q['generated_text'] for q in t['rounds']],'feedback_events':t['tool_events'],'software_forced_tool':t['software_forced_tool'],'candidate_coverage':t['state_coverage']})
(r/'protocol_witnesses.json').write_text(json.dumps({'selection':'first generation-order record per observed group/stop/strict status; utility unobserved','witnesses':w},ensure_ascii=False,indent=2)+'\n');print('PROTOCOL_WITNESSES_WRITTEN',len(w))

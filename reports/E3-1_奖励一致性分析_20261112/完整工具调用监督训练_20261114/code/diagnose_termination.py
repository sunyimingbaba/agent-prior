"""Exploratory controller comparison from existing fixed-policy records."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer

torch.set_num_threads(2)
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;raw=r/'原始运行记录'
scores=json.loads((raw/'scoring/all_scores.json').read_text());cases=json.loads((r/'inputs/evaluation_cases.json').read_text())
index={(x['arm'],x['image'],x['role']):x for x in scores if x['role']=='greedy'}
tok=Tokenizer.from_file(str(raw/'tokenizer_snapshot/training_tokenizer.json'));rows=[]
for item in cases:
    s=index['sft_strict',item['image'],'greedy'];l=index['sft_legacy_greedy',item['image'],'greedy']
    st=json.loads((raw/s['trace_file']).read_text());lt=json.loads((raw/l['trace_file']).read_text());sr=st['rounds'][-1];lr=lt['rounds'][-1]
    sh=torch.load(raw/s['state_file'],map_location='cpu',weights_only=True).double();lh=torch.load(raw/l['state_file'],map_location='cpu',weights_only=True).double()
    strip_eos=lambda z:[x for x in z['generated_ids'] if x not in z['eos_token_ids']]
    rows.append({'image':item['image'],'split':item['split'],'same_final_decoded_text':sr['generated_text']==lr['generated_text'],
      'same_final_generated_ids_excluding_eos':strip_eos(sr)==strip_eos(lr),
      'strict_state_input_id':sr['state_input_token_id'],'strict_state_input_text':tok.decode([sr['state_input_token_id']],skip_special_tokens=False),
      'strict_last_emitted_id':sr['last_emitted_token_id'],'strict_last_emitted_text':tok.decode([sr['last_emitted_token_id']],skip_special_tokens=False),
      'legacy_state_input_id':lr['state_input_token_id'],'legacy_state_input_text':tok.decode([lr['state_input_token_id']],skip_special_tokens=False),
      'legacy_last_emitted_id':lr['last_emitted_token_id'],'legacy_last_emitted_text':tok.decode([lr['last_emitted_token_id']],skip_special_tokens=False),
      'strict_ended_on_eos':sr['last_emitted_token_id'] in sr['eos_token_ids'],'legacy_ended_on_eos':lr['last_emitted_token_id'] in lr['eos_token_ids'],
      'strict_first_call':st['rounds'][0]['generated_text'],'legacy_first_call':lt['rounds'][0]['generated_text'],
      'hidden_cosine':float(torch.nn.functional.cosine_similarity(sh[None],lh[None])),
      'strict_B_margin':s['margin_B'],'legacy_B_margin':l['margin_B'],'strict_B_correct':s['correct_B'],'legacy_B_correct':l['correct_B'],
      'strict_record_id':s['id'],'legacy_record_id':l['id'],'strict_final_text':sr['generated_text'],'legacy_final_text':lr['generated_text']})
groups={}
for split in ['train','val']:
    xs=[x for x in rows if x['split']==split];n=len(xs);ds=np.array([float(x['strict_B_correct'])-float(x['legacy_B_correct']) for x in xs])
    draw=np.random.default_rng(2026111438).integers(0,n,size=(10000,n))
    groups[split]={'images':n,'same_final_decoded_text':sum(x['same_final_decoded_text'] for x in xs),
      'same_final_generated_ids_excluding_eos':sum(x['same_final_generated_ids_excluding_eos'] for x in xs),
      'strict_ended_on_eos':sum(x['strict_ended_on_eos'] for x in xs),'legacy_ended_on_eos':sum(x['legacy_ended_on_eos'] for x in xs),
      'mean_hidden_cosine':float(np.mean([x['hidden_cosine'] for x in xs])),
      'strict_minus_legacy_B_pairacc':float(ds.mean()),'interval95':[float(v) for v in np.percentile(ds[draw].mean(axis=1),[2.5,97.5])]}
out={'ok':True,'scope':'exploratory comparison of two existing controllers on the same trained 10cb checkpoint; original readiness gates unchanged',
 'observed_changes':'complete versus early tool boundary and answer-closing versus EOS stopping coexist',
 'next_isolation':'keep full tool calls and original system fixed; generate final answers through EOS and compare cached actual last-step states',
 'groups':groups,'per_image':rows}
(r/'termination_alignment_diagnostic.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (r/'终止状态逐图对照.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print(json.dumps(groups))

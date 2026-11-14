import json
from pathlib import Path
from transformers import AutoTokenizer
from milestone_common import ROOT,ASSETS,dump,sha

base=ASSETS/'models/Qwen2.5-VL-7B-Instruct';tokenizer=AutoTokenizer.from_pretrained(str(base))
tool_ids={s:tokenizer.convert_tokens_to_ids(s) for s in ['<tool_call>','</tool_call>']}
full='<tool_call>scene_classify</tool_call>';encoded=tokenizer.encode(full,add_special_tokens=False)
groups={}
for arm in ['strict_boundary_t1','neutral_contract_t1','neutral_contract_t07']:
    records=[json.loads(s) for s in (ROOT/'pilot'/arm/'records.jsonl').read_text().splitlines()]
    examples=[];bare_eos=closed=0
    for x in records:
        trace=json.loads((ROOT/x['trace_file']).read_text())
        for z in trace['rounds']:
            ids=z['generated_ids'];text=z['generated_text'].strip()
            closed+=tool_ids['</tool_call>'] in ids
            if text=='<tool_call>scene_classify' and ids[-1] in z['eos_token_ids']:
                bare_eos+=1
                if len(examples)<3:examples.append({'record_id':x['id'],'generated_ids':ids,
                    'raw_decode':tokenizer.decode(ids,skip_special_tokens=False),'saved_decode':text,
                    'trace_sha256':x['trace_sha256']})
    groups[arm]={'records':len(records),'bare_registered_call_followed_by_eos':bare_eos,
                 'rounds_emitting_tool_closing_token':closed,'examples':examples}
dump(ROOT/'tool_terminal_diagnosis.json',{'ok':True,'scope':'observed token endings in the fixed three calibration arms',
  'tool_token_ids':tool_ids,'full_call_encoding':encoded,
  'full_call_decode_skip_special_tokens':tokenizer.decode(encoded,skip_special_tokens=True),
  'full_call_decode_raw':tokenizer.decode(encoded,skip_special_tokens=False),
  'tokenizer_config_sha256':sha(base/'tokenizer_config.json'),'groups':groups})
print(json.dumps({'tool_token_ids':tool_ids,'full_call_round_trip':tokenizer.decode(encoded,skip_special_tokens=True),
                 'groups':{k:{a:b for a,b in v.items() if a!='examples'} for k,v in groups.items()}}))

import json
from transformers import AutoProcessor
from milestone_common import ROOT,ASSETS,dump,sha
from supervised_format import target_tokens

processor=AutoProcessor.from_pretrained(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'))
sources=json.loads((ROOT/'inputs/teacher_sources.json').read_text());rows=[]
for source in sources:
    for kind in ['call','answer']:
        x=target_tokens(processor.tokenizer,kind,source['source_body'] if kind=='answer' else '')
        if kind=='call':assert processor.tokenizer.convert_tokens_to_ids('</tool_call>') in x['target_ids']
        else:
            lo,hi=x['body_char_span'];assert any(not v for v in x['loss_mask'][:-1])
            for (start,end),active in zip(x['offsets'][:-1],x['loss_mask'][:-1]):
                if active:assert end<=lo or start>=hi
        rows.append({'image':source['image'],**x})
folder=ROOT/'tokenization_gate';folder.mkdir(exist_ok=False)
(folder/'training_tokenizer.json').write_text(processor.tokenizer.backend_tokenizer.to_str(),encoding='utf-8')
dump(folder/'result.json',{'ok':True,'images':15,'assistant_targets':len(rows),
    'image_merge_size':int(processor.image_processor.merge_size),
    'tokenizer_snapshot_sha256':sha(folder/'training_tokenizer.json'),'targets':rows})
print('ALL_30_ACTUAL_TOKENIZER_FORMAT_TARGETS_VERIFIED',sum(x['supervised_tokens'] for x in rows),flush=True)

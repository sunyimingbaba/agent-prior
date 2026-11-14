"""Verify the actual rendered input caption for every saved trajectory."""
import argparse
import json
from pathlib import Path
import re
import unicodedata

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root
branch=json.loads((r/'pilot/branch_decision.json').read_text())
g=json.loads((r/'inputs/training_text_gallery.json').read_text())
metadata={s:json.loads((r/'inputs'/(s+'_metadata.json')).read_text())['rows'] for s in ['train','val']}
canon=lambda s:re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).casefold()).strip().rstrip(' .!?')
val_text={canon(c) for x in metadata['val'] for c in x['captions']}
paths=[r/'pilot/records.jsonl']+[r/'training'/x/'samples.jsonl' for x in ['smoke']+branch['arms']]+[r/'evaluation'/x/'records.jsonl' for x in branch['policies']]
counts={'train':0,'val':0}
for path in paths:
    for line in path.read_text().splitlines():
        x=json.loads(line);split=x['split'];row=x['row'];assert metadata[split][row]['image']==x['image']
        expected=g['maps'][str(row)]['prompt_caption'] if split=='train' else metadata['val'][row]['captions'][0]
        if split=='train':assert canon(expected) not in val_text
        trace=json.loads((r/x['trace_file']).read_text());prompt=trace['rounds'][0]['prompt_text']
        users=re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>',prompt,re.S)
        assert users and users[-1].endswith('查询文本: '+expected)
        counts[split]+=1
out={'ok':True,'actual_rendered_training_captions_checked':counts['train'],'actual_rendered_validation_captions_checked':counts['val'],
     'training_queries_canonical_overlap_with_val':0,
     'shared_fewshot_query_canonical_overlap_with_val':int(canon('There is a harbor with several ships.') in val_text)}
a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out))

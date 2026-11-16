"""Offline cache verification on server or local evidence, with optional parent replay."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
import torch

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=a.root;plan=read(r/'cache_plan.json');records=read(r/'cache/records.json')
assert len(records)==2100 and len({x['image'] for x in records})==2100
quality=Counter();splits=Counter();replays=Counter();rounds=0
old={}
if a.parent:
    old.update({x['image']:x for x in read(a.parent/'states/records.json') if x['family']=='balanced32' and x['role']=='greedy' and x['split']=='train'})
    old.update({x['image']:x for x in read(a.parent/'full_validation/records.json')})
for x in records:
    assert sha(r/x['state_file'])==x['state_file_sha256'] and sha(r/x['trace_file'])==x['trace_sha256']
    h=torch.load(r/x['state_file'],map_location='cpu',weights_only=True)
    assert h.shape==(3584,) and h.dtype==torch.bfloat16 and torch.isfinite(h).all()
    raw=hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()
    tr=read(r/x['trace_file'])
    assert raw==x['numeric_state_sha256']==tr['state_tensor_sha256']==tr['rounds'][-1]['state_tensor_sha256']
    first=tr['rounds'][0]['prompt_text'].rsplit('<|im_start|>user',1)[-1]
    assert first.endswith('查询文本: '+plan['prompt']+'<|im_end|>\n<|im_start|>assistant\n')
    assert not tr['forced_tool_witness']
    for q in tr['rounds']:
        ids=q['generated_ids'];assert ids and len(ids)==q['generated_token_count']
        assert (q['prompt_token_ids']+ids)[-2]==q['state_input_token_id']
        assert not any(t in q['eos_token_ids'] for t in ids[:-1])
        rounds+=1
    if tr['quality_usable']:
        assert tr['rounds'][-1]['generated_ids'][-1] in tr['rounds'][-1]['eos_token_ids']
    quality[str(tr['quality_usable'])]+=1;splits[x['split']]+=1
    if x['image'] in old:
        y=old[x['image']]
        assert torch.equal(h,torch.load(a.parent/y['state_file'],map_location='cpu',weights_only=True))
        assert raw==y['numeric_state_sha256'];replays[x['split']]+=1
assert dict(splits)=={'train':1680,'val':210,'test':210}
if a.parent:assert replays['train']==63 and replays['val']==210
j={'ok':True,'states_verified':2100,'files_verified':4200,'split_counts':dict(splits),
   'uniform_queries_checked':2100,'quality_counts':dict(quality),'generation_rounds_checked':rounds,
   'parent_exact_state_replays':dict(replays),'Agent_optimizer_updates':0}
a.out.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');print(json.dumps(j))

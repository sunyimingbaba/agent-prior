"""Verify all cache files and transcript contracts before readout training."""
import hashlib
import re
from collections import Counter
from pathlib import Path
import torch
from milestone_common import ROOT, PLAN
from stage_io import read, dump, sha

torch.set_num_threads(2)
parent = Path(PLAN['parent_root'])
records = read(ROOT/'cache/records.json')
assert len(records) == 2100 and len({(x['split'],x['row']) for x in records}) == 2100
old_train = {x['image']: x for x in read(parent/'states/records.json')
             if x['family']=='balanced32' and x['role']=='greedy' and x['split']=='train'}
old_val = {x['image']: x for x in read(parent/'full_validation/records.json')}
matched = Counter()
quality = Counter()
tools = 0
rounds = 0
for x in records:
    for key, hkey in [('state_file','state_file_sha256'),('trace_file','trace_sha256')]:
        assert sha(ROOT/x[key]) == x[hkey]
    h = torch.load(ROOT/x['state_file'],map_location='cpu',weights_only=True)
    assert h.shape==(3584,) and h.dtype==torch.bfloat16 and torch.isfinite(h).all()
    numeric = hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()
    assert numeric == x['numeric_state_sha256']
    tr = read(ROOT/x['trace_file'])
    assert numeric == tr['state_tensor_sha256'] == tr['rounds'][-1]['state_tensor_sha256']
    assert tr['rounds'][0]['prompt_text'].endswith('查询文本: '+PLAN['prompt']+'<|im_end|>\n<|im_start|>assistant\n')
    assert 1 <= len(tr['rounds']) <= PLAN['rollout']['max_rounds']
    for r in tr['rounds']:
        ids = r['generated_ids']
        assert len(ids) == r['generated_token_count'] and r['last_emitted_token_id']==ids[-1]
        assert r['state_input_token_id'] == (ids[-2] if len(ids)>1 else r['prompt_token_ids'][-1])
        assert not any(t in r['eos_token_ids'] for t in ids[:-1])
        rounds += 1
    for event in tr['tool_events']:
        assert event['origin']=='model_requested_actual_tool' and event['name']=='scene_classify'
        k = event['after_round']
        assert re.fullmatch(r'<tool_call>\s*scene_classify\s*</tool_call>',tr['rounds'][k-1]['generated_text'].strip())
        if k < len(tr['rounds']):
            assert '<result>'+event['text']+'</result>' in tr['rounds'][k]['prompt_text']
        else:
            assert tr['stop_reason'] == 'round_limit'
        tools += 1
    assert tr['quality_usable'] == x['quality_usable']
    quality[str(tr['quality_usable'])] += 1
    old = old_val.get(x['image']) if x['split']=='val' else old_train.get(x['image'])
    if old:
        assert numeric == old['numeric_state_sha256'], x['image']
        assert torch.equal(h,torch.load(parent/old['state_file'],map_location='cpu',weights_only=True))
        matched[x['split']] += 1
assert matched['val']==210 and matched['train']==63
dump(ROOT/'cache_audit.json',{'ok':True,'states_verified':len(records),'state_and_trace_file_hashes_verified':len(records)*2,
     'uniform_query_traces_verified':len(records),'generation_rounds_checked':rounds,'actual_tool_feedbacks_verified':tools,
     'old_state_exact_replays':dict(matched),'quality_counts':dict(quality),'optimizer_updates':0})
print('ALL_2100_CACHE_RECORDS_VERIFIED', flush=True)

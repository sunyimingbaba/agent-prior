"""Fixed image-only query; every split is cached without caption input to the policy."""
import datetime
import hashlib
import io
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
from PIL import Image
from milestone_common import ROOT, PLAN, ASSETS, setup, seed, load_policy, real_tools, lora_hash
from checkpoint_io import restore_adapter_exact
from live_rollout_eos import rollout, tensor_sha
from stage_io import read, dump, sha

torch = setup()
started = time.monotonic()
parent = Path(PLAN['parent_root'])
for n, h in read(ROOT/'cache_code_hashes.json').items():
    assert sha(ROOT/n) == h, n
for n, h in read(parent/'source_snapshot/inventory_before.json')['files'].items():
    assert sha(Path(PLAN['base_root'])/'CUP'/n) == h, n
out = ROOT/'cache'
out.mkdir(exist_ok=True)
old = read(parent/'full_validation/records.json')
byval = {x['row']: x for x in old}
assert len(byval) == 210
tables = {}
inventory = {}
cases = []
for split, count in PLAN['cache']['splits'].items():
    path = ASSETS/f'ucm/ucm_caption_karpathy_label_{split}.arrow'
    table = pa.ipc.open_file(str(path)).read_all()
    assert table.num_rows == count
    tables[split] = table
    inventory[split] = {'images': count, 'arrow_sha256': sha(path)}
    for row in range(count):
        img = Image.open(io.BytesIO(table['image'][row].as_py())).convert('RGB')
        cases.append({'split': split, 'row': row, 'image': table['path'][row].as_py(),
                      'rgb_sha256': hashlib.sha256(img.tobytes()).hexdigest()})
assert len({x['image'] for x in cases}) == 2100
dump(out/'image_inventory.json', cases)
dump(out/'dataset_sources.json', inventory)
records = []
for c in cases:
    if c['split'] != 'val':
        continue
    old = byval[c['row']]
    assert old['image'] == c['image']
    sf = out/f"priors/ucm_caption_karpathy_label_val/{c['row']}.pt"
    tf = out/f"traces/val_{c['row']:04d}.json"
    sf.parent.mkdir(parents=True, exist_ok=True)
    tf.parent.mkdir(parents=True, exist_ok=True)
    for source, target, expected in [(parent/old['state_file'], sf, old['state_file_sha256']),
                                      (parent/old['trace_file'], tf, old['trace_sha256'])]:
        assert sha(source) == expected
        if target.exists():
            assert sha(target) == expected
        else:
            shutil.copy2(source, target)
    records.append({**c, 'reused_val': True, 'state_file': str(sf.relative_to(ROOT)),
                    'trace_file': str(tf.relative_to(ROOT)), 'state_file_sha256': sha(sf),
                    'trace_sha256': sha(tf), 'numeric_state_sha256': old['numeric_state_sha256'],
                    'quality_usable': old['quality_usable'], 'strict_protocol_usable': old['strict_protocol_usable'],
                    'stop_reason': old['stop_reason'], 'tools_used': old['tools_used']})
ledger = out/'generated_records.jsonl'
completed = {}
if ledger.exists():
    for line in ledger.read_text().splitlines():
        x = json.loads(line)
        assert x['image'] not in completed
        for key, hkey in [('state_file','state_file_sha256'),('trace_file','trace_sha256')]:
            assert sha(ROOT/x[key]) == x[hkey]
        assert tensor_sha(torch.load(ROOT/x['state_file'], map_location='cpu', weights_only=True)) == x['numeric_state_sha256']
        completed[x['image']] = x
policy = PLAN['policies'][0]
seed(0)
model, processor = load_policy(Path(policy['checkpoint']), False)
restore_adapter_exact(model, Path(policy['checkpoint']))
assert lora_hash(model) == policy['lora_hash']
from agent.rollout import SYSTEM_PROMPT
tools = real_tools()
for row in [0, 1]:
    img = Image.open(io.BytesIO(tables['val']['image'][row].as_py())).convert('RGB')
    seed(0)
    h, tr = rollout(model, processor, img, PLAN['prompt'], tools, SYSTEM_PROMPT, PLAN['rollout'], temperature=0.)
    assert tensor_sha(h) == byval[row]['numeric_state_sha256']
    dump(out/f'anchor_val_{row}.json', tr)
with ledger.open('a', encoding='utf-8') as log:
    for c in cases:
        split, row, image_name = c['split'], c['row'], c['image']
        if split == 'val':
            continue
        if image_name in completed:
            x = completed[image_name]
            assert all(x[k] == v for k, v in c.items())
            records.append(x)
            continue
        img = Image.open(io.BytesIO(tables[split]['image'][row].as_py())).convert('RGB')
        seed(0)
        h, tr = rollout(model, processor, img, PLAN['prompt'], tools, SYSTEM_PROMPT, PLAN['rollout'], temperature=0.)
        assert h.shape == (3584,) and h.dtype == torch.bfloat16 and torch.isfinite(h).all()
        assert tr['rounds'][0]['prompt_text'].endswith('查询文本: '+PLAN['prompt']+'<|im_end|>\n<|im_start|>assistant\n')
        sf = out/f'priors/ucm_caption_karpathy_label_{split}/{row}.pt'
        tf = out/f'traces/{split}_{row:04d}.json'
        sf.parent.mkdir(parents=True, exist_ok=True)
        # Interrupted, uncommitted files are preserved before regenerating their record.
        for p in [sf, tf]:
            if p.exists():
                orphan = out/'interrupted_uncommitted'/str(time.time_ns())/p.name
                orphan.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(p, orphan)
        torch.save(h, sf)
        dump(tf, tr)
        x = {**c, 'reused_val': False, 'state_file': str(sf.relative_to(ROOT)),
             'trace_file': str(tf.relative_to(ROOT)), 'state_file_sha256': sha(sf), 'trace_sha256': sha(tf),
             'numeric_state_sha256': tensor_sha(h), 'quality_usable': tr['quality_usable'],
             'strict_protocol_usable': tr['strict_protocol_usable'], 'stop_reason': tr['stop_reason'],
             'tools_used': tr['tools_used']}
        log.write(json.dumps(x, ensure_ascii=False)+'\n')
        log.flush()
        records.append(x)
        completed[image_name] = x
        progress = {'phase': 'cache_generation', 'new_completed': len(completed), 'new_total': 1890,
                    'reused_val': 210, 'latest_image': image_name, 'split': split,
                    'elapsed_seconds': time.monotonic()-started,
                    'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        dump(ROOT/'cache_progress.json', progress)
        if len(completed) % 10 == 0:
            print('IMAGE_ONLY_CACHE', json.dumps(progress), flush=True)
assert len(records) == 2100 and len(completed) == 1890
assert lora_hash(model) == policy['lora_hash']
records.sort(key=lambda x: (list(PLAN['cache']['splits']).index(x['split']), x['row']))
dump(out/'records.json', records)
dump(ROOT/'cache_result.json', {'ok': True, 'states': 2100, 'new_states': 1890, 'reused_val_states': 210,
     'warm_anchor_replays': 2, 'quality_counts': dict(Counter(str(x['quality_usable']) for x in records)),
     'stop_reasons': dict(Counter(x['stop_reason'] for x in records)), 'optimizer_updates': 0,
     'lora_hash_unchanged': policy['lora_hash'], 'elapsed_seconds': time.monotonic()-started})
print('ALL_2100_IMAGE_ONLY_CACHE_COMPLETE', flush=True)

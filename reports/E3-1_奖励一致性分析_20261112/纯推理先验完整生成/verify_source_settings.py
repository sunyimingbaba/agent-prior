#!/usr/bin/env python3
"""Check local/archived source identity and the original rollout's greedy rule."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
raw = root/'原始生成记录'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
settings = json.loads((raw/'run_01/settings.json').read_text(encoding='utf-8'))
manifest = json.loads((root/'输入与模型清单.json').read_text(encoding='utf-8'))
source_hashes = {}
for name in ['generate_full_priors.py', 'run_full_generation.sh', 'monitor_full.py',
             'verify_full_priors.py', 'collect_evidence.py']:
    assert sha(root/name) == sha(raw/name), name
    source_hashes[name] = sha(root/name)
assert settings['generator_sha256'] == source_hashes['generate_full_priors.py']
assert settings['rollout_source_sha256'] == manifest['rollout_source_sha256']
assert settings['model_asset_hashes'] == manifest['model_asset_hashes']
assert settings['max_rounds'] == 3 and settings['injected_h_end'] is False
assert settings['temperature'] == 0.0
records = [json.loads(line) for line in
           (raw/'run_01/records.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(records) == 1890
for record in records:
    trace = json.loads((raw/'run_01'/record['trace_file']).read_text(encoding='utf-8'))[0]
    assert trace['last_emitted_token'] == record['last_emitted_token']
    assert trace.get('state_input_token') == record['state_input_token']
    # Original rollout.py: do_sample=(temperature>0), temperature=(... else 1.0).
    assert trace['generation_kwargs']['do_sample'] is False
    assert trace['generation_kwargs']['temperature'] == 1.0
result = {'verified': True, 'local_source_matches_server_archive': source_hashes,
          'settings_generator_hash_matches': True,
          'settings_model_and_rollout_hashes_match_manifest': True,
          'max_rounds': 3, 'injected_h_end': False,
          'outer_rollout_temperature': 0.0, 'actual_generate_do_sample': False,
          'actual_generate_temperature': 1.0,
          'all_1890_trace_terminal_tokens_and_generation_kwargs_match': True}
(root/'补充配置核验.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))

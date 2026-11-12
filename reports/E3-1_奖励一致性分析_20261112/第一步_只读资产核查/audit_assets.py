#!/usr/bin/env python3
"""Read source assets on CPU and write audit records to this new report folder."""
import collections
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import pyarrow as pa
import torch
from PIL import Image
from safetensors import safe_open

REPO = Path('/Users/botlv/CUP-code')
ASSETS = Path('/Users/botlv/cup-deploy-assets')
ARCHIVE = ASSETS / 'experiment_archive_2026-09'
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_info(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def simple(value):
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else {'shape': list(value.shape)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


# Inert placeholders for callback metadata, following the existing E1 loader.
class MyBackboneFinetuning:
    pass


for module_name in ('pytorch_lightning', 'pytorch_lightning.callbacks',
                    'pytorch_lightning.callbacks.model_checkpoint'):
    sys.modules[module_name] = types.ModuleType(module_name)
ModelCheckpoint = type('ModelCheckpoint', (), {
    '__module__': 'pytorch_lightning.callbacks.model_checkpoint'})
sys.modules['pytorch_lightning.callbacks.model_checkpoint'].ModelCheckpoint = ModelCheckpoint

result = {'date': '2026-11-12', 'execution': 'local CPU, source assets read only',
          'runtime': {'python': sys.version, 'torch': torch.__version__,
                      'pyarrow': pa.__version__},
          'git': {'head': git('rev-parse', 'HEAD'),
                  'branch': git('branch', '--show-current'),
                  'handoff_commit': git('rev-parse', '18fa341')}}

result['checkpoints'] = {}
for name in ('ucm_main_domain_tool.ckpt', 'ucm_ccb_clustering_prior.ckpt'):
    path = ARCHIVE / 'checkpoints' / name
    print('Reading checkpoint:', name, flush=True)
    with torch.serialization.safe_globals([MyBackboneFinetuning, ModelCheckpoint]):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    config = checkpoint['hyper_parameters']['config']
    weights = checkpoint['state_dict']
    callbacks = []
    for callback_key, callback_value in checkpoint.get('callbacks', {}).items():
        if isinstance(callback_value, dict):
            callbacks.append({'key': str(callback_key), **{
                k: simple(v) for k, v in callback_value.items()
                if k in ('dirpath', 'best_model_path', 'last_model_path',
                         'best_model_score', 'best_k_models')}})
    selected_config = ('use_agent_prior', 'agent_hidden_dim', 'agent_cache_dir',
                       'clip_model', 'prompt_length', 'img_features_dim',
                       'txt_features_dim', 'datasets', 'seed', 'load_path')
    entry = {**file_info(path), 'epoch': simple(checkpoint.get('epoch')),
             'global_step': simple(checkpoint.get('global_step')),
             'parameter_entries': len(weights),
             'config': {k: config.get(k) for k in selected_config},
             'agent_weights': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)}
                               for k, v in weights.items() if k.startswith('agent_proj.')},
             'callbacks': callbacks}
    result['checkpoints'][name] = entry
    print(json.dumps({'checkpoint': name, 'config': entry['config'],
                      'epoch': entry['epoch'], 'global_step': entry['global_step'],
                      'callbacks': callbacks}, ensure_ascii=False), flush=True)
    del checkpoint, weights

result['feature_caches'] = {}
caches = {}
for name in ('grpo_val_cache.pt', 'grpo_val_cache_best.pt'):
    path = ARCHIVE / 'stats' / name
    cache = torch.load(path, map_location='cpu', weights_only=True)
    caches[name] = cache
    features, indices = cache['image_features'], list(cache['image_indices'])
    norms = features.float().norm(dim=1)
    result['feature_caches'][name] = {
        **file_info(path), 'keys': list(cache), 'shape': list(features.shape),
        'dtype': str(features.dtype), 'all_finite': bool(torch.isfinite(features).all()),
        'norm_min': norms.min().item(), 'norm_max': norms.max().item(),
        'index_count': len(indices), 'indices_are_identity': indices == list(range(len(indices))),
        'indices_unique': len(indices) == len(set(indices)),
        'provenance_metadata_present': any(k not in ('image_features', 'image_indices') for k in cache)}
left, right = (caches[n]['image_features'] for n in caches)
result['cache_comparison'] = {'feature_tensors_equal': torch.equal(left, right),
                              'max_abs_difference': (left - right).abs().max().item()}

result['arrows'] = {}
tables = {}
val_map = []
for split in ('train', 'val', 'test'):
    path = ASSETS / 'arrow' / 'ucm' / f'ucm_caption_karpathy_label_{split}.arrow'
    table = pa.ipc.open_file(str(path)).read_all()
    tables[split] = table
    paths = table['path'].to_pylist()
    captions = table['caption'].to_pylist()
    result['arrows'][split] = {**file_info(path), 'rows': table.num_rows,
                               'columns': table.column_names,
                               'paths_unique': len(paths) == len(set(paths)),
                               'caption_count_histogram': dict(collections.Counter(map(len, captions)))}
    if split == 'val':
        for index, image_path in enumerate(paths):
            image_bytes = table['image'][index].as_py()
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb_sha = hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
            prior_path = (ARCHIVE / 'agent_cache' / 'agent_cache_domain_tool'
                          / f'ucm_caption_karpathy_label_val/{index}.pt')
            prior = torch.load(prior_path, map_location='cpu', weights_only=True)
            val_map.append({'cache_column_expected_from_current_builder': index,
                            'cache_image_index': caches['grpo_val_cache.pt']['image_indices'][index],
                            'arrow_row': index, 'image_path': image_path,
                            'image_bytes_sha256': hashlib.sha256(image_bytes).hexdigest(),
                            'rgb_sha256': rgb_sha, 'first_caption': captions[index][0],
                            'prior_path': str(prior_path), 'prior_sha256': sha(prior_path),
                            'prior_shape': str(tuple(prior.shape)), 'prior_dtype': str(prior.dtype),
                            'prior_finite': bool(torch.isfinite(prior).all())})
result['val_prior_checks'] = {
    'count': len(val_map), 'all_3584': all(r['prior_shape'] == '(3584,)' for r in val_map),
    'all_finite': all(r['prior_finite'] for r in val_map)}
path_sets = {split: set(table['path'].to_pylist()) for split, table in tables.items()}
result['cross_split_path_overlap'] = {
    f'{left}_{right}': len(path_sets[left] & path_sets[right])
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]}

e1_path = REPO / 'reports/E1_UCM试验_20261110/第十一步_模型与缓存核查/100图像与先验核查.csv'
with e1_path.open(encoding='utf-8-sig', newline='') as handle:
    e1 = list(csv.DictReader(handle))
e1_matches = []
for row in e1:
    split, index = row['来源'], int(row['Arrow行号'])
    table = tables[split]
    image_bytes = table['image'][index].as_py()
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb_sha = hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
    e1_matches.append({'image': row['图像编号'], 'split': split, 'arrow_row': index,
                       'path_matches': table['path'][index].as_py() == row['图像编号'],
                       'caption_matches': table['caption'][index].as_py()[0] == row['第一条描述'],
                       'rgb_matches': rgb_sha == row['图像RGB_SHA256'],
                       'included_in_historical_val_gallery_by_path': row['图像编号'] in path_sets['val']})
result['e1_overlap'] = {'samples': len(e1_matches),
    'split_counts': dict(collections.Counter(r['split'] for r in e1_matches)),
    'all_path_matches': all(r['path_matches'] for r in e1_matches),
    'all_caption_matches': all(r['caption_matches'] for r in e1_matches),
    'all_rgb_matches': all(r['rgb_matches'] for r in e1_matches),
    'historical_gallery_overlap': sum(r['included_in_historical_val_gallery_by_path'] for r in e1_matches)}

stats_path = ARCHIVE / 'stats/domain_tool_stats.jsonl'
with stats_path.open() as handle:
    stats = [json.loads(line) for line in handle if line.strip()]
groups = collections.defaultdict(list)
for row in stats:
    groups[row['step']].append(row)
result['historical_stats'] = {
    **file_info(stats_path), 'rows': len(stats), 'fields': sorted(set().union(*(set(r) for r in stats))),
    'steps': len(groups), 'step_min': min(groups), 'step_max': max(groups),
    'candidates_per_step': dict(collections.Counter(map(len, groups.values()))),
    'rank_min': min(r['rank'] for r in stats), 'rank_max': max(r['rank'] for r in stats),
    'soft_reward_formula_matches_all': all(math.isclose(r['r_retrieval'], 1/(1+r['rank']), abs_tol=1e-12) for r in stats),
    'total_formula_lambda1_matches_all': all(math.isclose(r['r_total'], r['r_retrieval']+r['r_format'], abs_tol=1e-12) for r in stats),
    'format_reward_histogram': dict(collections.Counter(r['r_format'] for r in stats)),
    'has_tool_count': sum(r['has_tool'] for r in stats),
    'all_rewards_tied_steps': sum(len(set(r['r_retrieval'] for r in group)) == 1 for group in groups.values())}

code_paths = ['CUP/agent/grpo/scorer.py', 'CUP/agent/grpo/reward.py',
              'CUP/agent/grpo/train_grpo.py', 'CUP/agent/rollout.py']
result['source_versions'] = {}
with tarfile.open(ASSETS / 'agse-pre-stage-a-CUP-20261008.tar.gz') as archive:
    for relative in code_paths:
        current = (REPO / relative).read_bytes()
        old = archive.extractfile(relative).read()
        handoff = subprocess.check_output(['git', 'show', f'18fa341:{relative}'], cwd=REPO)
        result['source_versions'][relative] = {
            **file_info(REPO / relative), 'oct08_tar_sha256': hashlib.sha256(old).hexdigest(),
            'matches_oct08_tar': current == old,
            'matches_handoff_commit': current == handoff}

lora_dir = ARCHIVE / 'lora/ucm_main_lora'
lora_file = lora_dir / 'adapter_model.safetensors'
with safe_open(str(lora_file), framework='pt', device='cpu') as adapter:
    lora_shapes = {key: list(adapter.get_slice(key).get_shape()) for key in adapter.keys()}
result['lora'] = {'weights': file_info(lora_file),
                  'config': json.loads((lora_dir / 'adapter_config.json').read_text()),
                  'tensor_count': len(lora_shapes),
                  'parameter_count': sum(math.prod(shape) for shape in lora_shapes.values())}
base_dir = ASSETS / 'models/Qwen2.5-VL-7B-Instruct'
index_path = base_dir / 'model.safetensors.index.json'
index = json.loads(index_path.read_text())
shards = sorted(set(index['weight_map'].values()))
result['base_model'] = {'config': file_info(base_dir/'config.json'),
                         'index': file_info(index_path),
                         'shards': {name: {'exists': (base_dir/name).is_file(),
                                           'bytes': (base_dir/name).stat().st_size if (base_dir/name).is_file() else None}
                                    for name in shards}}
result['evidence_files'] = [file_info(ARCHIVE / relative) for relative in (
    'README.md', 'scripts/pipeline_domain_tool.sh', 'logs/grpo_domain_tool.log',
    'logs/pipeline_domain_tool.log')]
result['searched_local_checkpoints'] = sorted(
    str(p) for root in (ASSETS, REPO) for p in root.rglob('*.ckpt'))

for name, rows in [('val图库与本地资产映射.csv', val_map), ('E1样本复用核查.csv', e1_matches)]:
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
(OUT/'资产核查结果.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k: result[k] for k in ('feature_caches', 'cache_comparison', 'e1_overlap',
                                         'historical_stats', 'val_prior_checks')}, ensure_ascii=False), flush=True)

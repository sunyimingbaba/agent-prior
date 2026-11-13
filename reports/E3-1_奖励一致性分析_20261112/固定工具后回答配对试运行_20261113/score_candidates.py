#!/usr/bin/env python3
"""Pair original proxy rewards with frozen final-AGSE retrieval utilities."""
import csv
import hashlib
import io
import json
import time
from pathlib import Path

from pilot_common import ROOT, BASE, ASSETS, REBUILD, PLAN, sha, tensor_sha, dump, setup_torch, seed

torch = setup_torch()
import numpy as np
import pyarrow as pa
from PIL import Image
from prompt_clip import clip
from run_frozen import MyBackboneFinetuning
from vilt.modules.vilt_module import ViLTransformerSS
from agent.grpo.scorer import RetrievalScorer
from agent.grpo import reward

started = time.monotonic()
out = ROOT/'scoring'
out.mkdir(exist_ok=False)
assert json.loads((ROOT/'generation/verification.json').read_text())['ok']
records = [json.loads(line) for line in (ROOT/'generation/records.jsonl').read_text().splitlines()]
assert len(records) == 30
for item in PLAN['images']:
    records.append({'id': f'{item["row"]}_historical', 'role': 'historical', 'seed': None,
        'row': item['row'], 'image': item['image'], 'caption': item['caption'],
        'prior_file': f'generation/historical_priors/{item["row"]}.pt',
        'prior_sha256': item['historical_prior_sha256'], 'trajectory_text': None})
paths = [BASE/'CUP'/s for s in ['agent/grpo/scorer.py', 'agent/grpo/reward.py',
    'vilt/modules/vilt_module.py', 'prompt_clip/clip.py', 'prompt_clip/model.py', 'vilt/modules/vilt_utils.py']]
source_hashes = {str(p): sha(p) for p in paths}
clip_load = clip.load


def private_load(name, device='cpu', jit=False, download_root=None):
    if name == 'ViT-L/14':
        name = str(ASSETS/'models/ViT-L-14.pt')
    return clip_load(name, device='cpu', jit=jit, download_root=download_root)


clip.load = private_load


def load_retrieval(path, expected_sha, label):
    assert sha(path) == expected_sha
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    config = dict(saved['hyper_parameters']['config'])
    original_config = dict(config)
    config.update(load_path='', test_only=False)
    model = ViLTransformerSS(config)
    loaded = model.load_state_dict(saved['state_dict'], strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    model.eval().requires_grad_(False).cuda()
    assert model.use_agent_prior and model.prompt_length == 16
    dump(out/f'{label}_config.json', {'original': original_config, 'effective': config,
        'checkpoint_path': str(path), 'checkpoint_sha256': expected_sha, 'strict_load': True})
    return model


seed(0)
registry = json.loads((REBUILD/'scorer_resume_20261113/frozen_scorer_registry.json').read_text())
proxy_model = load_retrieval(Path(registry['selected_checkpoint']['path']),
    PLAN['proxy_reward']['scorer_sha256'], 'proxy')
cache = REBUILD/'proxy_gallery_20261113/grpo_val_cache_rebuilt_last50.pt'
assert sha(cache) == PLAN['proxy_reward']['gallery_sha256']
scorer = RetrievalScorer.__new__(RetrievalScorer)
scorer.pl, scorer.device = proxy_model, 'cuda'
scorer.val_arrow_path = str(ASSETS/'ucm/ucm_caption_karpathy_label_val.arrow')
scorer.cache_path = str(cache)
scorer.image_features, scorer.image_indices = scorer._load_or_build_cache()
assert scorer.image_indices == list(range(210)) and list(scorer.image_features.shape) == [210, 768]
utility_model = load_retrieval(ASSETS/'ucm_main_domain_tool.ckpt', PLAN['utility_model_sha256'], 'utility')
table = pa.ipc.open_file(str(ASSETS/'ucm/ucm_caption_karpathy_label_val.arrow')).read_all()
texts, text_index, raw_count = [], {}, 0
for row in range(len(table)):
    for caption in table['caption'][row].as_py():
        raw_count += 1
        if caption not in text_index:
            text_index[caption] = len(texts)
            texts.append(caption)
gallery_size = len(texts)
images, positives, quantity_indices = {}, {}, {}
for item in PLAN['images']:
    row = item['row']
    image = Image.open(io.BytesIO(table['image'][row].as_py())).convert('RGB')
    assert hashlib.sha256(image.tobytes()).hexdigest() == item['image_rgb_sha256']
    images[row] = utility_model.preprocess(image).unsqueeze(0).cuda()
    positives[row] = sorted({text_index[c] for c in table['caption'][row].as_py()})
    quantity_indices[row] = {}
    for pair in item['E1_pairs']:
        indices = []
        for caption in [pair['正确描述'], pair['错误描述']]:
            if caption not in text_index:
                text_index[caption] = len(texts)
                texts.append(caption)
            indices.append(text_index[caption])
        quantity_indices[row][pair['句式'][0]] = indices
tokens = clip.tokenize(texts, context_length=61)
token_groups = {}
for i, token_row in enumerate(tokens):
    token_groups.setdefault(tensor_sha(token_row), []).append(i)
dump(out/'text_gallery.json', {'raw_val_caption_count': raw_count,
    'unique_val_caption_count': gallery_size, 'all_encoded_text_count': len(texts),
    'text_order': texts, 'positive_gallery_columns': positives, 'quantity_text_columns': quantity_indices,
    'token_equivalent_raw_text_groups': [g for g in token_groups.values() if len(g) > 1]})
text_features = []
with torch.inference_mode():
    for start in range(0, len(tokens), PLAN['text_feature_batch_size']):
        text_features.append(utility_model.txt_embeds(tokens[start:start+PLAN['text_feature_batch_size']].cuda())['text_feats'].cpu())
text_features = torch.cat(text_features)
assert torch.isfinite(text_features).all()
text_gpu = text_features.cuda()
results, visual_features, proxy_features, actual_similarities, proxy_similarities = [], [], [], [], []
for record in records:
    row = record['row']
    prior_path = ROOT/record['prior_file']
    assert sha(prior_path) == record['prior_sha256']
    hidden = torch.load(prior_path, map_location='cpu', weights_only=True)
    assert list(hidden.shape) == [3584] and torch.isfinite(hidden).all()
    captured = []
    encode = proxy_model.model.encode_image

    def capture(*args, **kwargs):
        feature = encode(*args, **kwargs)
        captured.append(feature.detach().clone())
        return feature

    proxy_model.model.encode_image = capture
    try:
        rank, r_soft = scorer.score(hidden.float().tolist(), row)
    finally:
        proxy_model.model.encode_image = encode
    assert len(captured) == 1
    proxy = captured[0]/captured[0].norm(dim=1, keepdim=True)
    with torch.inference_mode():
        sims_proxy = (proxy @ scorer.image_features.cuda().T).cpu().flatten()
        visual = utility_model.img_embeds(images[row], agent_hidden=hidden.unsqueeze(0).cuda())['image_feats']
        scores = (visual @ text_gpu.T).cpu().flatten()
    assert all(torch.isfinite(x).all() for x in [proxy, visual, scores, sims_proxy])
    rank_independent = int(np.count_nonzero(scorer.image_features.numpy().astype(np.float64)
        @ proxy.cpu().numpy().astype(np.float64)[0] > float((scorer.image_features.numpy().astype(np.float64)
        @ proxy.cpu().numpy().astype(np.float64)[0])[row])))
    assert rank == rank_independent
    r_fmt = reward.r_format(record['trajectory_text']) if record['trajectory_text'] is not None else None
    total = reward.total_reward(record['trajectory_text'], rank, lam=1.0, mode='soft') if r_fmt is not None else None
    gallery_scores = scores[:gallery_size]
    positive = positives[row]
    negative = [i for i in range(gallery_size) if i not in positive]
    best_positive = float(gallery_scores[positive].max())
    best_negative = float(gallery_scores[negative].max())
    actual_rank = int((gallery_scores > best_positive).sum())
    assert actual_rank == sum(float(s) > best_positive for s in gallery_scores)
    scores64 = text_features[:gallery_size].numpy().astype(np.float64) @ visual.cpu().numpy().astype(np.float64)[0]
    actual_rank64 = int(np.count_nonzero(scores64 > scores64[positive].max()))
    e1_margins = {key: float(scores[columns[0]]-scores[columns[1]]) for key, columns in quantity_indices[row].items()}
    result = {**record, 'proxy_rank0': rank, 'proxy_rank1': rank+1,
        'proxy_rank64': rank_independent, 'r_retrieval': r_soft, 'r_format': r_fmt, 'r_total': total,
        'utility_rank0': actual_rank, 'utility_rank1': actual_rank+1, 'utility_rr': 1/(1+actual_rank),
        'utility_rank64': actual_rank64, 'utility_rank64_equal': actual_rank == actual_rank64,
        'utility_margin': best_positive-best_negative, 'best_positive_cosine': best_positive,
        'best_negative_cosine': best_negative, 'E1_margin_A': e1_margins['A'], 'E1_margin_B': e1_margins['B'],
        'utility_tied_at_best_positive': int((gallery_scores == best_positive).sum()),
        'h_end_tensor_sha256': tensor_sha(hidden)}
    results.append(result)
    visual_features.append(visual.cpu())
    proxy_features.append(proxy.cpu())
    actual_similarities.append(scores)
    proxy_similarities.append(sims_proxy)
    print('SCORED', len(results), '/33', record['id'], 'proxy_rank1', rank+1, 'utility_rank1', actual_rank+1, flush=True)
for result in results:
    reference = next(r for r in results if r['row'] == result['row'] and r['role'] == 'reference')
    for metric in ['utility_rr', 'utility_margin', 'E1_margin_A', 'E1_margin_B']:
        result[metric+'_delta'] = result[metric]-reference[metric]
for i, record in enumerate(results):
    if record['role'] != 'repeat':
        continue
    j = next(j for j, other in enumerate(results) if other['row'] == record['row'] and other['role'] == 'candidate' and other['seed'] == record['seed'])
    assert torch.equal(visual_features[i], visual_features[j]) and torch.equal(proxy_features[i], proxy_features[j])
    for key in ['r_retrieval', 'utility_rr', 'utility_margin', 'E1_margin_A', 'E1_margin_B']:
        assert record[key] == results[j][key]
dump(out/'scores.json', results)
flat_fields = [k for k, value in results[0].items() if not isinstance(value, (dict, list))]
with (out/'scores.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=flat_fields, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    writer.writerows(results)
torch.save({'record_ids': [r['id'] for r in results], 'text_tokens': tokens,
    'text_features': text_features, 'image_features': torch.cat(visual_features),
    'proxy_features': torch.cat(proxy_features), 'actual_similarities': torch.stack(actual_similarities),
    'proxy_similarities': torch.stack(proxy_similarities)}, out/'scoring_tensors.pt')

# Replay six E1 historical quantity comparisons on CPU with identical inputs.
utility_model.cpu()
references = json.loads((ROOT/'E1_reference_scores.json').read_text())
cross_checks = []
with torch.inference_mode():
    for item in PLAN['images']:
        row = item['row']
        hidden = torch.load(ROOT/f'generation/historical_priors/{row}.pt', map_location='cpu', weights_only=True).unsqueeze(0)
        feature = utility_model.img_embeds(images[row].cpu(), agent_hidden=hidden)['image_feats']
        for pair in item['E1_pairs']:
            label = pair['句式'][0]
            text_feature = utility_model.txt_embeds(clip.tokenize([pair['正确描述'], pair['错误描述']], context_length=61))['text_feats']
            values = (feature @ text_feature.T).flatten().tolist()
            old = next(r for r in references['records'] if r['image'] == item['image'] and r['wording'] == pair['句式'])
            error = max(abs(values[0]-old['correct_score']), abs(values[1]-old['wrong_score']))
            historical_result = next(r for r in results if r['row'] == row and r['role'] == 'historical')
            gpu_cpu_margin_error = abs(historical_result['E1_margin_'+label]-(values[0]-values[1]))
            assert max(error, gpu_cpu_margin_error) <= PLAN['cross_device_E1_reference_tolerance']
            cross_checks.append({'row': row, 'wording': label, 'historical_CPU_max_abs_score_error': error,
                'GPU_cached_text_vs_CPU_direct_margin_error': gpu_cpu_margin_error})
assert all(not p.requires_grad and p.grad is None for p in utility_model.parameters())
assert all(not p.requires_grad and p.grad is None for p in proxy_model.parameters())
assert all(sha(path) == digest for path, digest in source_hashes.items())
dump(out/'verification.json', {'ok': True, 'records': len(results), 'strict_checkpoint_load': True,
    'frozen_eval_no_grad': True, 'source_files_unchanged': source_hashes,
    'all_features_and_similarities_finite': True, 'proxy_rank64_all_equal': True,
    'utility_rank64_equal_count': sum(r['utility_rank64_equal'] for r in results),
    'repeated_candidates_features_and_scores_bitwise_equal': True, 'E1_cross_checks': cross_checks,
    'plan_sha256': sha(ROOT/'plan.json'), 'scorer_script_sha256': sha(__file__),
    'text_gallery_sha256': sha(out/'text_gallery.json'), 'gallery_size': gallery_size,
    'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'elapsed_seconds': time.monotonic()-started})
print('SCORING_COMPLETE', flush=True)

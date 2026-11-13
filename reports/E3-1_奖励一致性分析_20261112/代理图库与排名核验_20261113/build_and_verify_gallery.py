#!/usr/bin/env python3
"""Build the original RFPO gallery with the frozen reconstructed scorer.

Original RetrievalScorer methods are called without editing project sources.
The wrapper only provides strict loading, private asset paths and audit records.
"""
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/private/AGSE_E3_1_rebuild_20261112/proxy_gallery_20261113')
BASE = Path('/private/AGSE_E1_20261110/legacy')
REBUILD = ROOT.parent
PRIOR_RUN = REBUILD/'plain_prior_full_20261112/run_01'
REGISTRY = REBUILD/'scorer_resume_20261113/frozen_scorer_registry.json'
INPUTS = ROOT/'input_model_manifest.json'
ARROW = BASE/'assets/ucm/ucm_caption_karpathy_label_val.arrow'
CACHE = ROOT/'grpo_val_cache_rebuilt_last50.pt'
ANCHORS = [0, 105, 209]  # Fixed before scoring: first, middle, last Arrow row.

for name in ['scratch', 'scratch/hf', 'scratch/matplotlib', 'scratch/cache']:
    (ROOT/name).mkdir(parents=True, exist_ok=True)
os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'TMPDIR': str(ROOT/'scratch'), 'HF_HOME': str(ROOT/'scratch/hf'),
    'MPLCONFIGDIR': str(ROOT/'scratch/matplotlib'),
    'XDG_CACHE_HOME': str(ROOT/'scratch/cache'),
    'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
sys.path.insert(0, str(BASE/'CUP'))
import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from prompt_clip import clip
from run_frozen import MyBackboneFinetuning  # checkpoint pickle reference
from vilt.modules.vilt_module import ViLTransformerSS
from agent.grpo.scorer import RetrievalScorer
from agent.grpo import reward


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


started = time.time()
assert not CACHE.exists(), 'Preserve existing cache; choose a new output directory.'
torch.manual_seed(0)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
registry = json.loads(REGISTRY.read_text(encoding='utf-8'))
checkpoint = Path(registry['selected_checkpoint']['path'])
assert registry['status'] == 'frozen'
assert sha(checkpoint) == registry['selected_checkpoint']['sha256']
manifest = json.loads(INPUTS.read_text(encoding='utf-8'))
assert sha(ARROW) == manifest['arrow_sha256']['val']
assert sha(BASE/'assets/models/ViT-L-14.pt') == 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
source_paths = [BASE/'CUP'/name for name in [
    'agent/grpo/scorer.py', 'agent/grpo/reward.py', 'vilt/modules/vilt_module.py',
    'prompt_clip/clip.py', 'prompt_clip/model.py', 'vilt/modules/vilt_utils.py']]
sources = {str(p): sha(p) for p in source_paths}
records = [json.loads(line) for line in (PRIOR_RUN/'records.jsonl').read_text(encoding='utf-8').splitlines()]
records = {r['row']: r for r in records if r['split'] == 'val'}
assert sorted(records) == list(range(210))
table = pa.ipc.open_file(str(ARROW)).read_all()
assert len(table) == 210
expected = {r['row']: r for r in manifest['rows'] if r['split'] == 'val'}
mapping = []
for row in range(len(table)):
    raw = table['image'][row].as_py()
    image = Image.open(io.BytesIO(raw)).convert('RGB')
    item = {'split': 'val', 'arrow_row': row, 'feature_row': row,
            'similarity_target_column': row, 'image_index': row,
            'image': table['path'][row].as_py(), 'first_caption': table['caption'][row].as_py()[0],
            'image_bytes_sha256': hashlib.sha256(raw).hexdigest(),
            'image_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest()}
    reference = expected[row]
    assert item['image'] == reference['image'] == records[row]['image']
    assert item['first_caption'] == reference['caption'] == records[row]['caption']
    for key in ['image_bytes_sha256', 'image_rgb_sha256']:
        assert item[key] == reference[key] == records[row][key]
    prior_path = PRIOR_RUN/records[row]['prior_file']
    assert sha(prior_path) == records[row]['prior_file_sha256']
    item.update(prior_file=str(prior_path), prior_file_sha256=sha(prior_path))
    mapping.append(item)
dump(ROOT/'row_mapping.json', mapping)
with (ROOT/'row_mapping.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(mapping[0]))
    writer.writeheader()
    writer.writerows(mapping)

clip_load = clip.load


def private_clip_load(name, device='cpu', jit=False, download_root=None):
    if name == 'ViT-L/14':
        name = str(BASE/'assets/models/ViT-L-14.pt')
    return clip_load(name, device='cpu', jit=jit, download_root=download_root)


clip.load = private_clip_load
saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
config = dict(saved['hyper_parameters']['config'])
assert config['load_path'] == '' and not config['test_only']
model = ViLTransformerSS(config)
loaded = model.load_state_dict(saved['state_dict'], strict=True)
assert not loaded.missing_keys and not loaded.unexpected_keys
model.eval().requires_grad_(False).cuda()
del saved
assert not model.training and all(not p.requires_grad for p in model.parameters())
dump(ROOT/'resolved_config.json', config)

# Use original methods directly, supplying an already strictly loaded model.
scorer = RetrievalScorer.__new__(RetrievalScorer)
scorer.pl, scorer.device = model, 'cuda'
scorer.val_arrow_path, scorer.cache_path = str(ARROW), str(CACHE)
scorer.image_features, scorer.image_indices = scorer._load_or_build_cache()
features = scorer.image_features
assert list(features.shape) == [210, 768] and features.dtype == torch.float32
assert scorer.image_indices == list(range(210))
assert bool(torch.isfinite(features).all())
norm_error = float((features.norm(dim=1)-1).abs().max())
assert norm_error < 1e-6
reload_features, reload_indices = scorer._load_or_build_cache()
assert torch.equal(features, reload_features) and scorer.image_indices == reload_indices

# Test strict-greater ranking and ties with hand-specified similarity arrays.
tie_checks = []
for values, target, expected_rank in [([.8, .8, .2], 0, 0),
    ([.8, .8, .2], 2, 2), ([.1, .3, .2], 2, 1), ([.4, .4, .4], 1, 0)]:
    sims = torch.tensor([values], device='cuda')
    rank = int((sims > sims[0, target]).sum().item())
    assert rank == expected_rank and reward.r_retrieval(rank) == 1/(1+expected_rank)
    tie_checks.append({'similarities': values, 'target': target, 'rank0': rank,
                       'soft_reward': reward.r_retrieval(rank)})
assert reward.r_format('<answer>ok</answer>') == .2
assert reward.r_format('ok') == 0

anchor_feature_checks, smoke, tensors = [], [], {}
for anchor_position, row in enumerate(ANCHORS):
    raw = table['image'][row].as_py()
    image = model.preprocess(Image.open(io.BytesIO(raw)).convert('RGB')).unsqueeze(0).cuda()
    with torch.inference_mode():
        repeated = scorer._encode_image(image).cpu()
        repeated = repeated / repeated.norm(dim=1, keepdim=True)
    delta = float((repeated[0]-features[row]).abs().max())
    assert torch.equal(repeated[0], features[row])
    anchor_feature_checks.append({'row': row, 'max_abs_delta': delta, 'bitwise_equal': True})
    other = ANCHORS[(anchor_position+1) % len(ANCHORS)]
    for label, source_row in [('own_plain', row), ('other_plain_control', other), ('zero_control', None)]:
        h = torch.zeros(3584) if source_row is None else torch.load(
            PRIOR_RUN/records[source_row]['prior_file'], map_location='cpu', weights_only=True).float()
        assert list(h.shape) == [3584] and bool(torch.isfinite(h).all())
        captured = []
        encode = model.model.encode_image

        def capture(*args, **kwargs):
            out = encode(*args, **kwargs)
            captured.append(out.detach().clone())
            return out

        model.model.encode_image = capture
        try:
            rank, r_soft = scorer.score(h.tolist(), row)
        finally:
            model.model.encode_image = encode
        assert len(captured) == 1
        query = captured[0]
        query = query / query.norm(dim=1, keepdim=True)
        with torch.inference_mode():
            sims_gpu = (query @ features.cuda().T).cpu().squeeze(0)
        query_np = query.cpu().numpy().astype(np.float64)[0]
        sims64 = features.numpy().astype(np.float64) @ query_np
        independent_rank = int(np.count_nonzero(sims64 > sims64[row]))
        assert rank == independent_rank
        ties = int((sims_gpu == sims_gpu[row]).sum())
        nonself = (sims_gpu-sims_gpu[row]).abs()
        nonself[row] = float('inf')
        text = '' if source_row is None else records[source_row]['generated_text']
        r_fmt = reward.r_format(text)
        total = reward.total_reward(text, rank, lam=1.0, mode='soft')
        assert r_soft == reward.r_retrieval(rank) and total == r_fmt+r_soft
        repeat_rank, repeat_reward = scorer.score(h.tolist(), row)
        assert (rank, r_soft) == (repeat_rank, repeat_reward)
        identifier = f'{row}:{label}'
        tensors[identifier] = {'h_end_float32': h, 'normalized_proxy_feature': query.cpu(),
                               'similarities_float32': sims_gpu}
        smoke.append({'row': row, 'image': mapping[row]['image'], 'case': label,
            'source_prior_row': source_row, 'h_end_sha256_float32': tensor_sha(h),
            'generated_text': text, 'rank0': rank, 'rank1': rank+1,
            'independent_float64_rank0': independent_rank, 'r_retrieval': r_soft,
            'r_format': r_fmt, 'lambda': 1.0, 'r_total': total,
            'tied_target_columns_including_self': ties,
            'minimum_other_column_similarity_gap': float(nonself.min()),
            'cpu64_gpu32_max_abs_similarity_delta': float(np.max(np.abs(sims64-sims_gpu.numpy()))),
            'repeated_rank_and_reward_equal': True})
torch.save(tensors, ROOT/'smoke_tensors.pt')
dump(ROOT/'smoke_scores.json', smoke)
with (ROOT/'smoke_scores.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(smoke[0]))
    writer.writeheader()
    writer.writerows(smoke)
assert all(p.grad is None and not p.requires_grad for p in model.parameters())
assert not model.training
assert all(sha(Path(path)) == digest for path, digest in sources.items())
result = {'ok': True, 'stage': 'gallery_and_reward_numerical_verification',
    'checkpoint': registry['selected_checkpoint'], 'arrow': str(ARROW), 'arrow_sha256': sha(ARROW),
    'input_manifest_sha256': sha(INPUTS), 'prior_records_sha256': sha(PRIOR_RUN/'records.jsonl'),
    'gallery_cache': str(CACHE), 'gallery_cache_sha256': sha(CACHE),
    'gallery_tensor_shape': list(features.shape), 'gallery_dtype': str(features.dtype),
    'all_210_rows_match_input_manifest': True, 'all_210_prior_file_hashes_match': True,
    'feature_indices_equal_arrow_rows': True, 'max_norm_error': norm_error,
    'save_reload_bitwise_equal': True, 'anchor_feature_recomputations': anchor_feature_checks,
    'rank_tie_unit_checks': tie_checks, 'smoke_count': len(smoke), 'smoke_anchor_rows': ANCHORS,
    'original_vs_independent_rank_equal': True, 'frozen_eval_no_grad': True,
    'gallery_path': 'RetrievalScorer._load_or_build_cache -> _encode_image: actual RGB image + zero visual prompt',
    'proxy_path': 'RetrievalScorer.score: h_end -> agent_proj -> fused visual prompt -> zero image -> encode_image',
    'source_files_unchanged': sources, 'wrapper_sha256': sha(__file__),
    'preprocess': repr(model.preprocess), 'torch': torch.__version__,
    'gpu': torch.cuda.get_device_name(), 'tf32': False,
    'mount': subprocess.check_output(['findmnt','-T',str(ROOT),'-n','-o','SOURCE,TARGET,FSTYPE'],text=True).strip(),
    'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'elapsed_seconds': time.time()-started,
    'scope': '3 fixed image rows, each with own prior, other-image prior and zero numerical controls; downstream utility pending'}
dump(ROOT/'verification.json', result)
print(json.dumps({'ok': True, 'cache_sha256': result['gallery_cache_sha256'],
                  'shape': result['gallery_tensor_shape'], 'smoke_count': len(smoke),
                  'elapsed_seconds': result['elapsed_seconds']}, ensure_ascii=False), flush=True)

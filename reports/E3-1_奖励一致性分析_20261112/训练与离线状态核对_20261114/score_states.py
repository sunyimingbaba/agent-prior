"""Same frozen proxy/final models; reuse the original saved text gallery."""
import io
import json
import shutil
import time

from audit_common import ROOT, BASE, ASSETS, PLAN, SOURCE, dump, sha, setup_torch, seed
torch = setup_torch()
import numpy as np
import pyarrow as pa
from PIL import Image
from prompt_clip import clip
from run_frozen import MyBackboneFinetuning
from vilt.modules.vilt_module import ViLTransformerSS
from agent.grpo.scorer import RetrievalScorer

start = time.monotonic()
assert json.loads((ROOT/'generation_verification.json').read_text())['ok']
records = json.loads((ROOT/'state_records.json').read_text())
out = ROOT/'scoring'
out.mkdir(exist_ok=False)
verify = ROOT/'verification_inputs'
verify.mkdir()
proxy_cache = SOURCE.parent/'proxy_gallery_20261113/grpo_val_cache_rebuilt_last50.pt'
assert sha(proxy_cache) == PLAN['source_models']['gallery_sha256']
for source, name in [(proxy_cache, 'proxy_gallery.pt'),
    (SOURCE/'scoring/scoring_tensors.pt', 'original_scoring_tensors.pt'),
    (SOURCE/'scoring/text_gallery.json', 'text_gallery.json')]:
    shutil.copy2(source, verify/name)
gallery = json.loads((SOURCE/'scoring/text_gallery.json').read_text())
old = torch.load(SOURCE/'scoring/scoring_tensors.pt', map_location='cpu', weights_only=True)
text_features = old['text_features']
text_gpu = text_features.cuda()
cache = torch.load(proxy_cache, map_location='cpu', weights_only=True)
assert cache['image_indices'] == list(range(210))
original_clip_load = clip.load


def private_clip_load(name, device='cpu', jit=False, download_root=None):
    return original_clip_load(str(ASSETS/'models/ViT-L-14.pt') if name == 'ViT-L/14' else name,
                              device='cpu', jit=jit, download_root=download_root)


clip.load = private_clip_load


def load_model(path, expected, name):
    assert sha(path) == expected
    ckpt = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    original = dict(ckpt['hyper_parameters']['config'])
    config = dict(original)
    config.update(load_path='', test_only=False)
    model = ViLTransformerSS(config)
    loaded = model.load_state_dict(ckpt['state_dict'], strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    model.eval().requires_grad_(False).cuda()
    dump(out/(name+'_config.json'), {'checkpoint_path': str(path), 'checkpoint_sha256': expected,
                                    'original': original, 'effective': config})
    return model


seed(0)
registry = json.loads((SOURCE.parent/'scorer_resume_20261113/frozen_scorer_registry.json').read_text())
proxy_model = load_model(registry['selected_checkpoint']['path'], PLAN['source_models']['scorer_sha256'], 'proxy')
utility_model = load_model(ASSETS/'ucm_main_domain_tool.ckpt', PLAN['utility_model_sha256'], 'utility')
scorer = RetrievalScorer.__new__(RetrievalScorer)
scorer.pl, scorer.device = proxy_model, 'cuda'
scorer.image_features, scorer.image_indices = cache['image_features'], cache['image_indices']
table = pa.ipc.open_file(str(ASSETS/'ucm/ucm_caption_karpathy_label_val.arrow')).read_all()
images = {row: utility_model.preprocess(Image.open(io.BytesIO(table['image'][row].as_py())).convert('RGB')).unsqueeze(0).cuda()
          for row in PLAN['rows']}
scores, visual_features, proxy_features, similarities, proxy_similarities = [], [], [], [], []
replays = []
for n, record in enumerate(records, 1):
    path = ROOT/record['state_file']
    assert sha(path) == record['state_file_sha256']
    hidden = torch.load(path, map_location='cpu', weights_only=True)
    captured = []
    encode = proxy_model.model.encode_image

    def capture(*args, **kwargs):
        feature = encode(*args, **kwargs)
        captured.append(feature.detach().clone())
        return feature

    proxy_model.model.encode_image = capture
    try:
        rank, reward = scorer.score(hidden.float().tolist(), record['row'])
    finally:
        proxy_model.model.encode_image = encode
    assert len(captured) == 1
    proxy = captured[0]/captured[0].norm(dim=1, keepdim=True)
    with torch.no_grad():
        ps = (proxy@cache['image_features'].cuda().T).cpu().flatten()
        visual = utility_model.img_embeds(images[record['row']], agent_hidden=hidden.unsqueeze(0).cuda())['image_feats']
        sims = (visual@text_gpu.T).cpu().flatten()
    assert all(torch.isfinite(x).all() for x in [hidden, visual, proxy, ps, sims])
    positive = gallery['positive_gallery_columns'][str(record['row'])]
    negatives = [i for i in range(gallery['unique_val_caption_count']) if i not in positive]
    best_positive, best_negative = float(sims[positive].max()), float(sims[negatives].max())
    actual_rank = int((sims[:gallery['unique_val_caption_count']]>best_positive).sum())
    item = {**record, 'proxy_rank0': rank, 'r_retrieval': reward, 'utility_rank0': actual_rank,
        'utility_rr': 1/(1+actual_rank), 'utility_margin': best_positive-best_negative}
    for label, columns in gallery['quantity_text_columns'][str(record['row'])].items():
        item['E1_margin_'+label] = float(sims[columns[0]]-sims[columns[1]])
    if record['variant'] == 'saved_offline':
        source = next(x['source_score'] for x in PLAN['records'] if x['source_id'] == record['source_id'])
        error = max(abs(source[k]-item[k]) for k in ['utility_margin','E1_margin_A','E1_margin_B'])
        assert source['proxy_rank0'] == rank and source['utility_rank0'] == actual_rank and error < 1e-6
        replays.append({'source_id': record['source_id'], 'margin_max_abs_error': error, 'ranks_equal': True})
    scores.append(item)
    proxy_features.append(proxy.cpu()); visual_features.append(visual.cpu())
    similarities.append(sims); proxy_similarities.append(ps)
    dump(ROOT/'progress.json', {'phase': 'scoring', 'completed': n, 'total': len(records)})
    print('SCORED', n, '/', len(records), record['id'], flush=True)
dump(out/'scores.json', scores)
torch.save({'record_ids': [x['id'] for x in scores], 'text_features': text_features,
    'image_features': torch.cat(visual_features), 'proxy_features': torch.cat(proxy_features),
    'actual_similarities': torch.stack(similarities), 'proxy_similarities': torch.stack(proxy_similarities)}, out/'scoring_tensors.pt')
dump(out/'verification.json', {'ok': True, 'states': len(scores), 'original_saved_replays': replays,
    'text_features_reused_bitwise': True, 'strict_checkpoint_load': True, 'frozen_eval_no_grad': True,
    'verification_input_sha256': {p.name: sha(p) for p in verify.iterdir()},
    'elapsed_seconds': time.monotonic()-start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
print('SCORING_COMPLETE', flush=True)

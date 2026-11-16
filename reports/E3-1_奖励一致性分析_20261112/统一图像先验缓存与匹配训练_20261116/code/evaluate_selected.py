"""Lock validation-selected checkpoints first; then evaluate all declared test arms."""
import datetime
import io
import json
import gc
from pathlib import Path
import torch
import pyarrow as pa
from PIL import Image
from milestone_common import ROOT, ASSETS, setup, private_clip
from stage_io import read, dump, sha
from matched_common_v3 import TRAIN_PLAN, CACHE_PLAN, PARENT, load_initial, install_arm, prepare_gallery, evaluate, state_hash

setup()
for n, h in read(ROOT/'evaluation_code_hashes.json').items():
    assert sha(ROOT/n) == h, n
selection = {}
initials = []
for arm in TRAIN_PLAN['arms']:
    out = ROOT/'training'/arm
    result = read(out/'training_result.json')
    assert result['ok'] and result['optimizer_steps'] == TRAIN_PLAN['optimizer_steps_per_arm']
    vals = [(e, read(out/f'validation/epoch_{e:02d}/metrics.json')) for e in range(1, TRAIN_PLAN['epochs']+1)]
    epoch, metrics = max(vals, key=lambda x: (x[1]['mR'], -x[0]))
    cp = result['checkpoints']['validation_best']
    assert epoch == cp['epoch'] and sha(ROOT/cp['path']) == cp['sha256']
    selection[arm] = {'epoch': epoch, 'val_mR': metrics['mR'], 'checkpoint': cp}
    initials.append(read(out/'initialization.json'))
assert all(x == initials[0] for x in initials)
lock = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'selection_metric': 'unrounded val210 mR; ties earliest epoch', 'arms': selection,
        'all_four_initializations_identical': True, 'test_metrics_available_at_selection': False}
assert not (ROOT/'selection_lock.json').exists()
dump(ROOT/'selection_lock.json', lock)
summaries = {}
quantity = []
cases = read(PARENT/'inputs/evaluation_cases.json')
qgallery = read(PARENT/'features/gallery.json')
words = {w: i+1 for i, w in enumerate('one two three four five six seven'.split())}
for arm in TRAIN_PLAN['arms']:
    model, _ = load_initial()
    checkpoint = torch.load(ROOT/selection[arm]['checkpoint']['path'], map_location='cpu', weights_only=False)
    assert checkpoint['matched_readout']['arm'] == arm
    assert not any(n.startswith('model.') for n in checkpoint['state_dict'])
    missing = model.load_state_dict(checkpoint['state_dict'], strict=False)
    assert not missing.unexpected_keys and all(n.startswith('model.') for n in missing.missing_keys)
    model.eval().requires_grad_(False).cuda()
    before = state_hash(model.state_dict().items())
    hooks = install_arm(model, arm)
    gallery = prepare_gallery(model, 'test')
    if not (ROOT/'test/gallery.json').exists():
        dump(ROOT/'test/gallery.json', {k: v for k, v in gallery.items() if k not in ['images','hidden','tokens']})
    summaries[arm] = evaluate(model, gallery, ROOT/'test'/arm)
    if arm in ['native', 'calibrated_own']:
        shared = next(x for x in read(PARENT/'states/records.json') if x['family'] == 'balanced32'
                      and x['image'] == '102.tif' and x['role'] == 'greedy')
        sh = torch.load(PARENT/shared['state_file'], map_location='cpu', weights_only=True)
        handle = model.agent_proj.register_forward_pre_hook(lambda module, args: (sh.to(args[0]).expand(args[0].shape[0], -1),))
        label = arm+'_same_checkpoint_shared'
        summaries[label] = evaluate(model, gallery, ROOT/'test'/label)
        handle.remove()
    # Quantity labels are the unchanged 63 train +22 val cases. After full readout
    # training, train cases are explicitly reported as training diagnostics.
    clip = private_clip()
    tokens = clip.tokenize(qgallery['quantity_text_order'], context_length=61, truncate=True)
    with torch.no_grad():
        text = model.txt_embeds(tokens.cuda())['text_feats'].cpu()
        fs = []
        tables = {s: pa.ipc.open_file(str(ASSETS/f'ucm/ucm_caption_karpathy_label_{s}.arrow')).read_all()
                  for s in ['train','val']}
        for c in cases:
            table = tables[c['split']]
            assert table['path'][c['row']].as_py() == c['image']
            img = Image.open(io.BytesIO(table['image'][c['row']].as_py())).convert('RGB')
            h = torch.load(ROOT/f"cache/priors/ucm_caption_karpathy_label_{c['split']}/{c['row']}.pt",map_location='cpu',weights_only=True)
            fs.append(model.img_embeds(model.preprocess(img)[None].cuda(),agent_hidden=h[None].cuda())['image_feats'].cpu())
        f = torch.cat(fs)
        scores = f@text.T
    qout = ROOT/'quantity'/arm
    qout.mkdir(parents=True)
    torch.save({'images': [c['image'] for c in cases], 'image_features': f, 'text_features': text,
                'similarities': scores, 'tokens': tokens}, qout/'tensors.pt')
    for i, c in enumerate(cases):
        row = {'arm': arm, 'image': c['image'], 'split': c['split'], 'actual_count': c['actual_count']}
        for w in ['A','B']:
            positive, negative = qgallery['quantity_columns'][c['image']][w]
            prefix = qgallery['quantity_text_order'][positive].rsplit(' ',1)[0]
            columns = {words[s.rsplit(' ',1)[1][:-1]]: j for j, s in enumerate(qgallery['quantity_text_order'])
                       if s.rsplit(' ',1)[0] == prefix}
            ns = [float(scores[i,j]) for count,j in columns.items() if count != c['actual_count']]
            value = float(scores[i, positive])
            row.update({w+'_pair_correct': value > float(scores[i, negative]),
                        w+'_hard_correct': value > max(ns), w+'_hard_margin': value-max(ns)})
        quantity.append(row)
    assert state_hash(model.state_dict().items()) == before
    for h in hooks:
        h.remove()
    del model, checkpoint, gallery
    gc.collect()
    torch.cuda.empty_cache()
dump(ROOT/'quantity/records.json', quantity)
dump(ROOT/'evaluation_result.json', {'ok': True, 'test_summary': summaries, 'test_arms': len(summaries),
     'test_image_encodings': 210*len(summaries), 'quantity_images': 85, 'quantity_arm_scores': len(quantity),
     'readout_updates_during_evaluation': 0, 'selection_lock_sha256': sha(ROOT/'selection_lock.json')})
print('ALL_SELECTED_CHECKPOINT_EVALUATIONS_COMPLETE', json.dumps(summaries), flush=True)

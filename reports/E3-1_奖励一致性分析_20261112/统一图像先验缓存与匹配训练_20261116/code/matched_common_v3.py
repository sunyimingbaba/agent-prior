"""Matched readout initialization, explicit hooks, original data/loss and retrieval rules."""
import hashlib
import io
import json
import sys
from pathlib import Path
import numpy as np
import torch
import pyarrow as pa
from PIL import Image
from milestone_common import ROOT, PLAN as CACHE_PLAN, BASE, ASSETS, setup, seed, private_clip
from stage_io import read, dump, sha

TRAIN_PLAN = read(ROOT/'training_plan_v2.json')
PARENT = Path(CACHE_PLAN['parent_root'])


def state_hash(named):
    h = hashlib.sha256()
    for n, p in sorted(named):
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def configure_private_clip():
    private_clip()
    import model.clip as dataset_clip
    original = dataset_clip.load
    if not getattr(original, '_matched_private', False):
        def load(name, device='cpu', jit=False, download_root=None):
            return original(str(ASSETS/'models/ViT-L-14.pt') if name == 'ViT-L/14' else name,
                            device='cpu', jit=jit, download_root=download_root)
        load._matched_private = True
        dataset_clip.load = load


def load_initial(cls=None):
    configure_private_clip()
    from run_frozen import MyBackboneFinetuning
    from vilt.modules.vilt_module import ViLTransformerSS
    setattr(sys.modules['__main__'], 'MyBackboneFinetuning', MyBackboneFinetuning)
    assert sha(ASSETS/'ucm_main_domain_tool.ckpt') == CACHE_PLAN['utility_model_sha256']
    saved = torch.load(ASSETS/'ucm_main_domain_tool.ckpt', map_location='cpu', weights_only=False, mmap=True)
    config = dict(saved['hyper_parameters']['config'])
    config.update(current_epoch=0, load_path='', test_only=False, resume_from=None, max_epoch=TRAIN_PLAN['epochs'],
                  max_steps=TRAIN_PLAN['optimizer_steps_per_arm'], get_recall_metric=False,
                  learning_rate=TRAIN_PLAN['learning_rate'], warmup_steps=TRAIN_PLAN['warmup_fraction'],
                  num_gpus=1, num_nodes=1, precision=32, num_workers=4,
                  batch_size=256, per_gpu_batchsize=32,
                  data_root=str(ASSETS/'ucm'), agent_cache_dir=str(ROOT/'cache/priors'),
                  use_agent_prior=True, log_dir=str(ROOT/'training/logs'),
                  save_image_path=str(ROOT/'scratch/images'))
    seed(0)
    model = (cls or ViLTransformerSS)(config)
    model.load_state_dict(saved['state_dict'], strict=True)
    assert state_hash(model.state_dict().items()) == TRAIN_PLAN['initial_model_state_hash']
    for n, p in model.named_parameters():
        p.requires_grad_(not n.startswith('model.'))
    return model, config


def install_arm(model, arm):
    assert arm in TRAIN_PLAN['arms']
    pt = torch.load(PARENT/'path/tensors.pt', map_location='cpu', weights_only=True)
    offset = pt['offsets']['balanced32'].reshape(-1)
    assert offset.shape == (1024,)
    shared_record = next(x for x in read(PARENT/'states/records.json')
                         if x['family'] == 'balanced32' and x['image'] == '102.tif' and x['role'] == 'greedy')
    shared = torch.load(PARENT/shared_record['state_file'], map_location='cpu', weights_only=True)
    handles = []
    if arm in ['calibrated_shared', 'learned_visual_only']:
        def share(module, args):
            return (shared.to(args[0]).expand(args[0].shape[0], -1),)
        handles.append(model.agent_proj.register_forward_pre_hook(share))
    if arm != 'native':
        def shift(module, args):
            return (args[0]+offset.to(args[0]),)
        handles.append(model.agent_proj[1].register_forward_pre_hook(shift))
    if arm == 'learned_visual_only':
        handles.append(model.cluster_mapping.register_forward_hook(
            lambda module, args, out: torch.ones_like(out)))
    return handles


def original_train_dataset(config):
    from vilt.datasets import UCMCaptionKarpathyDataset
    return UCMCaptionKarpathyDataset(config['data_root'], config['train_transform_keys'],
          image_size=config['image_size'], split='train', max_text_len=config['max_text_len'],
          prompt_length=config['prompt_length'], draw_false_image=0, draw_false_text=0,
          image_only=False, use_agent_prior=True, agent_cache_dir=config['agent_cache_dir'])


def prepare_gallery(model, split):
    assert split in ['val', 'test']
    if split == 'test':
        assert (ROOT/'selection_lock.json').exists(), 'Seal every val-selected checkpoint before test access'
    path = ASSETS/f'ucm/ucm_caption_karpathy_label_{split}.arrow'
    table = pa.ipc.open_file(str(path)).read_all()
    assert table.num_rows == 210
    images, hidden, names, captions, owners = [], [], [], [], []
    for row in range(table.num_rows):
        img = Image.open(io.BytesIO(table['image'][row].as_py())).convert('RGB')
        images.append(model.preprocess(img))
        hidden.append(torch.load(ROOT/f'cache/priors/ucm_caption_karpathy_label_{split}/{row}.pt',
                                 map_location='cpu', weights_only=True))
        names.append(table['path'][row].as_py())
        cs = table['caption'][row].as_py()
        captions.extend(cs)
        owners.extend([row]*len(cs))
    clip = private_clip()
    tokens = clip.tokenize(captions, context_length=61, truncate=True)
    groups, ids = {}, []
    for row in tokens.tolist():
        key = tuple(row)
        if key not in groups:
            groups[key] = len(groups)
        ids.append(groups[key])
    return {'images': torch.stack(images), 'hidden': torch.stack(hidden), 'tokens': tokens,
            'names': names, 'captions': captions, 'owners': owners, 'groups': ids,
            'split': split, 'arrow_sha256': sha(path)}


def retrieval_metrics(scores, gallery):
    from vilt.modules.objectives import tr_top_k, ir_top_k
    groups = np.array(gallery['groups'])
    owners = np.array(gallery['owners'])
    n = len(gallery['names'])
    top_i = scores.topk(10, dim=1).indices
    top_t = scores.topk(10, dim=0).indices
    positive = {g: set(owners[groups == g].tolist()) for g in set(groups)}
    own_groups = [set(groups[owners == i].tolist()) for i in range(n)]
    ih = np.array([[any(groups[j] in own_groups[i] for j in top_i[i, :k].tolist())
                    for k in [1, 5, 10]] for i in range(n)], dtype=float)
    th = np.array([[any(i in positive[groups[j]] for i in top_t[:k, j].tolist())
                    for k in [1, 5, 10]] for j in range(len(owners))], dtype=float)
    correspondence = {str(g): sorted(v) for g, v in positive.items()}
    rep = torch.tensor(groups)
    for j, k in enumerate([1, 5, 10]):
        assert abs(float(tr_top_k(torch.arange(n)[None], correspondence, rep[top_i[:, :k]], k))-ih[:, j].mean()) < 1e-6
        assert abs(float(ir_top_k(rep[:, None], correspondence, top_t[:k], k))-th[:, j].mean()) < 1e-6
    values = np.r_[ih.mean(0), th.mean(0)]
    keys = ['I2T_R1', 'I2T_R5', 'I2T_R10', 'T2I_R1', 'T2I_R5', 'T2I_R10']
    metrics = {k: float(v) for k, v in zip(keys, values)}
    metrics.update(mR=float(values.mean()), images=n, captions=len(owners), token_groups=len(positive))
    return metrics, {'i2t_hits': torch.tensor(ih), 't2i_hits': torch.tensor(th), 'top_i': top_i, 'top_t': top_t}


@torch.no_grad()
def evaluate(model, gallery, out):
    was_training = model.training
    model.eval()
    fs, ts = [], []
    for a in range(0, len(gallery['names']), 32):
        fs.append(model.img_embeds(gallery['images'][a:a+32].cuda(),
                                  agent_hidden=gallery['hidden'][a:a+32].cuda())['image_feats'].cpu())
    for tokens in gallery['tokens'].split(64):
        ts.append(model.txt_embeds(tokens.cuda())['text_feats'].cpu())
    f, t = torch.cat(fs), torch.cat(ts)
    scores = model.model.logit_scale.exp().cpu()*f@t.T
    metrics, tensors = retrieval_metrics(scores, gallery)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({**tensors, 'image_features': f, 'text_features': t, 'scores': scores,
                'tokens': gallery['tokens'], 'owners': torch.tensor(gallery['owners']),
                'groups': torch.tensor(gallery['groups']), 'logit_scale': float(model.model.logit_scale.exp())},
               out/'tensors.pt')
    dump(out/'metrics.json', metrics)
    model.train(was_training)
    return metrics

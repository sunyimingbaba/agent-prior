"""Roll out and score every locked photo for one saved weight state."""
import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList

from shared import ASSETS, REBUILD, ROOT, dump, read, rgb, setup, sha, table
from readouts import clip, load_model
from rollout import SYSTEM_PROMPT, load_mllm, rollout_one
from tools import scene_classify_with_conf
from agent.grpo.reward import r_format

PREVIOUS = REBUILD / 'matched_policy_20261120'


def tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class SuppressFirstEOS(LogitsProcessor):
    def __init__(self, prompt_length, eos):
        self.prompt_length = prompt_length
        self.eos = eos

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] == self.prompt_length:
            scores[:, self.eos] = -float('inf')
        return scores


def evaluate(arm, step, suppress=False):
    setup()
    for file, expected in read(ROOT/'inputs/lock.json').items():
        assert sha(ROOT/'inputs'/file) == expected
    if arm == 'base':
        assert step == 0 and not suppress
        adapter = ASSETS/'ucm_main_lora'
        ref = read(PREVIOUS/'review/base_generic_replay.json')['reloaded']
        expected_file, expected_params = ref['file_hash'], ref['loaded_params']
    else:
        assert arm in read(ROOT/'inputs/protocol.json')['arms'] and step in (6,12,24)
        assert not suppress or (arm.startswith('b_') and step == 24)
        checkpoint = ROOT/('policy_'+arm)/f'step_{step:02d}'
        ref = read(checkpoint/'update.json')
        expected_file, expected_params = ref['adapter_hash'], ref['final_params']
        adapter = checkpoint/'adapter'
    assert sha(adapter/'adapter_model.safetensors') == expected_file
    out = ROOT/'eval'/f'{arm}_{step:02d}{"_force" if suppress else ""}'
    out.mkdir(parents=True, exist_ok=True)
    model, processor = load_mllm(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'), 'cuda', str(adapter))
    active = hashlib.sha256()
    for name, param in model.named_parameters():
        if '.lora_' in name:
            active.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    assert active.hexdigest() == expected_params
    eos = sorted(model.generation_config.eos_token_id)
    assert eos == [151643, 151645], eos
    tool, preprocess = clip.load(str(ASSETS/'models/ViT-L-14.pt'), device='cpu', jit=False)
    tool = tool.float().eval().cuda()
    cp = torch.load(ASSETS/'ucm_scene_cls.pt', map_location='cpu', weights_only=True)
    head = torch.nn.Linear(cp['feat_dim'], len(cp['classes']))
    head.load_state_dict(cp['head'])
    head = head.eval().cuda()
    frozen = load_model(False)
    t = table('val')
    reference = PREVIOUS/'eval/base'
    scorer_hashes = read(reference/'scored.json')
    assert sha(reference/'text.pt') == scorer_hashes['text']
    texts = torch.load(reference/'text.pt', weights_only=True, map_location='cpu')
    captions = [caption for five in t['caption'].to_pylist() for caption in five]
    raw_ids = clip.tokenize(captions, context_length=77-frozen.prompt_length)
    assert torch.equal(texts['native_tokens'][texts['caption_order']], raw_ids)
    with torch.inference_mode():
        assert (frozen.txt_embeds(raw_ids[:4].cuda())['text_feats'].cpu()-texts['native'][texts['caption_order'][:4]]).abs().max() < 2e-5
    detail = read(ROOT/'inputs/detail.json')
    val = read(ROOT/'inputs/validation.json')
    assert len(val)==len(t)==210 and len(detail)==16
    generate = model.generate
    rounds = []
    def capture(**kw):
        if suppress and not rounds:
            kw['logits_processor'] = LogitsProcessorList([SuppressFirstEOS(kw['input_ids'].shape[1], eos)])
        ans = generate(**kw)
        tokens = ans.sequences[0,kw['input_ids'].shape[1]:].cpu().tolist()
        rounds.append({'ids':tokens,'text':processor.decode(tokens,skip_special_tokens=True)})
        return ans
    model.generate = capture
    parts = ['native','detail'] if suppress else ['native','generic','detail']
    bindings = {}
    for part in parts:
        path = out/(part+'.pt')
        info = out/(part+'.json')
        if info.exists():
            old = read(info)
            assert old['weights']==expected_file and old['active']==expected_params and old['suppress']==suppress and old['bundle']==sha(path)
            bindings[part] = sha(info)
            continue
        records, hidden, visual = [], [], []
        for r in (detail if part=='detail' else val):
            original = rgb(t,r)
            views = ['upright','rotated'] if part=='detail' else ['upright']
            query = r['orig_query'] if part=='native' else r['scene_query'] if part=='detail' else 'An aerial photograph.'
            for view in views:
                image = original if view=='upright' else original.transpose(Image.Transpose.ROTATE_180)
                events = []
                def scene(im):
                    answer, conf = scene_classify_with_conf(im, model=tool, preprocess=preprocess, cls_head=head, cls_names=list(cp['classes']))
                    events.append({'answer':answer,'confidence':float(conf)})
                    return answer
                rounds.clear()
                seed = 394000+r['row']*10
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                random.seed(seed)
                np.random.seed(seed)
                with torch.inference_mode():
                    h, used, filtered = rollout_one(model, processor, image, query, {'scene_classify':scene},3,64,'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
                    v = frozen.img_embeds(frozen.preprocess(image)[None].cuda(), agent_hidden=h[None].cuda())['image_feats'][0].cpu()
                assert h.shape==(3584,) and h.dtype==torch.bfloat16 and v.shape==(768,) and used==len(events) and filtered==0
                assert torch.isfinite(h).all() and abs(v.norm().item()-1)<1e-5
                if suppress:
                    assert rounds[0]['ids'][0] not in eos
                records.append(dict(image=r['image'],row=r['row'],view=view,query=query,seed=seed,
                                    rendered_sha=hashlib.sha256(image.tobytes()).hexdigest(),
                                    rounds=list(rounds),events=events,format=r_format('\n'.join(q['text'] for q in rounds)),
                                    immediate_eos=rounds[0]['ids'] in [[i] for i in eos],hidden_sha=tensor_digest(h)))
                hidden.append(h)
                visual.append(v)
        assert len(records)==(32 if part=='detail' else 210)
        tmp=path.with_suffix('.part')
        torch.save({'hidden':torch.stack(hidden),'visual':torch.stack(visual)},tmp)
        tmp.replace(path)
        dump(info,dict(weights=expected_file,active=expected_params,suppress=suppress,part=part,bundle=sha(path),records=records))
        bindings[part]=sha(info)
        print(arm,step,suppress,part,len(records),flush=True)
    dump(out/'complete.json',dict(arm=arm,step=step,suppress=suppress,source=sha(Path(__file__)),parts=bindings,text=scorer_hashes['text'],readout=scorer_hashes['model'],active=expected_params))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('arm')
    ap.add_argument('step',type=int)
    ap.add_argument('--suppress',action='store_true')
    args=ap.parse_args()
    evaluate(args.arm,args.step,args.suppress)

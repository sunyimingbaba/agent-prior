"""Roll out and score every locked photo for one saved weight state."""
import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from shared import ASSETS, REBUILD, ROOT, dump, read, rgb, setup, sha, table
from readouts import clip, load_model
from rollout import SYSTEM_PROMPT, load_mllm, rollout_one
from tools import scene_classify_with_conf
from agent.grpo.reward import r_format

PREVIOUS = REBUILD / 'matched_policy_20261120'


def tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def evaluate(arm, step, quarter_only=False):
    setup()
    for file, expected in read(ROOT/'inputs/lock.json').items():
        assert sha(ROOT/'inputs'/file) == expected
    if arm == 'base':
        assert step == 0
        adapter = ASSETS/'ucm_main_lora'
        ref = read(PREVIOUS/'review/base_generic_replay.json')['reloaded']
        expected_file, expected_params = ref['file_hash'], ref['loaded_params']
    else:
        assert arm in read(ROOT/'inputs/protocol.json')['arms'] and step in (8,16)
        checkpoint = ROOT/('policy_'+arm)/f'step_{step:02d}'
        ref = read(checkpoint/'update.json')
        expected_file, expected_params = ref['adapter_hash'], ref['final_params']
        adapter = checkpoint/'adapter'
    assert sha(adapter/'adapter_model.safetensors') == expected_file
    if quarter_only:
        assert step in (0,16)
        extra=read(ROOT/'inputs/quarter_protocol.json')
        assert extra['original']==sha(ROOT/'inputs/detail.json') and extra['questions']==sha(ROOT/'inputs/quarter_questions.json')
    out = ROOT/('quarter_eval' if quarter_only else 'eval')/f'{arm}_{step:02d}'
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
    frozen = load_model()
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
    training=read(ROOT/'inputs/train.json')
    train_table=table('train')
    phrase_cases=read(ROOT/'inputs/quarter_questions.json') if quarter_only else training
    words=('C','D') if quarter_only else ('A','B')
    train_phrases=sorted({q for r in phrase_cases for w in words for q in r[w]})
    toks=clip.tokenize(train_phrases,context_length=77-frozen.prompt_length)
    with torch.inference_mode():
        fv=torch.cat([frozen.txt_embeds(b.cuda())['text_feats'].cpu() for b in toks.split(24)])
    train_text=out/('quarter_text.pt' if quarter_only else 'train_text.pt')
    torch.save(dict(phrases=train_phrases,features=fv),train_text)
    generate = model.generate
    rounds = []
    def capture(**kw):
        ans = generate(**kw)
        tokens = ans.sequences[0,kw['input_ids'].shape[1]:].cpu().tolist()
        rounds.append({'ids':tokens,'text':processor.decode(tokens,skip_special_tokens=True)})
        return ans
    model.generate = capture
    parts = ['quarter'] if quarter_only else ['native','generic','detail','trainpanel']
    bindings = {}
    for part in parts:
        path = out/(part+'.pt')
        info = out/(part+'.json')
        if info.exists():
            old = read(info)
            assert old['weights']==expected_file and old['active']==expected_params and old['bundle']==sha(path)
            bindings[part] = sha(info)
            continue
        records, hidden, visual = [], [], []
        cases = training if part=='trainpanel' else detail if part in ('detail','quarter') else val
        for r in cases:
            original = rgb(train_table if part=='trainpanel' else t,r)
            views = ['upright','rotated'] if part in ('trainpanel','detail') else ['upright']
            if part=='quarter':views=['cw90','ccw90']
            query = r['orig_query'] if part=='native' else r['scene_query'] if part in ('detail','trainpanel','quarter') else 'An aerial photograph.'
            for view in views:
                rotations={'rotated':Image.Transpose.ROTATE_180,'cw90':Image.Transpose.ROTATE_270,'ccw90':Image.Transpose.ROTATE_90}
                image = original if view=='upright' else original.transpose(rotations[view])
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
                records.append(dict(image=r['image'],row=r['row'],view=view,query=query,seed=seed,
                                    rendered_sha=hashlib.sha256(image.tobytes()).hexdigest(),
                                    rounds=list(rounds),events=events,format=r_format('\n'.join(q['text'] for q in rounds)),
                                    immediate_eos=rounds[0]['ids'] in [[i] for i in eos],hidden_sha=tensor_digest(h)))
                hidden.append(h)
                visual.append(v)
        assert len(records)==(32 if part in ('detail','trainpanel','quarter') else 210)
        tmp=path.with_suffix('.part')
        torch.save({'hidden':torch.stack(hidden),'visual':torch.stack(visual)},tmp)
        tmp.replace(path)
        dump(info,dict(weights=expected_file,active=expected_params,part=part,bundle=sha(path),records=records))
        bindings[part]=sha(info)
        print(arm,step,part,len(records),flush=True)
    dump(out/'complete.json',dict(arm=arm,step=step,source=sha(Path(__file__)),parts=bindings,text=scorer_hashes['text'],readout=scorer_hashes['model'],active=expected_params,train_text=sha(train_text)))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('arm')
    ap.add_argument('step',type=int)
    ap.add_argument('--quarter',action='store_true')
    args=ap.parse_args()
    evaluate(args.arm,args.step,args.quarter)

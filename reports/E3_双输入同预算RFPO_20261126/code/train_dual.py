"""A two-by-two RFPO experiment with two eight-trajectory groups per image and update."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parent / 'soft_guard_20261125'
sys.path.insert(0, str(ROOT.parent / 'source_transfer_2h_20261125/code'))
from remote_common import setup, table, rgb, read, dump, sha, savept
from runtime import BASE as LEGACY, initial
from policy import policy
from full_traces import trace_batch
from prefixes import greedy
from train import sequence_logp
from orders import advantages
sys.path.insert(0, str(PREVIOUS / 'code'))
from strict_reward import strict_rewards


def score_group(worker, name, states, anchor):
    # The verified CPU reader uses a fixed 17-row packet and caches equal vectors.
    assert len(states) == 8
    vectors = torch.stack(states + states + [anchor]).float().numpy()
    encoded = base64.b64encode(vectors.tobytes()).decode()
    worker.stdin.write(json.dumps(dict(image=name, data=encoded)) + '\n')
    worker.stdin.flush()
    scores = json.loads(worker.stdout.readline())
    for field in ('rr', 'detail'):
        values = scores[field]
        assert len(values) == 17 and values[:8] == values[8:16]
        scores[field] = values[:8] + values[-1:]
    return scores


def replay(model, contexts, histories, adv):
    groups = {}
    for i, history in enumerate(histories):
        for context, record in history:
            key = context, tuple(record[0]), tuple(tuple(x.tolist()) for x in record[3])
            if key not in groups:
                groups[key] = [0., context, record]
            groups[key][0] += float(adv[i])
    errors = []
    for weight, context, record in groups.values():
        if abs(weight) < 1e-12:
            continue
        logp, difference = sequence_logp(model, contexts[context], record)
        (-weight * logp / 16).backward()
        errors.extend(difference)
    return errors


def train(seed, arm):
    assert seed in ('a', 'b') and arm in ('singleplain', 'singlejoint', 'dualplain', 'dualjoint')
    lockfile = ROOT / 'inputs/lock.json'
    lock = read(lockfile)
    assert sha(ROOT/'实验口径.md') == lock['protocol']
    for name, h in lock['implementation'].items():
        assert sha(ROOT/'code'/name) == h
    for name, h in lock['sources'].items():
        assert sha(PREVIOUS/'code'/name) == h
    for name, h in lock['manifests'].items():
        assert sha(ROOT/'inputs'/name) == h
    roster = read(ROOT/'inputs/train.json')[seed]
    assert len(roster) == 24
    start = initial(seed)
    start_hash = sha(start/'adapter/adapter_model.safetensors')
    assert start_hash == lock['starts'][seed]
    setup()
    torch.set_num_threads(1)
    dest = ROOT/'models'/(seed+'_'+arm)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'complete.json').exists():
        proof = read(dest/'complete.json')
        assert proof['final'] == sha(dest/'adapter/adapter_model.safetensors') and proof['receipt'] == sha(lockfile)
        return
    model, processor = policy(start)
    params = {k: p.requires_grad_() for k, p in model.named_parameters() if '.lora_' in k}
    opt = torch.optim.AdamW(params.values(), lr=1e-6)
    tb = table()
    state = dest/'resume.pt'
    history = []
    if state.exists():
        saved = torch.load(state, weights_only=True, map_location='cpu')
        assert (saved['initial'], saved['decision']) == (start_hash, sha(lockfile))
        for k, p in params.items():
            p.data.copy_(saved['weights'][k].to(p.device))
        opt.load_state_dict(saved['opt'])
        history = saved['history']
        del saved
    worker = subprocess.Popen([str(LEGACY/'env/bin/python'), '-u', str(PREVIOUS/'code/mined_worker.py'), seed, 'selected'],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1'), stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, text=True)
    try:
        assert worker.stdout.readline().strip() == 'READY'
        for j, row in enumerate(roster):
            if j < len(history):
                continue
            image = rgb(tb, row)
            opt.zero_grad(set_to_none=True)
            views = []
            for block in (0, 1):
                field = 'scene_query' if block == 1 and arm.startswith('dual') else 'orig_query'
                q, feedback = row[field], row['feedback']
                rng = (91000000 if seed == 'a' else 101000000) + j*1000 + block*100
                anchor, actions, calls, _ = greedy(model, processor, image, q, feedback)
                ctx, traces, states = trace_batch(model, processor, image, q, feedback, rng)
                scores = score_group(worker, row['image'], states, anchor)
                rewards, joint = strict_rewards(scores)
                chosen = 'joint' if arm.endswith('joint') else 'ordinary'
                adv = advantages(rewards[chosen])
                errors = replay(model, ctx, traces, adv)
                views.append(dict(field=field, query=q, rng=rng, anchor=actions, calls=calls,
                    paths=[[rec[1][0] for rec in trace] for trace in traces], scores=scores,
                    rewards={k:v.tolist() for k,v in rewards.items()}, joint=joint, advantage=adv.tolist(),
                    replayed=len(errors), mean_error=float(np.mean(errors)) if errors else 0., max_error=max(errors,default=0.)))
                del ctx, traces, states
            norm = float(torch.nn.utils.clip_grad_norm_(params.values(), 1., error_if_nonfinite=True))
            if norm > 0:
                opt.step()
            history.append(dict(image=row['image'], step=j+1, views=views, norm=norm))
            if (j+1) % 4 == 0:
                savept(state, dict(initial=start_hash, decision=sha(lockfile), weights={k:p.detach().cpu() for k,p in params.items()},
                                   opt=opt.state_dict(), history=history))
            print(seed, arm, j+1, norm, flush=True)
    finally:
        worker.stdin.close()
        worker.wait()
    assert worker.returncode == 0 and len(history) == 24
    model.save_pretrained(dest/'adapter')
    dump(dest/'steps.json', history)
    dump(dest/'complete.json', dict(initial=start_hash, final=sha(dest/'adapter/adapter_model.safetensors'), steps=len(history),
         updated=sum(s['norm']>0 for s in history), receipt=sha(lockfile), logs=sha(dest/'steps.json')))


if __name__ == '__main__':
    train(*sys.argv[1:])

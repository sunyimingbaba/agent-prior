"""Check whether the same RFPO update reproduces across independent processes."""
import base64
import hashlib
import json
import os
import subprocess
import sys
import numpy as np
import torch
from train_pilot import ROOT, setup, table, rgb, read, sha, dump, initial, policy, trace_batch, advantages, replay, LEGACY


def digest_parameters(parameters, gradient=False):
    digest = hashlib.sha256()
    for name, parameter in sorted(parameters.items()):
        tensor = parameter.grad if gradient else parameter.data
        digest.update(name.encode())
        if gradient and tensor is None:
            digest.update(b"unused")
            continue
        assert torch.isfinite(tensor).all()
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def main(seed, repetition):
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    setup()
    torch.set_num_threads(1)
    receipt = read(ROOT / "inputs/pilot/decision.json")
    start = initial(seed)
    assert sha(start / "adapter/adapter_model.safetensors") == receipt["starts"][seed]
    row = read(ROOT / "inputs/pilot/train.json")[seed][0]
    model, processor = policy(start)
    params = {k:p.requires_grad_() for k,p in model.named_parameters() if ".lora_" in k}
    opt = torch.optim.AdamW(params.values(), lr=1e-6)
    rng = 91000000 if seed == "a" else 101000000
    context, paths, states = trace_batch(model, processor, rgb(table(), row), row["orig_query"], row["feedback"], rng)
    state_bytes = torch.stack(states).float().numpy().tobytes()
    worker = subprocess.run([str(LEGACY / "env/bin/python"), "-Bu", str(ROOT/"code/policy_worker.py"), seed],
        input=json.dumps({"image":row["image"], "data":base64.b64encode(state_bytes).decode()})+"\n",
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1"), text=True,
        stdout=subprocess.PIPE, check=True)
    ready, result = worker.stdout.splitlines()
    assert ready == "READY"
    winners = json.loads(result)["winners"]
    scores = np.zeros(8)
    scores[winners["old3"]] = 1
    adv = advantages(scores)
    replay(model, context, paths, adv*2)
    norm = float(torch.nn.utils.clip_grad_norm_(params.values(), 1., error_if_nonfinite=True))
    gradients = digest_parameters(params, gradient=True)
    opt.step()
    result = dict(seed=seed, rgb=row["rgb_sha256"], rng=rng, start=receipt["starts"][seed], norm=norm,
        states=hashlib.sha256(state_bytes).hexdigest(), winners=winners, advantages=adv.tolist(),
        gradients=gradients, weights=digest_parameters(params), paths=[[entry[1][0] for entry in path] for path in paths])
    folder = ROOT / "numeric"
    folder.mkdir(exist_ok=True)
    dump(folder / f"{seed}_{repetition}.json", result)


if __name__ == "__main__":
    main(*sys.argv[1:])

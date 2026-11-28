"""Keep policy outputs fixed, then compare each source against private and shared galleries."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
STAGE = ROOT.parent
REPORTS = STAGE.parent
FORM = REPORTS/"E3-2_UCM与RSITMD独立复验_20261124"
DPO = REPORTS/"E3_双收益回答对的偏好训练_20261127"
OLD = REPORTS/"E3_回填后双收益训练图筛选_20261127"
EXTRA = STAGE/"extra164"
sys.path.insert(0, str(FORM/"code"))
from metrics import fixed, recalls


def read(path):
    return json.loads(path.read_bytes())


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def metric(sims):
    n = len(sims)
    assert sims.shape == (n, 5*n)
    own = np.repeat(np.eye(n, dtype=bool), 5, axis=1)
    i2t = np.mean(list(recalls(sims, own).values()), axis=0)
    t2i = np.mean(list(recalls(sims.T, own.T).values()), axis=0).reshape(n, 5).mean(axis=1)
    return (i2t+t2i)/2


def main():
    names = read(FORM/"inputs/ucm.json")
    extra = read(EXTRA/"inputs/rows.json")
    formal = load(FORM/"results/ucm_text.pt")
    extra_text = load(EXTRA/"inputs/texts.pt")["captions"]
    nat = next(REPORTS.glob("E3-1_奖励一致性*/空间关系候选*/native"))
    bank = load(nat/"features.pt")
    src = read(nat/"index.json")["columns"]
    vectors = list(formal["native"])
    indices = {tuple(t.tolist()): i for i, t in enumerate(formal["native_tokens"])}
    fcols = formal["caption_order"].numpy().tolist()
    ecols, basecols = [], []
    for r in extra:
        for cap, j in zip(r["captions"], r["panel_positive_ids"], strict=True):
            token = tuple(bank["tokens"][src[cap]].tolist())
            assert torch.equal(bank["features"][src[cap]], extra_text[j])
            if token not in indices:
                indices[token] = len(vectors)
                vectors.append(extra_text[j])
            ecols.append(indices[token])
            basecols.append(j)
    unique = torch.stack(vectors)
    levels = {}
    for seed in "ab":
        for arm, folder, suffix in (("rfpo_r2", OLD, "r2"), ("rfpo_strict", STAGE, "strict"),
                                    ("dpo_r2", DPO, "r2"), ("dpo_r3", DPO, "r3")):
            f = load(folder/"evaluation/records"/f"{seed}_{suffix}"/"features.pt")
            e = load(EXTRA/"records"/f"{seed}_{arm}.pt")
            assert f["adapter"] == e["adapter"] and f["images"] == [r["image"] for r in names]
            assert [t["image"] for t in e["traces"]] == [r["image"] for r in extra]
            sims = fixed(torch.cat((f["features"], e["visual"])), unique)
            full = metric(sims[:, fcols+ecols])
            native = metric(fixed(e["visual"], extra_text)[:, basecols])
            own = metric(sims[:210, fcols])
            saved = read(folder/"evaluation/results/单次检索对照.json")
            expected = saved["official"]["levels"][suffix]["mR"]["seeds"]["ab".index(seed)]
            assert abs(own.mean() - expected) < 1e-12
            base_expect = read(EXTRA/"summary.json")["levels"][arm]["mR"]["seeds"]["ab".index(seed)]
            assert abs(native.mean() - base_expect) < 1e-12
            canon = metric(sims[210:, ecols])
            levels.setdefault(arm, {})[seed] = dict(
                official_native=own.mean(), extra_native=native.mean(),
                official_canonical=own.mean(), extra_canonical=canon.mean(),
                official_shared=full[:210].mean(), extra_shared=full[210:].mean())
    deltas = {}
    for end, start in (("rfpo_strict", "rfpo_r2"), ("dpo_r3", "dpo_r2")):
        deltas[end] = {k: np.mean([levels[end][s][k]-levels[start][s][k] for s in "ab"])
                       for k in levels[end]["a"]}
    result = dict(images=374, rows=1870, distinct_tokens=len(vectors), levels=levels, contrasts=deltas)
    (ROOT/"结果.json").write_text(json.dumps(result, indent=2)+"\n")
    print(deltas)


if __name__ == "__main__":
    main()

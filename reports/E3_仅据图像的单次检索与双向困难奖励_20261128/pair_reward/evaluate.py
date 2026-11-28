"""Test matched hard negatives in both retrieval directions on real candidate rollouts."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT.parents[1]
CALIBRATION = REPORTS / "E3_困难奖励跨图跨措辞校准_20261127"
sys.path.insert(0, str(CALIBRATION / "code"))
from reward import redistribute

ARMS = ("r2", "strict", "paired", "self_ref")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def guarded(scores, margins, ranks, visible):
    top = scores == scores.max()
    count = (margins > 0).sum(axis=1)
    qualified = count * top.sum() > count[top].sum()
    weakest = margins.min(axis=1)
    return redistribute(scores, np.where(qualified | top, weakest, weakest.min()), ranks, visible)


def votes(group, other):
    local = group["text_scores"]
    other_scores = other["text_scores"][:, [1, 3]]  # same true texts, opposite row orders them as false
    true_scores = local[:, [0, 2]]
    across = true_scores - other_scores.mean(axis=0)
    self_ref = true_scores - true_scores.mean(axis=0)
    margins = group["ab"]
    rr = group["rr"]
    scores = group["r2"]
    probabilities = {arm: [] for arm in ARMS}
    hidden_rr = {arm: [] for arm in ARMS}
    for h in range(rr.shape[1]):
        visible = [j for j in range(rr.shape[1]) if j != h]
        arms = {"r2": scores, "strict": guarded(scores, margins, rr, visible),
                "paired": guarded(scores, np.column_stack((margins, across)), rr, visible),
                "self_ref": guarded(scores, np.column_stack((margins, self_ref)), rr, visible)}
        for arm in ARMS:
            chosen = arms[arm] == arms[arm].max()
            p = chosen / chosen.sum()
            probabilities[arm].append(p)
            hidden_rr[arm].append(p @ rr[:, h])
    return ({arm: np.mean(probabilities[arm], axis=0) for arm in ARMS},
            {arm: np.mean(hidden_rr[arm]) for arm in ARMS})


def row_score(group, other, own_votes, partner_votes, held_mrr):
    local = group["text_scores"]
    opposite = other["text_scores"]
    out = {"MRR": held_mrr}
    for offset, label in ((4, "DE"), (8, "FG")):
        hit_image, hit_text = [], []
        for k in (offset, offset+2):
            hit_image.append(own_votes @ (local[:, k] > local[:, k+1]))
            delta = local[:, k, None] - opposite[None, :, k+1]
            successes = (delta > 0).astype(float) + 0.5 * (delta == 0)
            hit_text.append(own_votes @ successes @ partner_votes)
        out[label + "_I2T"] = np.mean(hit_image)
        out[label + "_T2I"] = np.mean(hit_text)
    return out


def study(source, expected_pairs):
    rows = read(source/"inputs/rows.json")
    text = load(source/"inputs/vectors.pt")
    manifest = read(source/"inputs/manifest.json")
    assert all(sha(source/"inputs"/file) == digest for file, digest in manifest["compiled"].items())
    proofs = read(source/"scoring_complete.json")
    groups = defaultdict(list)
    for row in rows:
        name = row["image"]
        file = name + ".pt"
        feature = source/"readouts"/file
        scores = source/"scores"/(name + ".json")
        assert sha(feature) == proofs["readouts"][name] and sha(scores) == proofs["files"][name]
        origin, archived = load(feature), read(scores)
        assert (archived["rgb"], origin["rgb"]) == (row["rgb_sha256"],) * 2
        if not archived["eligible"]:
            continue
        assert all(len(it["other_caption_rr"]) >= 2 for it in archived["items"])
        V = origin["visual"].double()
        words = text[name].double()
        similarities = torch.stack([words @ v for v in V]).numpy()
        actual = similarities[:, ::2] - similarities[:, 1::2]
        stored = np.array([it["hard_margins"]+it["DE"]+it["FG"] for it in archived["items"]])
        np.testing.assert_allclose(actual, stored, rtol=0, atol=1e-12)
        groups[row["pair"]].append(dict(row=row, text_scores=similarities, words=words, ab=stored[:, :2],
                                       rr=np.array([it["other_caption_rr"] for it in archived["items"]]),
                                       r2=np.asarray(archived["r2"])))
    per_image = []
    for pair, group in sorted(groups.items()):
        if len(group) != 2:
            continue
        a, b = sorted(group, key=lambda x: x["row"]["side"])
        assert [a["row"]["side"], b["row"]["side"]] == [0, 1]
        assert all(a["row"][q] == b["row"][q][::-1] for q in "ABDE")
        assert torch.equal(a["words"][::2], b["words"][1::2]) and torch.equal(a["words"][1::2], b["words"][::2])
        left, lrr = votes(a, b)
        right, rrr = votes(b, a)
        for own, partner, p, q, rr in ((a,b,left,right,lrr),(b,a,right,left,rrr)):
            result = {arm: row_score(own, partner, p[arm], q[arm], rr[arm]) for arm in ARMS}
            per_image.append(dict(image=own["row"]["image"], pair=pair, rgb=own["row"]["rgb_sha256"],
                                  winners={arm: p[arm].tolist() for arm in ARMS}, scores=result))
    assert len(per_image) == 2*expected_pairs
    count = expected_pairs
    weights = np.random.default_rng(281128).multinomial(count, np.full(count, 1/count), size=20000)
    keys = list(per_image[0]["scores"]["r2"])
    levels = {arm: np.array([[r["scores"][arm][k] for k in keys] for r in per_image]).reshape(count, 2, -1).mean(1)
              for arm in ARMS}
    def estimate(pair_values):
        ci = np.quantile(weights @ pair_values / count, [0.025, 0.975], axis=0)
        return {k: dict(mean=pair_values[:, i].mean(), ci=ci[:, i].tolist()) for i, k in enumerate(keys)}
    contrasts = {name: estimate(levels["paired"] - levels[name]) for name in ("r2", "strict", "self_ref")}
    return per_image, dict(images=len(per_image), pairs=count, levels={arm:estimate(v) for arm,v in levels.items()},
                           paired_minus=contrasts, images_with_change_vs_strict=sum(
                               not np.array_equal(r["winners"]["paired"], r["winners"]["strict"]) for r in per_image))


def main():
    torch.set_num_threads(4)
    results = {}
    pixels = set()
    sources = {"98": (REPORTS/"E3_双收益回答对的偏好训练_20261127/validation98/fresh98",49),
               "114": (CALIBRATION/"followup_pool",57)}
    for tag, (folder, n) in sources.items():
        rows, result = study(folder, n)
        assert pixels.isdisjoint(r["rgb"] for r in rows)
        pixels.update(r["rgb"] for r in rows)
        write(ROOT/("逐图_"+tag+".json"), rows)
        results[tag] = result
    dev, confirm = [results[t]["paired_minus"]["strict"] for t in ("98", "114")]
    results["advance"] = bool(confirm["DE_T2I"]["ci"][0] > 0 and dev["DE_T2I"]["mean"] > 0
                              and all(panel[k]["mean"] >= 0 for panel in (dev, confirm) for k in ("DE_I2T","MRR"))
                              and results["114"]["paired_minus"]["self_ref"]["DE_T2I"]["mean"] > 0)
    write(ROOT/"结果.json", results)
    print(results["advance"], dev, confirm)


if __name__ == "__main__":
    main()

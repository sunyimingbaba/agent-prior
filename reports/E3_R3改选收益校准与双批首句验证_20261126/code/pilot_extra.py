"""Alternate-caption retrieval and a new wording check for one greedy prior."""
from collections import defaultdict
import json
import numpy as np
import torch

from pilot_results import ROOT, RFPO, FORM, read, sha, get, paired, summary_by_seed, fixed
from transfer_scores import counts, summary, scalar_values


def main():
    receipt = read(ROOT / "inputs/pilot/decision.json")
    options = [*receipt["train_arms"], "retained"]
    gallery = read(FORM / "inputs/ucm.json")
    native = torch.load(FORM / "results/ucm_text.pt", weights_only=True)
    panel = read(RFPO / "inputs/panel.json")
    signatures = defaultdict(set)
    for row in read(ROOT / "inputs/real_roster.json"):
        signatures[row["attribute"], row["side"]].add(tuple(row["D"] + row["E"]))
    selected = [row for row in panel if tuple(row["D"] + row["E"]) in signatures[row["attribute"], row["side"]]]
    wordings = read(ROOT / "inputs/独立措辞.json")
    pack = torch.load(ROOT / "inputs/fresh_text.pt", weights_only=True)
    assert pack["source"] == sha(ROOT / "inputs/独立措辞.json")
    word_ids = {word: i for i, word in enumerate(pack["words"])}
    values = defaultdict(lambda: defaultdict(dict))
    stats = {}
    denom = None
    for arm in options:
        results = []
        for seed in "ab":
            for tag, rows in (("official", gallery), ("primary", panel)):
                features = get(tag, arm, seed, rows)
                if tag == "official":
                    ids, values_by_image, texts = counts(features, native, "alternatives")
                    if denom is None:
                        owners, denom = ids, texts
                    assert np.array_equal(ids, owners) and np.array_equal(texts, denom)
                    results.append(values_by_image)
                slots = {row["rgb_sha256"]: i for i, row in enumerate(rows)}
                covered = [r for r in selected if r["rgb_sha256"] in slots]
                assert len(covered) == (56 if tag == "official" else 66)
                scores = fixed(features, pack["features"])
                correct = []
                for r in covered:
                    choices = wordings[r["attribute"]]
                    side = r["side"]
                    row = scores[slots[r["rgb_sha256"]]]
                    correct.append(np.mean([row[word_ids[words[side]]] > row[word_ids[words[1-side]]] for words in choices]))
                values[tag][arm][seed] = {"FG": np.asarray(correct)}
        stats[arm] = np.stack(results)
    comp = {}
    comparisons = [("old3", "old"), ("old3", "ordinary"), ("old3", "retained"), ("old", "ordinary")]
    for left, right in comparisons:
        difference = stats[left] - stats[right]
        comp[f"{left}-{right}"] = {"metrics": summary(difference.mean(0), denom),
            "mR_by_seed": [float(scalar_values(x.sum(0), np.asarray(len(denom)), np.asarray(denom.sum()))[-1]) for x in difference]}
    out = {"other_captions": {"images": len(denom), "entries": int(denom.sum()),
           "levels": {a: summary(x.mean(0), denom) for a,x in stats.items()}, "contrasts": comp}, "new_wordings": {}}
    for tag, source in (("official", gallery), ("primary", panel)):
        slots = {row["rgb_sha256"] for row in source}
        labels = [row["pair"] for row in selected if row["rgb_sha256"] in slots]
        out["new_wordings"][tag] = {"images": len(labels), "pairs": len(set(labels)),
            "levels": {a: summary_by_seed(values[tag][a], labels) for a in options},
            "contrasts": {f"{a}-{b}": paired(values[tag][a], values[tag][b], labels) for a,b in comparisons}}
    (ROOT/"results/单次输出补充.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

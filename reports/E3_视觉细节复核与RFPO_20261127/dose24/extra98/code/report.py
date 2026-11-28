"""Compute the preregistered paired outcome from four independent policy rollouts."""
from pathlib import Path
import sys
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parents[1]
sys.path.insert(0, str(STAGE/"extra164/code"))
from score import read, sha, fixed, measures


def main():
    manifest = read(ROOT/"inputs/manifest.json")
    assert sha(ROOT/"inputs/rows.json") == manifest["rows"]
    assert sha(ROOT/"inputs/texts.pt") == manifest["texts"]
    rows = read(ROOT/"inputs/rows.json")
    texts = torch.load(ROOT/"inputs/texts.pt", weights_only=True)
    image_ids = [r["rgb_sha256"] for r in rows]
    pairs = sorted({r["pair"] for r in rows})
    slots = [[i for i, r in enumerate(rows) if r["pair"] == p] for p in pairs]
    assert len(pairs) == 49 and all(len(idx) == 2 for idx in slots)
    w = np.random.default_rng(271127).multinomial(49, np.full(49, 1 / 49), size=20000)

    def summarize(data):
        paired = data.mean(axis=0)[slots].mean(axis=1)
        assert paired.shape == (49, 6) and np.isfinite(paired).all()
        interval = np.quantile(w @ paired / 49, [.025, .975], axis=0)
        return {k: dict(mean=paired[:, j].mean(), ci=interval[:, j].tolist(),
                        seeds=data[:, :, j].mean(axis=1).tolist()) for j, k in
                enumerate(("DE", "MRR", "mR", "I2T", "T2I", "FG"))}
    parts = {}
    for arm in ("r2", "strict"):
        results = []
        for seed in "ab":
            path = ROOT/"records"/f"{seed}_{arm}.pt"
            meta = read(path.with_suffix(".json"))
            assert meta["sha256"] == sha(path) and meta["policy"] == sha(ROOT/"code/predict.py")
            assert meta["inputs"] == manifest["rows"] and meta["protocol"] == manifest["protocol"]
            d = torch.load(path, map_location="cpu", weights_only=True)
            assert [t["rgb"] for t in d["traces"]] == image_ids
            assert d["adapter"] == meta["adapter"]
            metrics = measures(rows, texts, d["visual"])
            dim = fixed(d["visual"], texts["details"])[np.arange(len(rows))[:, None],
                           np.array([r["panel_fg_ids"] for r in rows])]
            fg = (dim[:, ::2] > dim[:, 1::2]).mean(axis=1)
            results.append(np.column_stack((metrics, fg)))
        parts[arm] = np.stack(results)
    out = dict(levels={arm: summarize(values) for arm, values in parts.items()},
               difference=summarize(parts["strict"] - parts["r2"]), images=98, natural_pairs=49)
    x = out["difference"]
    out["conditions_98"] = (x["DE"]["ci"][0] > 0 and x["MRR"]["mean"] >= 0 and x["mR"]["mean"] >= 0)
    np.savez_compressed(ROOT/"per_image.npz", **parts)
    (ROOT/"summary.json").write_text(json.dumps(out, indent=2)+"\n")
    print(out["difference"])


if __name__ == "__main__":
    main()

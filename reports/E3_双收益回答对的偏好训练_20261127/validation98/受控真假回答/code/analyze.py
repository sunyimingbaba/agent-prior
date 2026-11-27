"""Summarize controlled true-minus-false effects and audit label reversal within natural pairs."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def metrics(group):
    return dict(weak=float(min(group["hard_margins"])), DE=float((np.asarray(group["DE"])>0).mean()),
                FG=float((np.asarray(group["FG"])>0).mean()), other_MRR=float(np.mean(group["other_caption_rr"])))


def estimate(rows):
    clusters = sorted({r["pair"] for r in rows})
    if not clusters:
        return dict(images=0, pairs=0)
    n = len(clusters)
    draws = np.random.default_rng(27112026).multinomial(n, np.full(n,1/n), size=20000)
    stats = {}
    for key in ("weak", "DE", "FG", "other_MRR", "truth_preferred"):
        totals = np.array([[sum(r["diff"][key] for r in rows if r["pair"] == pair),
                            sum(r["pair"] == pair for r in rows)] for pair in clusters])
        boots = draws @ totals
        value = totals[:,0].sum() / totals[:,1].sum()
        entry = dict(estimate=float(value), interval=np.quantile(boots[:,0]/boots[:,1],[.025,.975]).tolist())
        if key != "truth_preferred":
            entry.update({s: float(np.mean([r[s][key] for r in rows])) for s in ("true","false")})
        stats[key] = entry
    diff = np.array([r["diff"]["weak"] for r in rows])
    return dict(images=len(rows),pairs=n,metrics=stats,
                counts=dict(true_higher=int((diff>0).sum()),tied=int((diff==0).sum()),true_lower=int((diff<0).sum())))


def main():
    proof = read(ROOT / "scoring.json")
    generated = read(ROOT / "generation.json")
    assert digest(ROOT/"generation.json") == proof["generation"]
    assert generated["protocol"] == digest(ROOT/"口径.md")
    assert generated["code"] == digest(ROOT/"code/collect.py") and proof["code"] == digest(ROOT/"code/read.py")
    calibration = read(ROOT / "calibration_readouts.json")
    assert calibration["calibration_ok"] and proof["calibration_readouts"] == digest(ROOT/"calibration_readouts.json")
    roster = read(ROOT.parent / "fresh98/inputs/rows.json")
    fresh = read(ROOT.parent / "fresh98/c_complete.json")
    import torch
    rows = []
    groups = defaultdict(list)
    for row in roster:
        name = row["image"]
        path = ROOT / "scores" / (name+".json")
        original = ROOT/"raw"/(name+".pt")
        projected = ROOT/"readouts"/(name+".pt")
        assert digest(path) == proof["files"][name]
        sc = read(path)
        assert digest(original) == generated["files"][name] == sc["source"]
        assert digest(projected) == proof["readouts"][name] == sc["readout"]
        raw = torch.load(original, map_location="cpu", weights_only=True)
        source = torch.load(ROOT.parent/"fresh98/raw/c"/(name+".pt"),map_location="cpu",weights_only=True)
        assert raw["source"] == fresh["files"][name] and torch.equal(raw["context_prefix"],source["context"]["prefix"])
        assert raw["states"].shape == (2,3584) and torch.isfinite(raw["states"]).all()
        assert sc["rgb"] == row["rgb_sha256"] and sc["pair"] == row["pair"]
        assert sc["texts"] == raw["texts"] == ["<answer>"+q+"</answer>" for q in row["A"]]
        assert raw["ids"][0][-1] == raw["ids"][1][-1] == 151645
        assert len(sc["true"]["other_caption_rr"]) == len(sc["false"]["other_caption_rr"]) == len(row["alt_ids"]) > 0
        t,f = metrics(sc["true"]),metrics(sc["false"])
        d = {k:float(t[k]-f[k]) for k in t}
        d["truth_preferred"] = float(d["weak"]>0) + .5*float(d["weak"]==0)
        current = dict(image=name,pair=row["pair"],side=row["side"],texts=row["A"],rgb=row["rgb_sha256"],
                       true=t,false=f,diff=d)
        rows.append(current)
        groups[row["pair"]].append(current)
    swap = {}
    for key, members in groups.items():
        assert len(members) == 2 and sorted(r["side"] for r in members) == [0,1]
        p,q = sorted(members,key=lambda r:r["side"])
        is_swap = p["texts"] == q["texts"][::-1]
        swap[key] = dict(images=[p["image"],q["image"]], opposite=is_swap)
    swapped = [r for r in rows if swap[r["pair"]]["opposite"]]
    rest = [r for r in rows if not swap[r["pair"]]["opposite"]]
    total = dict(all=estimate(rows),reciprocal_pairs=estimate(swapped),other_pairs=estimate(rest),
                 calibration_exact=read(ROOT/"calibration.json")["exact_all"],calibration_readouts_exact=calibration["all_equal"],
                 calibration_slots=calibration["checked"],source=digest(ROOT/"scoring.json"),code=digest(Path(__file__)))
    write(ROOT / "paired_differences.json", rows)
    write(ROOT / "natural_pairs.json", swap)
    write(ROOT / "summary.json", total)
    print(json.dumps(total,ensure_ascii=False))


if __name__ == "__main__":
    main()

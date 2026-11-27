"""Recalculate the group-bootstrap outputs and each fresh selected slot independently."""
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_bytes())


def compare_numbers(panel, result):
    pairs = sorted({r["pair"] for r in panel})
    n = len(pairs)
    counts = np.random.default_rng(27112026).multinomial(n, np.full(n, 1 / n), size=20000)
    for target in ("first", "tied"):
        for field in ("DE", "FG", "held_rr", "switched"):
            totals = []
            for pair in pairs:
                inner = []
                for row in panel:
                    if row["pair"] != pair:
                        continue
                    trials = row["trials"]
                    blocks = sorted({(t["seed"], t["block"]) for t in trials})
                    inner.append(np.mean([np.mean([t[target][field] for t in trials
                                                  if (t["seed"], t["block"]) == loc]) for loc in blocks]))
                totals.append([sum(inner), len(inner)])
            totals = np.array(totals)
            summed = counts @ totals
            ci = np.quantile(summed[:, 0] / summed[:, 1], [.025, .975])
            actual = result[target]["metrics"][field]
            assert abs(actual["mean"] - totals[:, 0].sum() / totals[:, 1].sum()) < 1e-14
            assert np.allclose(actual["interval"], ci, rtol=0, atol=1e-14)


def verify_slots(row):
    d = read(ROOT / "fresh98/scores" / (row["image"] + ".json"))
    r2 = np.array(d["r2"])
    a = np.array([x["hard_margins"] for x in d["items"]])
    rr = np.array([x["other_caption_rr"] for x in d["items"]])
    de = (np.array([x["DE"] for x in d["items"]]) > 0).mean(1)
    fg = (np.array([x["FG"] for x in d["items"]]) > 0).mean(1)
    first = np.argmax(r2)
    top = np.flatnonzero(r2 == r2[first])
    for t in row["trials"]:
        h = t["hidden"]
        visible = np.delete(rr, h, axis=1)
        n_correct = (a > 0).sum(1)
        options = np.flatnonzero((r2 < r2[first]) & (n_correct > n_correct[first]) &
                                 (visible >= visible[first]).all(axis=1))
        best = max(options, key=lambda j:(n_correct[j], min(a[j]), r2[j], -j)) if options.size else first
        assert t["base"] == first and t["selected"] == best and t["top"] == top.tolist()
        for tag, base in (("first", [first]), ("tied", top)):
            for key, vec in (("DE", de), ("FG", fg), ("held_rr", rr[:, h])):
                assert abs(t[tag][key] - (vec[best] - vec[base].mean())) < 1e-15


def main():
    receipt = {}
    for tag in ("history98", "fresh98"):
        folder = ROOT / tag
        result = read(folder / "summary.json")
        with gzip.open(folder / "trials.jsonl.gz", "rt") as f:
            rows = [json.loads(line) for line in f]
        compare_numbers(rows, result)
        if tag == "fresh98":
            for row in rows:
                verify_slots(row)
        receipt[tag] = dict(images=len(rows), natural_pairs=len({r["pair"] for r in rows}), trials=sum(len(r["trials"]) for r in rows),
                            verified_means=8, verified_interval_bounds=16, independent_slot_checks=tag == "fresh98")
    (ROOT / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()

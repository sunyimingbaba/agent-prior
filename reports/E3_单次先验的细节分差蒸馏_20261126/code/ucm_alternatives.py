"""Apply the cross-dataset caption filtering rule to the 210-image UCM gallery."""
import numpy as np
from local import ROOT, FORM, read, load, dump
from margin_scores import features
from transfer_scores import counts, summary


def main():
    rows = read(FORM / "inputs/ucm.json")
    gallery = load(FORM / "results/ucm_text.pt")
    arms = ["retained", "parent", "coldR0", "warmordinary", "ordinary", "guard", "teacher", "reachable_ordinary",
            "reachable_teacher", "strict_warmstrict", "strict_coldordinary", "strict_coldstrict"]
    sources = [read(p / "results/score.json") for p in (ROOT, ROOT / "reachable", ROOT / "strict_rfpo")]
    levels = {k: v for src in sources for k, v in src["official"]["levels"].items()}
    raw = {}; ds = None; idx = None
    for arm in arms:
        per_seed = []
        for j, seed in enumerate("ab"):
            visual = features("official", arm, seed, rows)
            _, full, denom = counts(visual, gallery, "native")
            independent = (full[:, :3].mean() + full[:, 3:].sum() / (3 * denom.sum())) / 2
            assert abs(independent - levels[arm]["mR"]["seeds"][j]) < 1e-10
            ids, hits, d = counts(visual, gallery, "alternatives")
            if ds is None: idx, ds = ids, d
            assert np.array_equal(ids, idx) and np.array_equal(ds, d)
            per_seed.append(hits)
        raw[arm] = np.stack(per_seed)
    out = {"images": len(ds), "captions": int(ds.sum()), "levels": {}, "paired": {}}
    for arm, scores in raw.items():
        out["levels"][arm] = summary(scores.mean(0), ds)
    for left, refs in {"teacher": ["parent", "retained"], "reachable_teacher": ["teacher", "retained"],
                       "strict_warmstrict": ["retained", "parent"], "strict_coldstrict": ["strict_coldordinary", "retained"]}.items():
        out["paired"][left] = {r: summary((raw[left]-raw[r]).mean(0), ds) for r in refs}
    dump(ROOT / "results/ucm_alternatives.json", out)


if __name__ == "__main__":
    main()

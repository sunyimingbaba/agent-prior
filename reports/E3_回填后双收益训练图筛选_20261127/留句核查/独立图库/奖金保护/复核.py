"""Remove only a pre-existing detail bonus when visible captions fall below R2."""
from collections import defaultdict
from fractions import Fraction
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from 复算 import REPORTS, OLD, COMPARISON, bank_inputs, jsonl, readout, load, read, sha
from check import summarize
from compare import graded_bonus, original_mapping


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def fixed_gate(rr, hidden, base_winners):
    visible_sums = [
        sum(Fraction.from_float(float(v)) for j, v in enumerate(row) if j != hidden)
        for row in rr
    ]
    base_total = sum(visible_sums[j] for j in base_winners)
    return np.array([x * len(base_winners) >= base_total for x in visible_sums])


def main():
    torch.set_num_threads(1)
    old_proof_path = OUT.parent / "输入核验.json"
    proof = read(old_proof_path)
    for rel, signature in proof["inputs"].items():
        assert sha(REPORTS / rel) == signature, rel
    assert proof["roster_images"] == 98 and proof["overlap_by_rgb"] == 0 and proof["groups"] == 392
    rows = read(OLD / "inputs/real_roster.json")
    historical = COMPARISON / "分差核对98逐组.jsonl.gz"
    records = {(r["image"], r["seed"], r["block"]):r for r in jsonl(historical)}
    bank = bank_inputs()
    match = original_mapping()
    cases = defaultdict(list)
    nblocks = 0
    for seed in "ab":
        for row in rows:
            name = row["image"]
            path = OLD / "records/real" / seed / "features" / (name + ".pt")
            assert sha(path) == proof["records"][seed][name]["features"]
            feature = load(path)
            assert feature["source"] == proof["records"][seed][name]["raw"] and feature["rgb"] == row["rgb_sha256"]
            metrics = readout(row, feature, bank)
            captions = metrics["other"]
            assert len(captions) >= 2
            for block in (0, 1):
                group = records[name, seed, block]
                sl = slice(8 * block, 8 * block + 8)
                r2 = np.array(group["r2"])
                bonus = np.array(group["bonus"])
                hard = metrics["hard"][sl]
                rr = metrics["rr"][sl]
                assert np.array_equal(match(metrics["margin"][sl], 1 / (1 + np.array(feature["ranks"][sl]))), r2)
                assert np.array_equal(graded_bonus(r2, hard[:, :2]), bonus)
                r3 = r2 + bonus
                base = np.flatnonzero(r2 == r2.max())
                sealed = np.flatnonzero(r3 == r3.max())
                assert base.tolist() == group["winners_r2"] and sealed.tolist() == group["winners_r3"]
                de = (hard[:, 2:] > 0).mean(1)
                fg = (metrics["fg"][sl] > 0).mean(1)
                assert np.array_equal(de, group["outcomes"]["DE"]) and np.array_equal(fg, group["outcomes"]["FG"])
                assert np.allclose(rr.mean(1), group["outcomes"]["mrr"], atol=1e-15, rtol=0)
                nblocks += 1
                for hidden, caption in enumerate(captions):
                    allowed = fixed_gate(rr, hidden, base)
                    kept_bonus = np.where(allowed, bonus, 0.)
                    updated = r2 + kept_bonus
                    chosen = np.flatnonzero(updated == updated.max())
                    trial = dict(seed=seed, block=block, hidden_id=caption, hidden_index=hidden,
                                 had_bonus_zeroed=bool(np.any(kept_bonus != bonus)),
                                 bonus_zeroed_slots=np.flatnonzero(kept_bonus != bonus).tolist(),
                                 selected_slots=chosen.tolist())
                    for ref_name, ref in (("R2", base), ("R3", sealed)):
                        difference = dict(switched=int(not np.array_equal(chosen, ref)),
                                          held_rr=float(rr[chosen, hidden].mean() - rr[ref, hidden].mean()),
                                          DE=float(de[chosen].mean() - de[ref].mean()),
                                          FG=float(fg[chosen].mean() - fg[ref].mean()))
                        trial[ref_name] = difference
                        trial[ref_name + "_slots"] = ref.tolist()
                    cases[name].append(trial)
    assert nblocks == 392
    panel = [dict(image=r["image"], cohort="history98", pair=r["pair"], trials=cases[r["image"]]) for r in rows]
    for r in panel:
        ts = r["trials"]
        assert len(ts) == 4 * len({t["hidden_id"] for t in ts})
        assert len({(t["seed"], t["block"], t["hidden_id"]) for t in ts}) == len(ts)
    def summary(base, seed=None):
        return summarize([dict(r, trials=[t[base] for t in r["trials"] if seed is None or t["seed"] == seed]) for r in panel])
    with gzip.open(OUT / "逐图留句.jsonl.gz", "wt", encoding="utf-8") as gz:
        for x in panel:
            gz.write(json.dumps(x, ensure_ascii=False) + "\n")
    results = {base:summary(base) for base in ("R2", "R3")}
    results["by_seed"] = {s:{base:summary(base, s) for base in ("R2", "R3")} for s in "ab"}
    assert all(results[base]["held_trials"] == 1516 and results[base]["images"] == 98 for base in ("R2", "R3"))
    save("summary.json", results)
    save("输入摘要.json", dict(parent_verification=sha(old_proof_path), source=sha(Path(__file__)),
         protocol=sha(OUT / "口径.md"), archived_groups=sha(historical),
         reward_code=sha(COMPARISON.parent / "code/detail_reward.py"), groups_reproduced=nblocks,
         trials=1516, images=98, natural_pairs=49))
    for base in ("R2", "R3"):
        print(base, json.dumps(results[base]["metrics"]))


if __name__ == "__main__":
    main()

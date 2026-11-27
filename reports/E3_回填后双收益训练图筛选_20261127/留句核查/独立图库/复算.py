"""Rebuild the individual caption scores from archived 98-image retrieval vectors."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
STAGE = OUT.parents[1]
REPORTS = STAGE.parent
OLD = REPORTS / "E3_R3首位配文保护与首句复验_20261126"
COMPARISON = REPORTS / "E3_R2逐候选分值与困难细节加分_20261127/candidates"
sys.path[:0] = [str(OUT.parent), str(COMPARISON)]
from check import held_trials, summarize
from compare import LIBRARIES, NATIVE, OLD as PANEL270, original_mapping, sha, load, read


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def jsonl(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def tie_rr(negative_scores, true_score):
    higher = int((negative_scores > true_score).sum())
    equal = int((negative_scores == true_score).sum())
    return float(np.mean(1 / np.arange(higher + 1, higher + equal + 2)))


def bank_inputs():
    native = load(NATIVE / "features.pt")
    native_index = read(NATIVE / "index.json")
    assert sha(NATIVE / "features.pt") == native_index["sha256"]
    tokens = native["tokens"].tolist()
    import hashlib
    even = np.array([int(hashlib.sha256(",".join(map(str, q)).encode()).hexdigest()[-1], 16) % 2 == 0
                     for q in tokens])
    bank_path = LIBRARIES["v122"].parent / "reward_confirm/records/text.pt"
    bank = load(bank_path)
    ids = {tuple(q): i for i, q in enumerate(tokens)}
    remap = np.array([ids[tuple(q)] for q in bank["native_tokens"].tolist()])
    assert len(set(remap)) == len(remap) == len(tokens)
    fresh = load(PANEL270 / "inputs/fresh_text.pt")
    fresh_path = PANEL270 / "inputs/独立措辞.json"
    assert fresh["source"] == sha(fresh_path)
    return dict(bank=bank, path=bank_path, text=native["features"].double().numpy(),
                columns=native_index["columns"], remap=remap, even=even, fresh=read(fresh_path),
                word_ids={w: i for i, w in enumerate(fresh["words"])},
                fresh_features=fresh["features"].double().numpy(), hard=bank["hard"].double().numpy())


def readout(row, feature, bank):
    name = row["image"]
    marks = bank["bank"]["maps"][name]
    pos = bank["remap"][marks["native"]]
    assert pos.tolist() == [bank["columns"][q] for q in row["captions"]]
    for key in "ABDE":
        assert [bank["bank"]["texts"][i] for i in marks[key]] == row[key]
    other = sorted(set(pos) - {pos[0]})
    fit = bank["even"].copy()
    test = ~fit
    fit[pos] = test[pos] = False
    unique, inv = np.unique(feature["visual"][:16].double().numpy(), axis=0, return_inverse=True)
    sim = (unique @ bank["text"].T)[inv]
    margin = sim[:, pos[0]] - sim[:, fit].max(1)
    rr = np.array([[tie_rr(v[test], v[k]) for k in other] for v in sim])
    keys = np.array([marks[k] for k in "ABDE"])
    h = (unique @ bank["hard"][keys.ravel()].T)[inv].reshape(16, 4, 2)
    signed = h[:, :, 0] - h[:, :, 1]
    fresh = None
    if row["attribute"] in bank["fresh"]:
        w = [bank["word_ids"][x] for phrase in bank["fresh"][row["attribute"]] for x in phrase]
        f = (unique @ bank["fresh_features"][w].T)[inv].reshape(16, 2, 2)
        side = row["side"]
        fresh = f[:, :, side] - f[:, :, 1 - side]
    return dict(other=[int(v) for v in other], rr=rr, margin=margin, hard=signed, fg=fresh)


def main():
    torch.set_num_threads(1)
    roster_file = OLD / "inputs/real_roster.json"
    other_file = PANEL270 / "inputs/real_roster.json"
    rows, rows270 = read(roster_file), read(other_file)
    assert sha(roster_file) == read(OLD / "inputs/real_lock.json")["roster"]
    assert sha(other_file) == read(PANEL270 / "inputs/real_lock.json")["roster"]
    ids = {r["image"] for r in rows}
    pixels = {r["rgb_sha256"] for r in rows}
    assert len(rows) == len(ids) == len(pixels) == 98 and len(rows270) == 270
    assert not pixels.intersection(r["rgb_sha256"] for r in rows270)
    assert all(r["split"] == "train" and r["captions"][0] == r["orig_query"] for r in rows)
    assert set(Counter(r["pair"] for r in rows).values()) == {2}
    historical = COMPARISON / "分差核对98逐组.jsonl.gz"
    old_groups = {(r["seed"], r["block"], r["image"]): r for r in jsonl(historical)}
    assert len(old_groups) == 392
    arch = read(OLD / "archives/index.json")
    bank = bank_inputs()
    samples = defaultdict(list)
    other_ids = {}
    roots = {s: {} for s in "ab"}
    sampler = REPORTS / "E3_UCM中性观察问题候选对照_20261125/code/run.py"
    entry = REPORTS / "E3_普通配文连续扣分与独立检验_20261125/code/real_remote.py"
    inputs = [OUT / "口径.md", OUT.parent / "check.py", OUT.parent / "口径.md",
              COMPARISON / "tiebreak.py", COMPARISON / "compare.py", historical, roster_file, other_file,
              OLD / "archives/index.json", OLD / "inputs/real_lock.json", PANEL270 / "inputs/real_lock.json",
              OLD / "code/sample.py", entry, sampler,
              REPORTS / "E3-2_UCM回填后回答训练_20261124/code/train.py",
              NATIVE / "features.pt", NATIVE / "index.json", bank["path"], PANEL270 / "inputs/fresh_text.pt",
              PANEL270 / "inputs/独立措辞.json", Path(__file__)]
    match = original_mapping()
    for seed in "ab":
        root = OLD / "records/real" / seed
        raw_manifest = root / "raw/complete.json"
        receipt = read(raw_manifest)
        assert receipt["adapter"] == arch[seed]["adapter"] and sha(raw_manifest) == arch[seed]["raw_manifest"]
        assert receipt["sampler"] == sha(sampler) and receipt["entry"] == sha(entry)
        assert sha(OLD / "archives" / (seed + ".tar.gz")) == arch[seed]["archive_sha256"]
        manifests = sorted((root / "features").glob("complete*.json"))
        features = {name: sig for path in manifests for name, sig in read(path)["files"].items()}
        assert set(features) == set(receipt["files"]) == ids
        inputs.extend([raw_manifest, *manifests, OLD / "archives" / (seed + ".tar.gz")])
        for row in rows:
            name = row["image"]
            rawpath = root / "raw" / (name + ".pt")
            fpath = root / "features" / (name + ".pt")
            assert sha(rawpath) == receipt["files"][name] and sha(fpath) == features[name]
            raw, feature = load(rawpath), load(fpath)
            assert feature["source"] == receipt["files"][name]
            assert raw["rgb"] == feature["rgb"] == row["rgb_sha256"]
            assert raw["adapter"] == feature["adapter"] == arch[seed]["adapter"]
            assert raw["query"] == row["orig_query"] and raw["feedback"] == row["feedback"] and raw["image"] == name
            assert len(raw["paths"]) == 16 and raw["hidden"].shape == (17, 3584) and feature["visual"].shape == (17, 768)
            assert raw["rngs"] == [row["rng_" + seed] + i * 100 for i in range(2)]
            roots[seed][name] = dict(raw=sha(rawpath), features=sha(fpath), rngs=raw["rngs"])
            metrics = readout(row, feature, bank)
            previous = other_ids.setdefault(name, metrics["other"])
            assert metrics["other"] == previous
            for block in (0, 1):
                part = slice(8 * block, 8 * block + 8)
                old = old_groups[seed, block, name]
                ranks = np.array(feature["ranks"][part])
                r2 = match(metrics["margin"][part], 1 / (1 + ranks))
                hard, rr = metrics["hard"][part], metrics["rr"][part]
                fg = metrics["fg"][part] if metrics["fg"] is not None else None
                assert np.array_equal(r2, old["r2"])
                assert np.array_equal((hard[:, 2:] > 0).mean(1), old["outcomes"]["DE"])
                if fg is not None:
                    assert np.array_equal((fg > 0).mean(1), old["outcomes"]["FG"])
                if len(metrics["other"]):
                    assert np.allclose(rr.mean(1), old["outcomes"]["mrr"], atol=1e-15, rtol=0)
                baseline = np.flatnonzero(r2 == r2.max()).tolist()
                assert baseline == old["winners_r2"]
                items = [dict(hard_margins=h[:2].tolist(), DE=h[2:].tolist(), other_caption_rr=v.tolist(),
                              FG=None if fg is None else fg[i].tolist()) for i, (h, v) in enumerate(zip(hard, rr))]
                group = dict(items=items, r2=r2.tolist(), r2_top=baseline)
                for trial in held_trials(group):
                    samples[name].append(dict(seed=seed, block=block, **trial))
    panel = []
    for row in rows:
        name = row["image"]
        trials = samples[name]
        if len(other_ids[name]) < 2:
            assert not trials
            continue
        assert len(trials) == 4 * len(other_ids[name])
        counts = Counter((tr["seed"], tr["block"]) for tr in trials)
        assert len(counts) == 4 and set(counts.values()) == {len(other_ids[name])}
        panel.append(dict(image=name, cohort="history98", pair=row["pair"], rgb=row["rgb_sha256"],
                          other_ids=other_ids[name], trials=trials))
    with gzip.open(OUT / "逐图留句.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in panel:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    save("输入核验.json", dict(inputs={str(p.relative_to(REPORTS)): sha(p) for p in inputs}, records=roots,
         roster_images=98, reference_images=270, overlap_by_rgb=0, policy="deployed R2, full rollout",
         adapters={s: arch[s]["adapter"] for s in "ab"}, groups=392, archived_groups_concordant=392))
    summary = summarize(panel)
    summary["FG_pairs"] = len({r["pair"] for r in panel if "FG" in r["trials"][0]})
    summary["excluded_images"] = {r["image"]: len(other_ids[r["image"]]) for r in rows if len(other_ids[r["image"]]) < 2}
    summary["by_seed"] = {s: summarize([dict(r, trials=[t for t in r["trials"] if t["seed"] == s]) for r in panel])
                          for s in "ab"}
    save("summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("images", "pairs", "held_trials", "metrics", "FG_pairs")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Count changes of the actual top R2 slots under the frozen full-caption training rule."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT.parents[1]
POOL = REPORTS / "E3_回填后双收益训练图筛选_20261127"
PREVIOUS = REPORTS / "E3_困难奖励跨图跨措辞校准_20261127"
sys.path.insert(0, str(PREVIOUS / "code"))
from reward import redistribute


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return (row["cohort"], row["pair"])


def matched_control_capacity(repeated, never, fields):
    """At most one control per unused natural pair, with exact stratum agreement."""
    prohibited = set(map(key, repeated))
    options = {r["image"]: sorted({key(c) for c in never if key(c) not in prohibited
                                   and all(c[f] == r[f] for f in fields)}) for r in repeated}
    assigned = {}

    def assign(image, visited):
        for pair in options[image]:
            if pair in visited:
                continue
            visited.add(pair)
            if pair not in assigned or assign(assigned[pair], visited):
                assigned[pair] = image
                return True
        return False

    return sum(assign(image, set()) for image in options)


def replay(group):
    if not group["eligible"]:
        return dict(eligible=False, changed=None)
    r2 = np.asarray(group["r2"])
    hard = np.asarray([item["hard_margins"] for item in group["items"]])
    rr = [item["other_caption_rr"] for item in group["items"]]
    assert r2.shape == (8,) and hard.shape == (8, 2)
    assert len(set(map(len, rr))) == 1
    weak = hard.min(axis=1)
    correct = (hard > 0).sum(axis=1)
    top = r2 == r2.max()
    valid = correct * top.sum() > correct[top].sum()
    signal = np.where(valid | top, weak, weak.min())
    updated = redistribute(r2, signal, rr)
    new_top = updated == updated.max()
    return dict(eligible=True, changed=not np.array_equal(new_top, top), visible=len(rr[0]),
                original_slots=np.flatnonzero(top).tolist(), strict_slots=np.flatnonzero(new_top).tolist())


def main():
    archive = POOL / "scoring/results/candidate_scores.jsonl.gz"
    inputs = POOL / "scoring/inputs/rows.json"
    roster = POOL / "scoring/locks/training.json"
    rule = PREVIOUS / "code/reward.py"
    manifest = read(POOL / "scoring/inputs/manifest.json")
    assert sha(archive) == read(POOL / "scoring/results/score_archive.json")["sha256"]
    assert sha(inputs) == manifest["compiled"]["rows.json"]
    assert sha(roster) == (roster.with_suffix(".sha256")).read_text().strip()
    assert sha(rule) == read(ROOT.parent / "training/strict_lock.json")["redistribution"]
    rows = [{k: r[k] for k in ("image", "rgb_sha256", "cohort", "pair", "attribute", "side")} for r in read(inputs)]
    assert len(rows) == len({r["rgb_sha256"] for r in rows}) == 270
    pairs = defaultdict(list)
    for row in rows:
        pairs[key(row)].append(row)
    assert len(pairs) == 135 and all({r["side"] for r in pair} == {0, 1} for pair in pairs.values())
    train = read(roster)["rows"]
    train_sha = {r["rgb_sha256"] for r in train}
    train_pairs = set(map(key, train))
    candidate_panel = PREVIOUS / "inputs/test_panel.json"
    panel_rows = read(candidate_panel)
    old_candidate_roster = {r["rgb_sha256"] for r in panel_rows}
    assert len(train_sha) == 24 and len(old_candidate_roster) == 194 and train_sha.isdisjoint(old_candidate_roster)
    inspected = {s: {} for s in "ab"}
    with gzip.open(archive, "rt") as source:
        for line in source:
            g = json.loads(line)
            inspected[g["seed"]][g["rgb"]] = replay(g)
    assert all(len(s) == 270 for s in inspected.values())
    annotated = [dict(**r, old24=r["rgb_sha256"] in train_sha, pair_in_old24=key(r) in train_pairs,
                      old_candidate_validation194=r["rgb_sha256"] in old_candidate_roster,
                      draws={s: inspected[s][r["rgb_sha256"]] for s in "ab"}) for r in rows]
    eligible = [r for r in annotated if all(x["eligible"] for x in r["draws"].values())]
    both = [r for r in eligible if all(x["changed"] for x in r["draws"].values())]
    neither = [r for r in eligible if all(not x["changed"] for x in r["draws"].values())]
    both.sort(key=lambda r: (r["attribute"], key(r), r["side"]))
    repeat_pairs = set(map(key, both))
    repeat_attr = set(r["attribute"] for r in both)
    for row in both:
        row["matched_never_change_count"] = sum(
            all(c[f] == row[f] for f in ("cohort", "attribute", "side")) and key(c) not in repeat_pairs
            for c in neither)
    control_cap = {"+".join(fields): matched_control_capacity(both, neither, fields)
                   for fields in (("cohort", "attribute", "side"), ("attribute", "side"))}
    result = dict(roster=len(rows), pairs=len(pairs), old24_pairs=len(train_pairs), two_eligible=len(eligible),
                  seeds={s: dict(eligible=sum(x["eligible"] for x in inspected[s].values()),
                            changed=sum(x["changed"] is True for x in inspected[s].values()),
                            old24_changed=sum(r["draws"][s]["changed"] is True for r in annotated if r["old24"]))
                         for s in "ab"}, both_triggered=len(both),
                  exactly_one_seed_triggered=sum(sum(v["changed"] for v in r["draws"].values())==1 for r in eligible),
                  neither_seed_triggered=len(neither), repeat_pairs=len(repeat_pairs), repeat_attributes=len(repeat_attr),
                  attributes=dict(sorted(Counter(r["attribute"] for r in both).items())),
                  sides=dict(Counter(r["side"] for r in both)), cohorts=dict(Counter(r["cohort"] for r in both)),
                  images_in_old24=sum(r["old24"] for r in both), pairs_in_old24=len(repeat_pairs & train_pairs),
                  attrs_in_old24=sorted(repeat_attr & {r["attribute"] for r in train}),
                  unique_control_pairs_matched=control_cap,
                  side_counts_by_attribute={attr: dict(Counter(r["side"] for r in both if r["attribute"] == attr))
                                            for attr in sorted(repeat_attr)},
                  images_in_old_candidate_validation194=sum(r["old_candidate_validation194"] for r in both),
                  old_candidate_validation_images_in_affected_pairs=sum(key(r) in repeat_pairs for r in panel_rows),
                  rows=both, sources={str(p.relative_to(REPORTS)): sha(p) for p in (archive, inputs, roster, rule, candidate_panel)})
    (ROOT / "复算.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print({k: v for k, v in result.items() if k not in ("rows", "sources")})
    print([(r["image"], r["cohort"], r["pair"], r["attribute"], r["side"], r["old24"], r["pair_in_old24"],r["matched_never_change_count"]) for r in both])


if __name__ == "__main__":
    main()

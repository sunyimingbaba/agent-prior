"""Trace full-caption RFPO switches through genuine leave-one-caption-out choices."""
from collections import Counter
import json

import numpy as np
import torch

from cold_analysis import sources
from data import PREV, ROOT, best, packets, read, rr, sha
from online_reward import old_choices
from training_opportunities import winners


def visible_choices(row, bank, visual, coef):
    """Recover actual winner indices with all available captions; evaluation uses D/E only."""
    ids = bank["maps"][row["image"]]
    native_ids = ids["native"]
    other = sorted(set(native_ids) - {native_ids[0]})
    negative = ~bank["even"].copy()
    negative[native_ids] = False
    sim = visual[:16] @ bank["native"].T
    pos, neg = sim[:, other], sim[:, negative]
    ranks = np.array([[rr(s, pool) for s in values] for values, pool in zip(pos, neg)], dtype=object)
    average = ranks.mean(1)
    nearest = np.partition(neg, -8, axis=1)[:, -8:]
    gaps = pos - nearest.max(1)[:, None]
    base = np.column_stack((average.astype(float), ranks.min(1).astype(float), pos.mean(1), pos.min(1),
                            gaps.mean(1), gaps.min(1), pos.mean(1) - nearest.mean(1)))
    hard = visual[:16] @ bank["hard"][np.array([ids[k] for k in "ABCDE"]).ravel()].T
    margins = hard[:, ::2] - hard[:, 1::2]
    groups = []
    for start in (0, 8):
        sl = slice(start, start + 8)
        packet = {"ref": best(average[sl], range(8)), "exact": average[sl]}
        pred = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            h = margins[sl][:, [i, j]]
            x = np.column_stack((h, h.min(1), h > 0, (h > 0).all(1), base[sl]))
            pred.append((x - x.mean(0)) @ coef)
        old = old_choices(pred[0], packet)
        new = old_choices(np.mean(pred, axis=0), packet)
        detail = margins[sl, 3:] > 0
        groups.append({"old": old, "old3": new, "DE_delta": float(detail[new].mean() - detail[old].mean())})
    return groups


def audit_image(source, coef):
    row, seed, view, bank, visual = source
    cols = bank["maps"][row["image"]]["native"]
    names = dict(zip(cols, row["captions"]))
    others = sorted(set(cols) - {cols[0]})
    full = visible_choices(row, bank, visual, coef)
    check = winners(row, bank, visual, coef)
    records = []
    for block, choice in enumerate(full):
        changed = choice["old"] != choice["old3"]
        assert changed == check[block]["old3-old"]
        records.append({"image":row["image"], "seed":seed, "block":block, "pair":row["pair"],
                        "match":row["match"], "status":row["status"], "full":choice, "full_changed":changed, "folds":[]})
    for packet in packets([(row, seed, view, bank, visual)]):
        pred = packet["old"] @ coef
        old = old_choices(pred[0], packet)
        new = old_choices(pred.mean(0), packet)
        block = packet["block"]
        delta = {k:float(packet[k][new].mean() - packet[k][old].mean()) for k in ("fine", "mrr")}
        records[block]["folds"].append({
            "held_caption":names[others[packet["held"]]], "held_bank_id":int(others[packet["held"]]),
            "old":old, "old3":new, "changed":old != new, "same_two_sets_as_full":(
                old == full[block]["old"] and new == full[block]["old3"]),
            "DE_delta":delta["fine"], "MRR_delta":delta["mrr"]})
    return records


def direction(detail, mrr):
    eps = 1e-12
    d = int(detail > eps) - int(detail < -eps)
    m = int(mrr > eps) - int(mrr < -eps)
    if d == m == 0:
        return "tied"
    if d + m == 0:
        return "tradeoff"
    return "gain_without_loss" if d+m > 0 else "loss_without_gain"


def summarize(records):
    folds = [f for r in records for f in r["folds"]]
    changed = [f for f in folds if f["changed"]]
    evaluable = [r for r in records if r["folds"]]
    return {
        "groups":len(records), "fold_evaluable_groups":len(evaluable), "held_folds":len(folds),
        "groups_with_a_held_switch":sum(any(f["changed"] for f in r["folds"]) for r in records),
        "held_switches":len(changed),
        "held_switches_with_exact_full_choices":sum(f["same_two_sets_as_full"] for f in changed),
        "full_DE_delta":float(np.mean([r["full"]["DE_delta"] for r in records])) if records else None,
        "held_delta_group_equal":{k:float(np.mean([np.mean([f[k] for f in r["folds"]]) for r in evaluable]))
                                  if evaluable else None for k in ("DE_delta", "MRR_delta")},
        "held_switch_outcomes":dict(Counter(direction(f["DE_delta"], f["MRR_delta"]) for f in changed)),
        "held_both_strictly_improved":sum(f["DE_delta"] > 1e-12 and f["MRR_delta"] > 1e-12 for f in changed),
        "held_both_strictly_declined":sum(f["DE_delta"] < -1e-12 and f["MRR_delta"] < -1e-12 for f in changed)}


def main():
    folder = ROOT / "cold/inputs"
    lock = read(folder / "real_lock.json")
    assert sha(folder / "real_roster.json") == lock["roster"]
    roster = read(folder / "real_roster.json")
    model = read(PREV / "锁定评分参数.json")["model"]
    coef = np.array(model["coef"]) / np.array(model["scale_x"])[:, None]
    records = []
    for source in sources(roster, lock):
        records.extend(audit_image(source, coef))
    prior = {(x["image"], x["seed"], x["block"]): x for x in read(ROOT / "results/冷启动预筛逐组.json")}
    assert len(prior) == len(records) == len(roster)*4
    for r in records:
        old = prior[r["image"], r["seed"], r["block"]]
        assert old["old3-old"] == r["full_changed"]
        if r["folds"]:
            for k, reported in (("DE_delta", "held_detail_old3-old"), ("MRR_delta", "held_mrr_old3-old")):
                assert abs(np.mean([f[k] for f in r["folds"]])-old[reported]) < 1e-12
    summary = {}
    for status in ("all", "selected", "control"):
        subset = [r for r in records if status=="all" or r["status"]==status]
        summary[status] = {name:summarize([r for r in subset if r["full_changed"] == change])
                           for name,change in (("full_switch", True), ("full_same", False))}
    out = {"roster_sha256":lock["roster"], "summary":summary, "records":records}
    (ROOT/"results/冷启动改选逐留句核查.json").write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

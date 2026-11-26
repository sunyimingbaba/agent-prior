"""Evaluate a trained-image candidate selection rule on held images with one positive caption held out."""
import numpy as np
import torch
from local import ROOT, read, load, dump, digest, feature_path, banks, bank_for, summary_by_seed, paired
from margin_scores import rr


def top(ids, scores):
    return max(ids, key=scores.__getitem__) if ids else 16


def image_folds(row, seed, view, pack, odd):
    z = load(feature_path(row, seed, view))
    assert z["rgb"] == row["rgb_sha256"]
    vectors = z["visual"].double()
    positions = pack["maps"][row["image"]]
    positives = positions["native"]
    alts = sorted(set(positives) - {positives[0]})
    if len(alts) < 2:
        return []
    native = (vectors @ pack["native"].double().T).numpy()
    detail = (vectors @ pack["hard"].double().T).numpy()
    score_sets = []
    for subset in (odd, ~odd):
        negatives = subset.copy()
        negatives[positives] = False
        score_sets.append([[rr(scores[j], scores[negatives]) for j in alts] for scores in native])
    train_rr, eval_rr = score_sets
    correct = np.column_stack([detail[:, positions[k][0]] > detail[:, positions[k][1]] for k in "ABCDE"])
    ab = correct[:, :2].sum(1)
    cde = correct[:, 2:].sum(1)
    result = []
    for j, caption in enumerate(alts):
        train = [(sum(r) - r[j]) / (len(alts) - 1) for r in train_rr]
        ordinary = [i for i in range(16) if train[i] > train[16]]
        joint = [i for i in ordinary if ab[i] > ab[16]]
        choices = {"greedy": 16, "ordinary": top(ordinary, train), "teacher": top(joint, train),
                   "matched": top(ordinary, train) if joint else 16}
        fold = {}
        for name, idx in choices.items():
            score = eval_rr[idx][j]
            det = int(cde[idx]);delta = det - int(cde[16])
            fold[name] = dict(slot=idx, mrr=float(score), CDE=det / 3,
                              win=int(delta > 0 and score > eval_rr[16][j]),
                              loss=int(delta < 0 and score < eval_rr[16][j]))
        result.append(dict(seed=seed, image=row["image"], view=view, caption=caption, choices=fold))
    return result


def main():
    rows = read(ROOT / "inputs/holdout.json")
    packs = banks()
    odd = np.array([int(digest(",".join(map(str, x)))[-1], 16) % 2 == 1
                    for x in packs[0]["native_tokens"].tolist()])
    result = {}; records = []
    for view in ("scene", "first"):
        by = {arm: {} for arm in ("greedy", "ordinary", "matched", "teacher")}
        for seed in "ab":
            per = {arm: [] for arm in by}; labels = []; skipped = []
            for row in rows:
                folds = image_folds(row, seed, view, bank_for(row, packs), odd)
                if not folds:
                    skipped.append(row["image"])
                    continue
                labels.append(row["pair"]); records.extend(folds)
                for arm in per:
                    per[arm].append([np.mean([f["choices"][arm][k] for f in folds]) for k in ("mrr", "CDE", "win", "loss")])
            for arm, values in per.items():
                by[arm][seed] = dict(zip(("mrr", "CDE", "win", "loss"), np.array(values).T))
            if seed == "a":
                keep_labels = labels
            assert keep_labels == labels
        result[view] = dict(images=len(labels), pairs=len(set(labels)), skipped=skipped,
            folds=sum(r["view"] == view for r in records),
            levels={a: summary_by_seed(v, labels) for a, v in by.items()},
            teacher_minus={a: paired(by["teacher"], v, labels) for a, v in by.items() if a != "teacher"})
    dump(ROOT / "results/leave_caption.json", result)
    dump(ROOT / "results/leave_caption_folds.json", records)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

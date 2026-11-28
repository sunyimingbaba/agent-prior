"""Measure tie-aware top-one retrieval when every distractor belongs to the same UCM class."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import report as evidence

VARIANTS = [("strict", "real"), ("strict", "shuffled"), ("strict", "none"),
            ("r2", "real"), ("r2", "none")]
CONTRASTS = [("strict_real", "strict_none"), ("strict_real", "strict_shuffled"),
             ("strict_shuffled", "strict_none"), ("strict_real", "r2_real"),
             ("r2_real", "r2_none")]


def scene(row):
    # The UCM numbered filenames follow the project class convention, 100 images per class.
    return (int(Path(row["image"]).stem) - 1) // 100


def top_one(scores, target):
    tied = scores == scores.max(axis=1, keepdims=True)
    return (tied & target).sum(axis=1) / tied.sum(axis=1)


def directions(sims):
    n = len(sims)
    assert sims.shape == (n, n * 5)
    target = np.repeat(np.eye(n, dtype=bool), 5, axis=1)
    i2t = top_one(sims, target)
    t2i = top_one(sims.T, target.T).reshape(n, 5).mean(1)
    return {"I2T": i2t, "T2I": t2i, "both": (i2t + t2i) / 2}


def evaluate(vectors, texts, columns, classes, eligible):
    matrix = evidence.scoring.extra.fixed(vectors, texts)[:, columns]
    global_ = directions(matrix)
    within = {key: np.full(len(classes), np.nan) for key in global_}
    for label in np.unique(classes[eligible]):
        group = classes == label
        block = matrix[group][:, np.repeat(group, 5)]
        for key, val in directions(block).items():
            within[key][group] = val
    assert all(np.isfinite(a[eligible]).all() for a in within.values())
    results = {k+"_"+m: a[eligible] for k,z in (("whole", global_), ("within", within)) for m,a in z.items()}
    return results,global_


def main(baseline_only):
    root = HERE.parent
    read = evidence.read
    lock = read(root/"inputs/manifest.json")
    donor = read(root/"inputs/assignments.json")
    feedback = read(evidence.BASELINE/"feedback_verified.json")["checks"]
    roster = read(evidence.BASELINE/"inputs/roster.json")
    official = read(evidence.FORM/"inputs/ucm.json")
    extra = read(evidence.OLD/"extra164/inputs/rows.json")
    fine = read(evidence.FINE/"inputs/panel.json")
    known = {r["rgb_sha256"]: int(r["scene_id"]) for r in fine+extra}
    assert sum(r["rgb_sha256"] in known for r in official)==106
    assert all(scene(r)==known[r["rgb_sha256"]] for r in official+extra if r["rgb_sha256"] in known)
    formal_text = torch.load(evidence.FORM/"results/ucm_text.pt", weights_only=True)
    extra_text = torch.load(evidence.OLD/"extra164/inputs/texts.pt", weights_only=True)
    base = read(evidence.BASELINE/"results/单次检索.json")
    variants = [v for v in VARIANTS if v[1]=="none"] if baseline_only else VARIANTS
    output = {}
    for panel, rows, texts, cols in (
        ("official", official, formal_text["native"], formal_text["caption_order"]),
        ("extra", extra, extra_text["captions"], [i for r in extra for i in r["panel_positive_ids"]])):
        batch = [r for r in roster if r["panel"]==panel]
        assert [r["rgb_sha256"] for r in rows]==[r["rgb"] for r in batch]
        classes = np.array([scene(r) for r in rows]); counts = Counter(classes.tolist())
        eligible = np.array([counts[c]>1 for c in classes])
        labels = [r["pair"] if panel=="extra" else r["image"] for r,sel in zip(rows,eligible) if sel]
        summarize = evidence.extra_stats if panel=="extra" else evidence.summary_by_seed
        groups = {}
        for model, condition in variants:
            tag = model+"_"+condition
            groups[tag] = {}
            for seed in "ab":
                arm = seed+"_"+model
                v,_ = evidence.data(root,arm,condition,panel,batch,lock,donor,lock["policies"][arm]["adapter"],feedback)
                groups[tag][seed],reference = evaluate(v,texts,cols,classes,eligible)
                if condition=="none":
                    for direction in ("I2T","T2I"):
                        ref = base["official" if panel=="official" else "extra164"][model][direction+"_R1"]
                        assert abs(reference[direction].mean()-ref["seeds"]["ab".index(seed)]) < 1e-12
        differences = {}
        for end,start in CONTRASTS:
            if end in groups and start in groups:
                delta = {s:{k:vec-groups[start][s][k] for k,vec in groups[end][s].items()} for s in "ab"}
                differences[end+"-"+start] = summarize(delta,labels)
        output[panel] = dict(n=int(eligible.sum()),units=len(set(labels)),class_sizes=counts,
                             verified_scene_ids=sum(r["rgb_sha256"] in known for r in rows),
                             levels={k:summarize(z,labels) for k,z in groups.items()},differences=differences)
    evidence.dump(HERE/("基线核对.json" if baseline_only else "结果.json"),output)


if __name__=="__main__":
    torch.set_num_threads(1)
    main("--baseline-only" in sys.argv)

"""Choose with available captions and independently score the hidden caption."""
import gzip
import json
from pathlib import Path

import numpy as np


STAGE = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ROSTER = STAGE.parent / "E3_R3改选收益校准与双批首句验证_20261126/inputs/real_roster.json"


def read(path):
    return json.loads(path.read_text())


def held_trials(group):
    items = group["items"]
    ab = np.asarray([x["hard_margins"] for x in items], dtype=float)
    rr = np.asarray([x["other_caption_rr"] for x in items], dtype=float)
    de = np.asarray([x["DE"] for x in items], dtype=float)
    fg = [x.get("FG") for x in items]
    r2 = np.asarray(group["r2"])
    top = np.flatnonzero(r2 == r2.max())
    assert len(items) == 8 and ab.shape == de.shape == (8, 2)
    assert set(top) == set(group["r2_top"])
    if rr.shape[1] < 2:
        return []
    correct = (ab > 0).mean(axis=1)
    better = correct > correct[top].mean()
    fine = (de > 0).mean(axis=1)
    fresh = None if fg[0] is None else (np.asarray(fg) > 0).mean(axis=1)
    trials = []
    for hidden in range(rr.shape[1]):
        visible = np.delete(rr, hidden, axis=1).mean(axis=1)
        possible = np.flatnonzero(better & (visible >= visible[top].mean() - 1e-12))
        chosen = top
        if len(possible):
            weaker = ab.min(axis=1)
            strongest = possible[weaker[possible] == weaker[possible].max()]
            chosen = strongest[r2[strongest] == r2[strongest].max()]
        hidden_scores = rr[:, hidden]
        caption = hidden_scores[chosen].mean() - hidden_scores[top].mean()
        detail = fine[chosen].mean() - fine[top].mean()
        entry = dict(held_index=hidden, base_slots=top.tolist(), selected_slots=chosen.tolist(),
                     switched=int(len(possible)>0), held_rr=float(caption), DE=float(detail),
                     both_up=int(caption>1e-12 and detail>1e-12),
                     one_up_other_level=int((caption>1e-12 and abs(detail)<1e-12) or
                                            (detail>1e-12 and abs(caption)<1e-12)),
                     either_down=int(caption < -1e-12 or detail < -1e-12))
        if fresh is not None:
            entry["FG"] = float(fresh[chosen].mean() - fresh[top].mean())
        trials.append(entry)
    return trials


def summarize(images):
    if not images:
        return dict(images=0)
    pairs = sorted({(x["cohort"],x["pair"]) for x in images})
    rng = np.random.default_rng(27112026)
    draws = rng.multinomial(len(pairs), np.full(len(pairs), 1/len(pairs)), size=20000)
    keys = ("switched","held_rr","DE","FG","both_up","one_up_other_level","either_down")
    results = {}
    for key in keys:
        totals = np.zeros((len(pairs),2))
        for row in images:
            values = [t[key] for t in row["trials"] if key in t]
            if values:
                i = pairs.index((row["cohort"],row["pair"]))
                totals[i] += [np.mean(values),1]
        if totals[:,1].sum():
            boots = draws @ totals
            boots = boots[boots[:,1]>0]
            lo,hi = np.quantile(boots[:,0]/boots[:,1],[.025,.975])
            results[key] = dict(images=int(totals[:,1].sum()), mean=float(totals[:,0].sum()/totals[:,1].sum()),
                                interval=[float(lo),float(hi)])
    return dict(images=len(images),pairs=len(pairs),held_trials=sum(len(x["trials"]) for x in images),
                switches=sum(t["switched"] for x in images for t in x["trials"]),metrics=results)


def main():
    rows = read(ROSTER)
    locks = STAGE / "scores"
    images = []
    for row in rows:
        files = [locks / seed / (row["image"] + ".json") for seed in "ab"]
        if not all(f.is_file() for f in files):
            continue
        a,b = map(read,files)
        if not (a.get("items") and b.get("items")):
            continue
        if not a["items"][0].get("other_caption_rr") or not b["items"][0].get("other_caption_rr"):
            continue
        trials = held_trials(b)
        if trials:
            images.append(dict(image=row["image"],pair=row["pair"],cohort=row["cohort"],
                               selected=bool(a["achieved"]),trials=trials))
    assert images, "no scored eligible images"
    with gzip.open(OUT/"逐图留句.jsonl.gz","wt",encoding="utf-8") as f:
        for item in images:
            f.write(json.dumps(item,ensure_ascii=False)+"\n")
    result = {tag:summarize(group) for tag,group in
              [("all",images),("selected",[x for x in images if x["selected"]]),
               ("unselected",[x for x in images if not x["selected"]])]}
    (OUT/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()

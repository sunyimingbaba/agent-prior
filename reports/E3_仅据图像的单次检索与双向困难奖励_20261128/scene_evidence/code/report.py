"""Fixed-policy comparisons using the verified common evaluation galleries and original native scores."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent / "image_only"
sys.path.insert(0, str(BASELINE/"code"))
from score import (read, sha, dump, FORM, FINE, OLD, calculate, details, metrics_extra,
                   extra_stats, summary_by_seed)


def data(root, tag, condition, name, rows, lock, donor, checksum, feedback):
    if condition == "none":
        folder = BASELINE/"records"/tag/name
        f = folder/"features.pt"
        meta = read(folder/"complete.json")
        assert sha(f) == meta["features"]
        z = torch.load(f, map_location="cpu", weights_only=True)
        assert z["adapter"] == checksum and z["images"] == [r["image"] for r in rows]
        return z["features"], {}
    f = root/"records"/(tag+"_"+condition)/(name+".pt")
    meta = read(f.with_suffix(".json"))
    assert meta["sha256"] == sha(f) and meta["model"] == checksum
    assert meta["manifest"] == sha(root/"inputs/manifest.json")
    assert meta["generation"] == sha(root/"code/generate.py") and meta["common"] == sha(BASELINE/"code/predict.py")
    z = torch.load(f, map_location="cpu", weights_only=True)
    assert z["adapter"] == checksum
    assert z["visual"].shape == (len(rows), 768) and z["prior"].shape == (len(rows), 3584)
    assert torch.isfinite(z["prior"]).all() and torch.isfinite(z["visual"]).all()
    calls = Counter()
    stops = Counter()
    for r, trace in zip(rows, z["traces"], strict=True):
        expected = r["rgb"] if condition == "real" else donor[r["rgb"]]
        q = lock["query"] + "\n" + lock["label"] + feedback[expected]["feedback"]
        assert (trace["rgb"], trace["evidence_rgb"], trace["query"]) == (r["rgb"], expected, q)
        assert all(answer == r["feedback"] for answer in trace["calls"])
        assert trace["actions"] and trace["tokens"] and trace["template"]
        calls[len(trace["calls"])] += 1
        stops[str(trace["actions"] == [[151645]])] += 1
    return z["visual"], dict(filesha=sha(f), calls=dict(calls), direct_end=dict(stops))


def main():
    lock = read(ROOT/"inputs/manifest.json")
    donor = read(ROOT/"inputs/assignments.json")
    feedback = read(BASELINE/"feedback_verified.json")["checks"]
    manifest = read(BASELINE/"inputs/manifest.json")
    assert all(sha(BASELINE.parents[1]/f) == v for f,v in manifest["score_inputs"].items())
    roster = read(BASELINE/"inputs/roster.json")
    rows = {panel:[r for r in roster if r["panel"]==panel] for panel in ("extra","official")}
    official = read(FORM/"inputs/ucm.json")
    ids = {r["rgb_sha256"]: i for i, r in enumerate(official)}
    fine = [r for r in read(FINE/"inputs/panel.json") if r["rgb_sha256"] in ids]
    extra = read(OLD/"extra164/inputs/rows.json")
    assert [r["rgb"] for r in rows["official"]] == [r["rgb_sha256"] for r in official]
    assert [r["rgb"] for r in rows["extra"]] == [r["rgb_sha256"] for r in extra]
    h = torch.load(FINE/"records/text.pt",weights_only=True)
    text = torch.load(FORM/"results/ucm_text.pt",weights_only=True)
    xt = torch.load(OLD/"extra164/inputs/texts.pt",weights_only=True)
    cols = text["caption_order"].numpy().reshape(210,5)
    labels = {"official":[r["image"] for r in official],"first106":[r["pair"] for r in fine],
              "extra164":[r["pair"] for r in extra]}
    groups, receipts = {}, {}
    variants = (("strict","real"),("strict","shuffled"),("strict","none"),("r2","real"),("r2","none"))
    base_scores = read(BASELINE/"results/单次检索.json")
    by_image = []
    for arm, condition in variants:
        key = arm+"_"+condition; groups[key] = {p:{} for p in labels}
        for seed in "ab":
            tag = seed+"_"+arm
            vectors = {}
            for panel in ("official","extra"):
                vectors[panel], proof = data(ROOT,tag,condition,panel,rows[panel],lock,donor,
                                             lock["policies"][tag]["adapter"],feedback)
                if proof:
                    receipts[tag+"_"+condition+"_"+panel] = proof
            values = dict(official=calculate("official", vectors["official"], official),
                          first106=details(vectors["official"],fine,ids,cols,text,h),
                          extra164=metrics_extra(extra,xt,vectors["extra"]))
            for p, measures in values.items():
                groups[key][p][seed] = measures
                for metric,v in measures.items():
                    if condition == "none":
                        assert abs(v.mean()-base_scores[p][arm][metric]["seeds"]["ab".index(seed)]) < 1e-12
                for i,r in enumerate({"official":official,"first106":fine,"extra164":extra}[p]):
                    by_image.append(dict(policy=tag, panel=p, condition=condition, image=r["image"],
                                         measures={m:float(v[i]) for m,v in measures.items()}))
    def summary(per):
        return {p:(extra_stats if p=="extra164" else summary_by_seed)(per[p],labels[p]) for p in labels}
    levels = {key:summary(g) for key,g in groups.items()}
    differences = {}
    for end, start in (("strict_real","strict_none"),("strict_real","strict_shuffled"),("strict_shuffled","strict_none"),
                       ("strict_real","r2_real"),("r2_real","r2_none")):
        g = {p:{s:{m:arr-groups[start][p][s][m] for m,arr in groups[end][p][s].items()}
                for s in "ab"} for p in labels}
        differences[end+"-"+start] = summary(g)
    evidence = [differences["strict_real-"+tag] for tag in ("strict_none","strict_shuffled")]
    reward = differences["strict_real-r2_real"]
    evidence_ok = all(d["official"]["mR"]["interval"][0]>0 and d["extra164"]["mR"]["mean"]>0 for d in evidence)
    reward_ok = (reward["first106"]["DE"]["interval"][0]>0 and reward["official"]["mR"]["mean"]>=0
                 and reward["first106"]["mrr"]["mean"]>=0
                 and all(reward["extra164"][k]["mean"]>=0 for k in ("mR","mrr","DE")))
    dump(ROOT/"结果.json", dict(levels=levels,differences=differences,evidence_gain=evidence_ok,reward_pass=reward_ok))
    dump(ROOT/"轨迹清单.json",dict(new_outputs=2244,files=receipts))
    with gzip.open(ROOT/"逐图结果.jsonl.gz","wt",encoding="utf-8") as stream:
        for record in by_image:
            stream.write(json.dumps(record,allow_nan=False)+"\n")
    print("evidence",evidence_ok,"reward",reward_ok)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

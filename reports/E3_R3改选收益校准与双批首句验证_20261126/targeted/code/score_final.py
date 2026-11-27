"""Audit single-pass outputs, compute matched retrieval changes, and preserve raw observations."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import sys
import tarfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "code"))
from pilot_results import RFPO, FORM, read, rr, sha, calculate, summary_by_seed, fixed, paired
from transfer_scores import counts, summary, scalar_values

ARMS = ("ordinary", "old", "old3", "retained")


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def archive(name, folder, hashes):
    path = ROOT/"archives"/(name+".tar.gz")
    with tarfile.open(path, "w:gz", compresslevel=3) as tar:
        for rel, checksum in sorted(hashes.items()):
            assert sha(folder/rel) == checksum
            tar.add(folder/rel, arcname=rel)
    with tarfile.open(path, "r|gz") as tar:
        seen = set()
        for member in tar:
            assert member.name not in seen
            seen.add(member.name)
            assert hashlib.file_digest(tar.extractfile(member), "sha256").hexdigest() == hashes[member.name]
        assert seen == set(hashes)
    return {"sha256":sha(path), "files":len(hashes), "bytes":path.stat().st_size}


def get_and_archive(seed, arm, gallery, common_inputs):
    folder = ROOT/"evaluate"/(seed+"_"+arm)
    proof = read(folder/"complete.json")
    assert sha(ROOT/"单次推理验收.md") == proof["protocol"] and sha(ROOT/"code/eval_single.py") == proof["decoder"]
    final = (read(ROOT.parent/"archives/index.json")[seed]["adapter"] if arm == "retained"
             else read(ROOT/"models"/(seed+"_"+arm)/"complete.json")["final"])
    bundle = torch.load(folder/"features.pt", weights_only=True, map_location="cpu")
    assert bundle["adapter"] == proof["adapter"] == final
    assert len(proof["images"]) == len(bundle["features"]) == len(gallery) == 210
    assert bundle["images"] == [row["image"] for row in gallery]
    hashes = {"features.pt":proof["features"], "complete.json":sha(folder/"complete.json")}
    for i, row in enumerate(gallery):
        name = row["image"]
        path = folder/(name+".pt")
        hashes[path.name] = proof["images"][name]
        z = torch.load(path, weights_only=True, map_location="cpu")
        assert z["adapter"] == final and z["protocol"] == proof["protocol"] and z["decoder"] == proof["decoder"]
        assert (z["image"], z["query"], z["rgb"]) == (name, row["orig_query"], row["rgb_sha256"])
        assert torch.equal(z["v"], bundle["features"][i]) and torch.isfinite(z["prior"]).all()
        assert common_inputs.setdefault(name, z["feedback"]) == z["feedback"]
    artifact = archive(seed+"_"+arm, folder, hashes)
    return bundle["features"], {**artifact,"adapter":final,"torch":proof["torch"],"device":proof["device"]}


def details(visual, rows, gallery_ids, cols, native, hard):
    detail = fixed(visual, hard["hard"])
    scores = fixed(visual, native["native"])
    result = defaultdict(list)
    for row in rows:
        i = gallery_ids[row["rgb_sha256"]]
        ix = hard["maps"]["primary"][row["image"]]
        for letter in "DE":
            good, bad = ix[letter]
            result[letter].append(float(detail[i,good] > detail[i,bad]))
        pos = cols[i]
        alts = sorted(set(pos)-{pos[0]})
        assert alts
        negative = np.ones(scores.shape[1], dtype=bool)
        negative[pos] = False
        result["mrr"].append(np.mean([float(rr(scores[i,j],scores[i,negative])) for j in alts]))
    values = {k:np.array(v) for k,v in result.items()}
    values["DE"] = (values["D"] + values["E"])/2
    return values


def alt_summary(data, ds):
    x = np.stack([data[s] for s in "ab"])
    estimates = summary(x.mean(0), ds)
    keys = list(estimates)
    assert keys[-1] == "mR"
    seeds = np.stack([scalar_values(a.sum(0), np.asarray(len(ds)), np.asarray(ds.sum())) for a in x])
    for i,k in enumerate(keys):
        estimates[k]["seeds"] = seeds[:,i].tolist()
    return estimates


def criteria(out):
    mr = out["official"]["contrasts"]["old3-old"]["mR"]
    de = out["first106"]["contrasts"]["old3-old"]["DE"]
    mrr = out["first106"]["contrasts"]["old3-old"]["mrr"]
    relative = out["first106"]["contrasts"]["old3-retained"]["mrr"]
    improved = (mr["mean"] > 0 and de["mean"] > 0 and max(mr["interval"][0],de["interval"][0]) > 0
                and mrr["mean"] >= 0)
    return {"passes_relative_to_old":improved, "significant_other_caption_loss_vs_R2":relative["interval"][1] < 0}


def main():
    labels = read(RFPO/"inputs/panel.json")
    gallery = read(FORM/"inputs/ucm.json")
    gallery_ids = {r["rgb_sha256"]:i for i,r in enumerate(gallery)}
    overlap = [r for r in labels if r["rgb_sha256"] in gallery_ids]
    assert len(overlap) == 106
    native = torch.load(FORM/"results/ucm_text.pt", weights_only=True)
    hard = torch.load(RFPO/"records/text.pt", weights_only=True)
    cols = native["caption_order"].numpy().reshape(210,5)
    official, first, alternatives = (defaultdict(dict) for _ in range(3))
    archives, raw, inputs = {}, [], {}
    den = None
    for seed in "ab":
        for arm in ARMS:
            v, archives[seed+"_"+arm] = get_and_archive(seed, arm, gallery, inputs)
            official[arm][seed] = calculate("official",v,gallery)
            first[arm][seed] = details(v,overlap,gallery_ids,cols,native,hard)
            ids, hits, ds = counts(v,native,"alternatives")
            if den is None:
                owners, den = ids, ds
            assert np.array_equal(owners,ids) and np.array_equal(den,ds)
            alternatives[arm][seed] = hits
            image_metrics = {r["image"]:{"image":r["image"], "rgb":r["rgb_sha256"], "seed":seed, "arm":arm,
                             "official":{k:float(x[i]) for k,x in official[arm][seed].items()}}
                             for i,r in enumerate(gallery)}
            for i,r in enumerate(overlap):
                image_metrics[r["image"]]["first106"] = {k:float(x[i]) for k,x in first[arm][seed].items()}
            for i,n in enumerate(ids):
                image_metrics[gallery[n]["image"]]["alternatives"] = {"hits":hits[i].tolist(),"captions":int(ds[i])}
            raw.extend(image_metrics.values())
    assert len({(v["torch"],v["device"]) for v in archives.values()}) == 1
    out = {}
    comparisons = (("old3","old"), ("old3","ordinary"), ("old3","retained"), ("old","retained"), ("ordinary","retained"))
    for tag, values, units in (("official",official,[r["image"] for r in gallery]), ("first106",first,[r["pair"] for r in overlap])):
        out[tag] = {"images":len(units),"independent_units":len(set(units)),
                    "levels":{a:summary_by_seed(values[a],units) for a in ARMS},
                    "contrasts":{a+"-"+b:paired(values[a],values[b],units) for a,b in comparisons}}
    out["alternatives"] = {"images":len(den), "captions":int(den.sum()), "levels":{a:alt_summary(alternatives[a],den) for a in ARMS},
            "contrasts":{a+"-"+b:alt_summary({s:alternatives[a][s]-alternatives[b][s] for s in "ab"},den) for a,b in comparisons}}
    out["criteria"] = criteria(out)
    dump(ROOT/"archives/evaluate.json",archives)
    dump(ROOT/"results/逐图检索与细节.json",raw)
    dump(ROOT/"results/单次检索对照.json",out)
    files = ["archives/evaluate.json","results/逐图检索与细节.json","results/单次检索对照.json","单次推理验收.md","code/score_final.py"]
    dump(ROOT/"results/eval.complete.json",{"files":{f:sha(ROOT/f) for f in files}, "models":len(archives), "observations":len(raw)})


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

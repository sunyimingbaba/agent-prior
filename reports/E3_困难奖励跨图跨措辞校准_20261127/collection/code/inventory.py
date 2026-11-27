"""Inventory archived cold-start tool contexts, mirrored captions and controlled states."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT.parents[1]
CURRENT = REPORTS / "E3_双收益回答对的偏好训练_20261127/validation98"
OLD = REPORTS / "E3_回填后双收益训练图筛选_20261127"
FRESH = CURRENT / "fresh98"
CONTROL = CURRENT / "受控真假回答"
FG = REPORTS / "E3_R3改选收益校准与双批首句验证_20261126/inputs/独立措辞.json"


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    with p.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def get_data():
    return {"train98": (FRESH/"inputs/rows.json", FRESH/"c_complete.json", FRESH/"raw/c"),
            "pool270": (OLD/"scoring/inputs/rows.json", OLD/"sampling/a_complete.json", OLD/"sampling/raw/a")}


def main():
    rows_by_panel, info, records = {}, {}, {}
    fresh = read(FG)
    source = {}
    original = read(CONTROL/"generation.json")
    controls = read(CONTROL/"scoring.json")
    for panel, (roster_path, proof_path, raw_dir) in get_data().items():
        rows = read(roster_path)
        proof = read(proof_path)
        source[panel] = {str(x.relative_to(REPORTS)):sha(x) for x in (roster_path,proof_path)}
        rows_by_panel[panel] = rows
        pairs = defaultdict(list)
        entries = {}
        for r in rows:
            name = r["image"]
            p = raw_dir/(name+".pt")
            assert sha(p)==proof["files"][name]
            state = torch.load(p,weights_only=True,map_location="cpu")
            assert state["eligible"] == proof["eligibility"][name]
            assert state["adapter"]==proof["adapter"] and state["query"]==r["orig_query"] and state["rgb"]==r["rgb_sha256"]
            assert (state["context"] is not None)==state["eligible"]
            texts = {k:r[k] for k in "ABDE"}
            for key, variants in zip("FG", fresh[r["attribute"]], strict=True):
                texts[key] = [variants[r["side"]],variants[1-r["side"]]]
            assert all(z[0]!=z[1] for z in texts.values())
            assert all(not set(texts[k]) & set(texts["A"]+texts["B"]) for k in "DEFG")
            assert r["image"] not in entries
            entry = dict(image=name,row=r["row"],rgb=r["rgb_sha256"],source=sha(p), pair=r["pair"],
                         side=r["side"], attribute=r["attribute"], eligible=state["eligible"],texts=texts)
            if state["eligible"]:
                assert state["hidden"].shape==(8,3584) and state["context"]["answer"]==r["feedback"]
            if panel=="train98":
                controlled=CONTROL/"raw"/(name+".pt")
                visual=CONTROL/"readouts"/(name+".pt")
                result=CONTROL/"scores"/(name+".json")
                assert sha(controlled)==original["files"][name] and sha(visual)==controls["readouts"][name]
                assert sha(result)==controls["files"][name]
                cc=torch.load(controlled,weights_only=True,map_location="cpu")
                assert cc["states"].shape==(2,3584) and cc["eos"]==151645
                assert torch.equal(cc["context_prefix"],state["context"]["prefix"])
                vec=torch.load(visual,weights_only=True,map_location="cpu")
                assert vec["visual"].shape==(2,768)
                assert read(result)["texts"] == ["<answer>"+q+"</answer>" for q in r["A"]]
                entry.update(controlled_sha=sha(controlled),visual_sha=sha(visual),scores_sha=sha(result))
            entries[name]=entry
            pairs[r["pair"]].append(entry)
        records[panel] = entries
        for name, members in pairs.items():
            members.sort(key=lambda r:r["side"])
            assert len(members)==2 and [r["side"] for r in members]==[0,1]
            assert members[0]["attribute"]==members[1]["attribute"]
            for k in "ABDEFG":
                assert members[0]["texts"][k] == members[1]["texts"][k][::-1]
        complete=[q for q,v in pairs.items() if all(r["eligible"] for r in v)]
        info[panel]=dict(images=len(rows), attrs=dict(Counter(r["attribute"] for r in rows)),
            natural_pairs=len(pairs), complete_pairs=complete,adapter=proof["adapter"],
            usable_by_attribute=dict(Counter(pairs[q][0]["attribute"] for q in complete)))
    a,b = rows_by_panel.values()
    assert info["train98"]["adapter"]==info["pool270"]["adapter"]
    train_tags=set(info["train98"]["attrs"])
    pool=[x for x in records["pool270"].values() if x["attribute"] in train_tags and x["pair"] in info["pool270"]["complete_pairs"]]
    gallery = REPORTS/"E3-2_UCM与RSITMD独立复验_20261124/inputs"
    manifests={"train98":{x["rgb_sha256"] for x in a},"pool270":{x["rgb_sha256"] for x in b}}
    overlap={}
    for key in ("ucm","rsitmd"):
        p=gallery/(key+".json")
        source[key]=sha(p)
        ids={r["rgb_sha256"] for r in read(p)}
        overlap[key]={k:len(ids&v) for k,v in manifests.items()}
    overlap["98_vs_270"]=len(manifests["train98"] & manifests["pool270"])
    assert all(v==0 for k in ("ucm","rsitmd") for v in overlap[k].values()) and overlap["98_vs_270"]==0
    out = ROOT/"inputs"
    out.mkdir(exist_ok=True)
    save(out/"catalog.json",records)
    save(out/"inventory.json",dict(source=source, panels=info, common_complete_images=len(pool),
        common_complete_pairs=len({q["pair"] for q in pool}), common_attributes=sorted(train_tags&set(info["pool270"]["attrs"])),
        overlap=overlap, new_wording_sha=sha(FG), catalog=sha(out/"catalog.json"), code=sha(Path(__file__)),
        old_controlled=sha(CONTROL/"generation.json"), old_scores=sha(CONTROL/"scoring.json")))


if __name__ == "__main__":
    main()

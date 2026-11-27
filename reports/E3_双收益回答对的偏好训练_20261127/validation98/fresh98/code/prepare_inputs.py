"""Map every positive caption into the original native bank and lock source assets."""
import hashlib
import json
from pathlib import Path

import torch

SCORING = Path(__file__).resolve().parents[1]
REPORTS = SCORING.parents[2]
OLD = REPORTS / "E3_R3改选收益校准与双批首句验证_20261126/inputs"
RELATION = REPORTS / "E3-1_奖励一致性分析_20261112/空间关系候选奖励核验_20261120"
LIBRARIES = {
    "history98": REPORTS/"E3_UCM同图偏好对与独立验收_20261124/reward_confirm/records/text.pt",
}
ORIGINAL = REPORTS/"E3-2_UCM回填后回答训练_20261124/code"
BONUS = REPORTS/"E3_R2逐候选分值与困难细节加分_20261127/code/detail_reward.py"


def sha(p):
    with p.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def load(p):
    return torch.load(p, map_location="cpu",weights_only=True)


def main():
    roster = REPORTS/"E3_R3首位配文保护与首句复验_20261126/inputs/real_roster.json"
    assert sha(roster) == read(roster.parent/"real_lock.json")["roster"]
    rows = read(roster)
    banks = {k:load(p) for k,p in LIBRARIES.items()}
    canonical = RELATION/"native/features.pt"
    index = read(RELATION/"native/index.json")
    assert sha(canonical) == index["sha256"]
    tokens = load(canonical)["tokens"].tolist()
    ids = {tuple(q):i for i,q in enumerate(tokens)}
    remaps = {k:[ids[tuple(q)] for q in d["native_tokens"].tolist()] for k,d in banks.items()}
    assert all(len(set(v)) == len(v) == len(ids) == len(tokens) for v in remaps.values())
    fresh = load(OLD/"fresh_text.pt")
    assert fresh["source"] == sha(OLD/"独立措辞.json")
    words = read(OLD/"独立措辞.json")
    fresh_ids = {w:i for i,w in enumerate(fresh["words"])}
    vectors = {}
    for row in rows:
        row["cohort"] = "history98"
        cohort, name = row["cohort"], row["image"]
        bank = banks[cohort]
        marks = bank["maps"][name]
        pos = [remaps[cohort][i] for i in marks["native"]]
        assert pos == [index["columns"][q] for q in row["captions"]]
        row["native_ids"] = pos
        row["alt_ids"] = sorted(set(pos)-{pos[0]})
        vec = []
        for label in "ABDE":
            assert [bank["texts"][i] for i in marks[label]] == row[label]
            vec.extend(bank["hard"][marks[label]])
        if row["attribute"] in words:
            side = row["side"]
            for pair in words[row["attribute"]]:
                vec.extend([fresh["features"][fresh_ids[pair[k]]] for k in [side,1-side]])
        vectors[name] = torch.stack(vec)
        assert len(vec) in (8,12)
    assert len(rows) == len(vectors) == len({r["rgb_sha256"] for r in rows}) == 98
    out=SCORING/"inputs"
    save(out/"rows.json",rows)
    torch.save(vectors,out/"vectors.pt")
    gallery = RELATION/"gallery/train.pt"
    assert sha(gallery) == read(RELATION/"gallery/index.json")["gallery_sha256"]
    manifest = {
        "compiled": {f:sha(out/f) for f in ["rows.json","vectors.pt"]},
        "rows": len(rows), "alt_images":sum(bool(r["alt_ids"]) for r in rows),
        "fg_images": sum(len(v)==12 for v in vectors.values()),
        "native":sha(canonical),"gallery":sha(gallery),
        "sources":{str(x.relative_to(REPORTS)):sha(x) for x in [
            roster, OLD/"fresh_text.pt",OLD/"独立措辞.json", *LIBRARIES.values()]},
        "training_source":sha(ORIGINAL/"train.py"),"bonus_source":sha(BONUS),
        "readouts_source":sha(ORIGINAL/"readouts.py"),"shared_source":sha(ORIGINAL/"shared.py")
    }
    save(out/"manifest.json",manifest)


if __name__ == "__main__":
    main()

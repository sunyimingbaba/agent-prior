from pathlib import Path
import hashlib
import json
import torch

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parents[1]
REPORTS = STAGE.parent
SOURCE = REPORTS/"E3_双收益回答对的偏好训练_20261127/validation98/fresh98/inputs"
NATIVE = next(REPORTS.glob("E3-1_奖励一致性*/空间关系候选*/native"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def write(p, data):
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n")


def main():
    lock = read(SOURCE/"manifest.json")
    for name, digest in lock["compiled"].items():
        assert sha(SOURCE/name) == digest
    assert sha(NATIVE/"features.pt") == lock["native"]
    assert read(NATIVE/"index.json")["sha256"] == lock["native"]
    data = read(SOURCE/"rows.json")
    details = torch.load(SOURCE/"vectors.pt", weights_only=True)
    bank = torch.load(NATIVE/"features.pt", weights_only=True)
    ids = read(NATIVE/"index.json")["columns"]
    pixels = {r["rgb_sha256"] for r in data}
    others = [STAGE/"training_signal/third/inputs/roster.json", STAGE/"extra164/inputs/rows.json",
              REPORTS/"E3-2_UCM与RSITMD独立复验_20261124/inputs/ucm.json"]
    assert all(pixels.isdisjoint(r["rgb_sha256"] for r in read(p)) for p in others)
    assert len(pixels) == 98 and len({r["pair"] for r in data}) == 49
    caption_ids = sorted({ids[cap] for r in data for cap in r["captions"]})
    compact = {j: i for i, j in enumerate(caption_ids)}
    fine = []
    for r in data:
        native_ids = [ids[cap] for cap in r["captions"]]
        assert r["native_ids"] == native_ids
        r["panel_positive_ids"] = [compact[j] for j in native_ids]
        assert len(set(native_ids)) > 1
        v = details[r["image"]]
        assert v.shape == (12, 768)
        # The archived order is A true/false, B, D, E, F, G.
        first = len(fine)
        fine.extend(v[4:12])
        r["panel_de_ids"] = list(range(first, first+4))
        r["panel_fg_ids"] = list(range(first+4, first+8))
    directory = ROOT/"inputs"
    write(directory/"rows.json", data)
    torch.save(dict(captions=bank["features"][caption_ids], details=torch.stack(fine)), directory/"texts.pt")
    write(directory/"manifest.json", dict(rows=sha(directory/"rows.json"), texts=sha(directory/"texts.pt"),
          protocol=sha(ROOT/"事前口径.md"), sources={str(f.relative_to(REPORTS)):sha(f) for f in
              [SOURCE/"manifest.json", SOURCE/"rows.json", SOURCE/"vectors.pt", NATIVE/"features.pt", NATIVE/"index.json", *others]},
          policy_images=24, gallery=len(caption_ids), images=98, pairs=49))


if __name__ == "__main__":
    main()

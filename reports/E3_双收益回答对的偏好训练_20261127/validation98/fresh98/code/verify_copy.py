"""Verify the local copy of a completed sampling pass against its per-image receipts."""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def main(seed):
    assert seed == "c"
    roster = ROOT / "inputs/roster.json"
    rows = read(roster)
    feedback_path = ROOT / "feedback_verified.json"
    feedback = read(feedback_path)
    manifest = read(ROOT / f"{seed}_complete.json")
    eligibility = read(ROOT / f"{seed}_eligibility.json")
    dependencies = {Path(name).name: sig for name, sig in manifest["lock"]["dependencies"].items()}
    assert dependencies["generate.py"] == sha(ROOT / "code/generate.py")
    assert dependencies["feedback_verified.json"] == sha(feedback_path)
    assert sha(roster) == feedback["roster"] == manifest["lock"]["roster"] == read(ROOT / "inputs/roster_lock.json")["roster"]
    expected = {row["image"] for row in rows}
    assert expected == set(manifest["files"]) == set(eligibility) == set(feedback["images"])
    assert eligibility == manifest["eligibility"] and manifest["reached"] == sum(eligibility.values())
    directory = ROOT / "raw" / seed
    assert {path.name for path in directory.glob("*.pt")} == {name + ".pt" for name in expected}
    torch.set_num_threads(1)
    for row in rows:
        name = row["image"]
        path = directory / (name + ".pt")
        assert sha(path) == manifest["files"][name]
        record = torch.load(path, weights_only=True, map_location="cpu")
        assert record["rgb"] == feedback["images"][name]["rgb"] == row["rgb_sha256"]
        assert record["query"] == row["captions"][0] == row["orig_query"]
        captions = json.dumps(row["captions"], ensure_ascii=False).encode("utf-8")
        assert hashlib.sha256(captions).hexdigest() == record["captions_sha256"]
        assert record["feedback"] == feedback["images"][name]["feedback"] == row["feedback"]
        assert record["adapter"] == manifest["adapter"] and record["lock"] == manifest["lock"]
        assert record["rng"] == 251000000 + row["row"] * 1000
        assert record["seed"] == seed and record["image"] == name and record["row"] == row["row"]
        assert record["eligible"] == eligibility[name] and 0 < len(record["first"]) <= 64
        if eligibility[name]:
            ctx = record["context"]
            assert ctx["first"] == record["first"] and ctx["answer"] == row["feedback"]
            assert ctx["prefix"].dtype == torch.int64 and ctx["prefix"].ndim == 2 and ctx["prefix"].shape[0] == 1
            assert record["hidden"].shape == (8, 3584) and torch.isfinite(record["hidden"]).all()
            assert len(record["paths"]) == len(record["texts"]) == len(record["logps"]) == len(record["extras"]) == 8
            assert all(0 < len(ids) == len(probs) <= 64 for ids, probs in zip(record["paths"], record["logps"]))
            assert all(len(ids) <= 64 for ids in record["extras"])
        else:
            assert record["context"] is None and record["hidden"] is None and record["paths"] == []
    summary = dict(seed=seed, images=len(rows), reached=manifest["reached"], groups=manifest["reached"],
                   terminal_states=8 * manifest["reached"], skipped=len(rows) - manifest["reached"],
                   manifest=sha(ROOT / f"{seed}_complete.json"), verifier=sha(Path(__file__)))
    path = ROOT / f"{seed}_copy_verified.json"
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(path)
    print(json.dumps(summary))


if __name__ == "__main__":
    main(sys.argv[1])

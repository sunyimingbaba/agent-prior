"""Check actual RFPO updates and archive the complete single-pass evaluation."""
from pathlib import Path
import hashlib
import json
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def audit_training(seed, arm, roster, receipt):
    folder = ROOT / "models" / f"{seed}_{arm}"
    proof = load(folder / "complete.json")
    steps = load(folder / "steps.json")
    assert len(steps) == proof["steps"] == len(roster) == 24
    assert proof["scores"] == sha(folder / "steps.json")
    assert proof["lock"] == sha(ROOT / "inputs/pilot/decision.json")
    assert proof["initial"] == receipt["starts"][seed]
    assert proof["nonzero"] == sum(row["norm"] > 0 for row in steps)
    pairs = [("old", "ordinary"), ("old3", "ordinary"), ("old3", "old")]
    differences = {f"{a}-{b}": [] for a, b in pairs}
    for step, (row, ref) in enumerate(zip(steps, roster), 1):
        assert (row["step"], row["image"], row["query"]) == (step, ref["image"], ref["orig_query"])
        assert len(row["paths"]) == 8
        scores = np.zeros(8)
        scores[row["winners"][arm]] = 1
        expected = (scores - scores.mean()) / (scores.std(ddof=1) + 1e-8)
        assert np.allclose(row["advantages"], expected, atol=1e-7, rtol=0)
        assert np.isfinite([row["norm"], row["max_replay_error"], row["mean_replay_error"]]).all()
        for a, b in pairs:
            if set(row["winners"][a]) != set(row["winners"][b]):
                differences[f"{a}-{b}"].append(step)
    return {
        "first_norm": steps[0]["norm"], "first_winners": steps[0]["winners"][arm],
        "steps": len(steps), "nonzero": proof["nonzero"], "final": proof["final"],
        "changes_on_same_candidates": differences,
        "mean_replay_error": float(np.mean([row["mean_replay_error"] for row in steps])),
        "max_replay_error": max(row["max_replay_error"] for row in steps)
    }


def archive_outputs(name, expected_adapter):
    folder = ROOT / "evaluate" / name
    path = ROOT / "archives" / f"single_{name}.tar.gz"
    expected = {}
    for tag, size in (("official", 210), ("primary", 126)):
        manifest = folder / tag / "complete.json"
        proof = load(manifest)
        assert len(proof["images"]) == size
        expected[f"{tag}.pt"] = proof["features"]
        expected[f"{tag}/complete.json"] = sha(manifest)
        for image, digest in proof["images"].items():
            expected[f"{tag}/{image}.pt"] = digest
    with tarfile.open(path, "w:gz", compresslevel=3) as archive:
        for relative in sorted(expected):
            assert sha(folder / relative) == expected[relative]
            archive.add(folder / relative, arcname=relative)
    with tarfile.open(path, "r|gz") as archive:
        checked = set()
        for member in archive:
            assert member.name not in checked
            assert hashlib.file_digest(archive.extractfile(member), "sha256").hexdigest() == expected[member.name]
            checked.add(member.name)
    assert checked == set(expected)
    return {"adapter": expected_adapter, "files": len(expected), "archive_sha256": sha(path), "bytes": path.stat().st_size}


def main():
    receipt = load(ROOT / "inputs/pilot/decision.json")
    roster = load(ROOT / "inputs/pilot/train.json")
    training, archives = {}, {}
    for seed in ("a", "b"):
        for arm in receipt["train_arms"]:
            name = f"{seed}_{arm}"
            training[name] = audit_training(seed, arm, roster[seed], receipt)
            archives[name] = archive_outputs(name, training[name]["final"])
    save(ROOT / "results/训练信号.json", training)
    save(ROOT / "archives/单次推理索引.json", archives)


if __name__ == "__main__":
    main()

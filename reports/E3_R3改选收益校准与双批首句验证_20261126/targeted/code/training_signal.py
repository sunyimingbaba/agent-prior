"""Check how often the reward changes and when paired deterministic runs first separate."""
from itertools import combinations
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(seed):
    arms = ("ordinary", "old", "old3")
    rows = {}
    records = {}
    for arm in arms:
        base = ROOT / "models" / f"{seed}_{arm}"
        steps = read(base/"steps.json")
        proof = read(base/"complete.json")
        assert proof["scores"] == digest(base/"steps.json")
        assert len(steps) == proof["steps"] == 13
        changes = {"old3-old": [], "old3-ordinary": []}
        for i, step in enumerate(steps):
            assert step["step"] == i+1 and len(step["paths"]) == 8
            assert step["rng"] == (165000000 if seed == "a" else 175000000) + i*1000
            win = step["winners"]
            scores = np.zeros(8)
            scores[win[arm]] = 1
            expected = (scores-scores.mean()) / (scores.std(ddof=1)+1e-8)
            np.testing.assert_allclose(expected, step["advantages"], atol=1e-7, rtol=0)
            for other in ("old","ordinary"):
                if set(win["old3"]) != set(win[other]):
                    changes["old3-"+other].append(i+1)
        rows[arm] = steps
        records[arm] = {"image_order":[s["image"] for s in steps], "nonzero":proof["nonzero"],
                        "changes":changes, "max_replay_error":max(s["max_replay_error"] for s in steps)}
    for arm in arms:
        assert records[arm]["image_order"] == records["ordinary"]["image_order"]
    divergence = {}
    for a,b in combinations(arms, 2):
        first_feedback = None
        for i,(left,right) in enumerate(zip(rows[a],rows[b]),1):
            # All three policies have identical inputs and weights until the first different actual update.
            assert all(left[k] == right[k] for k in ("states","paths","winners","query","rng"))
            if left["advantages"] != right["advantages"]:
                first_feedback = i
                break
            assert left["norm"] == right["norm"]
        divergence[f"{b}-{a}"] = first_feedback
    return {"arms":records, "first_different_feedback":divergence}


if __name__ == "__main__":
    out = {s:audit(s) for s in ("a","b")}
    (ROOT/"results/训练分歧核查.json").write_text(json.dumps(out,indent=2)+"\n")

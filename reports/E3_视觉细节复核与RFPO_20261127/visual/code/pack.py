"""Verify old inputs and combine the per-image image-only and per-answer C scores."""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT.parents[1]
EXTERNAL = REPORTS / "E3_困难奖励跨图跨措辞校准_20261127/followup_pool"
MAIN = REPORTS / "E3_回填后双收益训练图筛选_20261127"
RESULTS = ROOT / "results"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def load(path):
    return torch.load(path, weights_only=True, map_location="cpu")


def external(rows, vector_sha):
    torch.set_num_threads(4)
    c_vectors = load(ROOT / "inputs/c_vectors.pt")
    hard_path = EXTERNAL / "inputs/vectors.pt"
    proof = read(EXTERNAL / "scoring_complete.json")
    assert sha(EXTERNAL / "inputs/manifest.json") == proof["compiled"]
    assert sha(hard_path) == read(EXTERNAL / "inputs/manifest.json")["compiled"]["vectors.pt"]
    assert sha(EXTERNAL / "d_complete.json") == proof["sampling"]
    raw_proof = read(EXTERNAL / "d_complete.json")["files"]
    hard = load(hard_path)
    scored = {}
    folder = RESULTS / "candidates/d"
    folder.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row["panel"] != "external":
            continue
        name = row["image"]
        score_path = EXTERNAL / "scores" / (name + ".json")
        projected_path = EXTERNAL / "readouts" / (name + ".pt")
        old = read(score_path)
        features = load(projected_path)
        assert old["rgb"] == features["rgb"] == row["rgb_sha256"]
        assert features["source"] == old["source"] == raw_proof[name]
        assert sha(score_path) == proof["files"][name]
        assert sha(projected_path) == proof["readouts"][name] == old["projections"]
        audit = dict(image=name, seed="d", eligible=old["eligible"], source=old["source"],
                     native_vectors=vector_sha, historical_score=proof["files"][name],
                     stored_projection_sha256=proof["readouts"][name], margin_true=[])
        if old["eligible"]:
            assert old["all_readouts_verified"] and features["visual"].shape == (8, 768)
            assert len(old["items"]) == 8
            errors = []
            for visual, item in zip(features["visual"].double(), old["items"]):
                scores = hard[name].double() @ visual
                margins = (scores[::2] - scores[1::2]).tolist()
                expected = item["hard_margins"] + item["DE"] + item["FG"]
                assert len(margins) == len(expected)
                errors.append(max(abs(a-b) for a, b in zip(margins, expected)))
                c = c_vectors[name].double() @ visual
                audit["margin_true"].append(float(c[0] - c[1]))
            audit["max_ABDEFG_error"] = max(errors)
            assert audit["max_ABDEFG_error"] < 1e-10
            scored[name] = audit["margin_true"]
        else:
            assert features["visual"] is None
        dump(folder / (name + ".json"), audit)
    assert len(scored) == 116
    dump(RESULTS / "external_c.json", scored)


def teacher(rows):
    proof = read(RESULTS / "visual_complete.json")
    old_dir = EXTERNAL.parent / "followup/直接看图"
    old = read(old_dir / "回执.json")
    assert old["source"] == sha(old_dir / "run.py")
    assert (proof["adapter"], proof["ids"], proof["torch"]) == (old["adapter"], old["label_ids"], old["version"])
    assert proof["images"] == len(rows) == 336
    assert proof["lock"] == sha(ROOT / "inputs/lock.json")
    assert proof["code"] == sha(ROOT / "code/visual.py")
    result = {}
    for row in rows:
        name = row["image"]
        path = RESULTS / "visual" / (name + ".json")
        assert sha(path) == proof["files"][name]
        visual = read(path)
        first, second = visual["observations"]
        z0 = first["ab"][0] - first["ab"][1]
        z1 = second["ab"][1] - second["ab"][0]
        sign = 1 - 2 * row["side"]
        assert visual["rgb"] == row["rgb_sha256"] and visual["side"] == row["side"]
        assert visual["options_side0"] == row["options_side0"]
        assert visual["margin_side0"] == (z0 + z1)/2
        assert visual["margin_true"] == sign*visual["margin_side0"]
        verified = z0*sign > 0 and z1*sign > 0
        result[name] = dict(margin_side0=visual["margin_side0"], margin_true=visual["margin_true"],
                            both_orders_true=verified, order_margins_side0=[z0, z1], verified_truth=verified)
    dump(RESULTS / "teacher.json", result)


def combine(rows, vector_sha):
    scores = {}
    errors = {}
    for seed in "ab":
        raw_proof = read(MAIN / "sampling" / (seed + "_complete.json"))["files"]
        vals = {}
        replay_errors = []
        for row in rows:
            if row["panel"] == "external":
                continue
            name = row["image"]
            record = read(RESULTS / "candidates" / seed / (name + ".json"))
            assert record["image"] == name and record["seed"] == seed
            assert record["native_vectors"] == vector_sha and record["source"] == raw_proof[name]
            historical_score = MAIN / "scores" / seed / (name + ".json")
            assert record["historical_score"] == sha(historical_score)
            assert record["max_ABDEFG_error"] < 1e-10 and len(record["margin_true"]) == 8
            vals[name] = record["margin_true"]
            replay_errors.append(record["max_ABDEFG_error"])
        assert len(vals) == 218
        scores[seed] = vals
        errors[seed] = max(replay_errors)
    scores["d"] = read(RESULTS / "external_c.json")
    external_records = [read(p) for p in (RESULTS / "candidates/d").glob("*.json")]
    assert len(external_records) == 118 and len(scores["d"]) == 116
    assert scores["d"] == {v["image"]: v["margin_true"] for v in external_records if v["eligible"]}
    errors["d"] = max(v["max_ABDEFG_error"] for v in external_records if v["eligible"])
    dump(RESULTS / "candidate_margins.json", scores)
    training = {s: {r["image"]: scores[s][r["image"]] for r in rows if r["panel"] == "train"} for s in "ab"}
    dump(RESULTS / "train_candidates.json", training)
    assert len(read(RESULTS / "teacher.json")) == 336
    files = ["candidate_margins.json", "visual_complete.json", "teacher.json", "external_c.json",
             "train_visual.json", "train_candidates.json"]
    scripts = {p.name: sha(p) for p in sorted((ROOT / 'code').iterdir()) if p.is_file()}
    dump(RESULTS / "complete.json", dict(files={f: sha(RESULTS / f) for f in files}, code=scripts,
         native_replay_errors=errors, groups={k: len(v) for k, v in scores.items()},
         candidates=sum(len(v)*8 for v in scores.values()), native_vectors=vector_sha))


if __name__ == "__main__":
    locked = read(ROOT / "inputs/lock.json")
    assert sha(ROOT / "inputs/rows.json") == locked["rows"]
    assert sha(ROOT / "inputs/c_vectors.pt") == locked["vectors"]
    for path, checksum in locked["historical"].items():
        assert sha(REPORTS / path) == checksum
    rows = read(ROOT / "inputs/rows.json")
    action = sys.argv[1]
    if action == "external":
        external(rows, locked["vectors"])
    elif action == "teacher":
        teacher(rows)
    elif action == "all":
        combine(rows, locked["vectors"])
    else:
        raise ValueError(action)

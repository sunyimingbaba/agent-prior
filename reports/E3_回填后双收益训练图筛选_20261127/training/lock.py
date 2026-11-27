"""Lock only the code, model and per-image inputs that the four training arms consume."""
from pathlib import Path
import hashlib
import json
import sys

stage = Path(__file__).resolve().parents[1]
source = stage.parent/"second_answer_5090_20261124"
graded = stage.parent/"graded_r2_detail_20261127"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    gate = read(stage/"scoring/results/summary.json")
    cases = read(stage/"scoring/locks/training.json")["rows"]
    if not gate["replicated"] or gate["primary"]["difference"]["interval"][0] <= 0 or len(cases) < 2:
        print("Selection criterion not met. Training and external evaluation skipped.")
        return 3
    assert len(cases) <= 24 and cases == sorted(cases,key=lambda r:r["rgb_sha256"])
    assert len({r["rgb_sha256"] for r in cases}) == len(cases)
    all_rows = {r["image"]:r for r in read(stage/"sampling/inputs/roster.json")}
    scored_rows = {r["image"]:r for r in read(stage/"scoring/inputs/rows.json")}
    for row in cases:
        original = all_rows[row["image"]]
        assert row == scored_rows[row["image"]] and all(row[k] == v for k,v in original.items())
    assert gate["eligible_for_training"] and gate["training_count"] == len(cases)
    assert gate["locks"]["training"] == sha(stage/"scoring/locks/training.json")
    assert gate["scorer_sha"] == sha(stage/"scoring/code/pool_reader.py")
    assert gate["inputs_sha"] == sha(stage/"scoring/inputs/manifest.json")
    inputs = ("training/run.py","training/lock.py","training/run.sh","training/训练复现口径.md",
              "scoring/code/pool_reader.py","scoring/inputs/manifest.json","scoring/results/summary.json",
              "scoring/locks/training.json","sampling/a_complete.json","sampling/b_complete.json")
    lock = {"n":len(cases),"training_inputs":{f:sha(stage/f) for f in inputs},"historical":{},"contexts":{}}
    old = read(graded/"inputs/train_lock.json")
    lock["graded_bonus"] = sha(graded/"code/detail_reward.py")
    assert lock["graded_bonus"] == old["stage"]["code/detail_reward.py"]
    used = ("code/train.py","code/policy.py","code/reader_client.py","code/shared.py","code/readouts.py",
            "initial.json","R0/adapter/adapter_model.safetensors","R0/adapter/adapter_config.json")
    for seed,path in (("a",source),("b",source/"repeat")):
        lock["historical"][seed] = {f:sha(path/f) for f in used}
        assert all(old[seed][f] == h for f,h in lock["historical"][seed].items())
        proof=read(stage/"sampling"/(seed+"_complete.json"))
        lock["contexts"][seed]={}
        for row in cases:
            n = row["image"]
            h = sha(stage/"sampling/raw"/seed/(n+".pt"))
            assert proof["eligibility"][n] and proof["files"][n]==h
            lock["contexts"][seed][n]=h
    path = stage/"training/train_lock.json"
    if path.exists():
        assert read(path) == lock, "inputs changed since training was locked"
    else:
        temp = path.with_suffix(".part")
        temp.write_text(json.dumps(lock,ensure_ascii=False,indent=2)+"\n")
        temp.replace(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

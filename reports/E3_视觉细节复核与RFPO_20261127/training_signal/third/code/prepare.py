"""Freeze the third paired draw using only the two historical draws and their roster."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parents[1]
POOL = STAGE.parent / "E3_回填后双收益训练图筛选_20261127"
PREV = STAGE.parent / "E3_困难奖励跨图跨措辞校准_20261127"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def dump(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    recorded = ROOT.parent / "复算.json"
    source = POOL / "sampling/inputs/roster.json"
    orig = read(source)
    selected = {x["rgb_sha256"] for x in read(recorded)["rows"]}
    pairs = {(r["cohort"], r["pair"]) for r in orig if r["rgb_sha256"] in selected}
    kept = [r for r in orig if (r["cohort"], r["pair"]) in pairs]
    details = [{k: r[k] for k in ("image", "row", "rgb_sha256", "split", "orig_query", "captions", "feedback")} for r in kept]
    details.sort(key=lambda x: x["rgb_sha256"])
    dir = ROOT / "inputs"
    files = {"roster.json": details, "groups.json": sorted([dict(image=r["image"], cohort=r["cohort"], pair=r["pair"],
              attribute=r["attribute"], side=r["side"], selected=r["rgb_sha256"] in selected) for r in kept],
              key=lambda x:(x["cohort"], x["pair"], -x["selected"]))}
    assert len(kept) == 24 and len(pairs) == len(selected) == 12
    for key in pairs:
        pair = [r for r in files["groups.json"] if (r["cohort"],r["pair"]) == key]
        assert {r["side"] for r in pair} == {0,1} and {r["selected"] for r in pair} == {True,False}
        assert len(set(r["attribute"] for r in pair)) == 1
    rng = {481127000+r["row"]*1000 for r in details}
    assert len(rng) == 24 and not (rng & {s+r["row"]*1000 for s in (211000000, 221000000) for r in orig})
    for f,v in files.items():
        dump(dir/f,v)
    feedback = POOL/"sampling/feedback_verified.json"
    old = [source, recorded, feedback, POOL/"sampling"/"a_complete.json", POOL/"sampling"/"b_complete.json",
           POOL/"scoring/inputs/manifest.json", POOL/"scoring/code/pool_reader.py", PREV/"code/reward.py",
           ROOT.parent/"replay.py", STAGE/"training/run.py"]
    digest = {str(p.relative_to(STAGE.parent)):sha(p) for p in old}
    dump(dir/"roster_lock.json",dict(images=24, pairs=12, roster=sha(dir/"roster.json"),
         groups=sha(dir/"groups.json"), protocol=sha(ROOT/"事前口径.md"), old=digest,
         reader=sha(POOL/"scoring/code/pool_reader.py"), redistribute=sha(PREV/"code/reward.py"),
         reader_inputs=sha(POOL/"scoring/inputs/manifest.json"), replay=sha(ROOT.parent/"replay.py"),
         model=read(POOL/"sampling/a_complete.json")["adapter"], seed_base=481127000, bootstrap_seed=271128,
         scripts={p.name:sha(p) for p in sorted((ROOT/"code").iterdir()) if p.is_file()}))


if __name__ == "__main__":
    main()

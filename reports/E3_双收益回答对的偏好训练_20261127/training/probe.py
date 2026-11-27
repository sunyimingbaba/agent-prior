"""Read final preferences on training pairs and pair-disjoint held inputs, using identical logits."""
from pathlib import Path
import os
import torch
from dpo import STAGE, POOL, SOURCE, SEED, ARM, read, sha, dump, setup, policy, table, build_input, log_snapshot


def inputs():
    training=read(STAGE/"training/data.json")
    probe_file=STAGE/"inputs/probe_pairs.json"
    assert sha(probe_file)==(STAGE/"inputs/probe_pairs.sha256").read_text().strip().split()[0]
    probe=read(probe_file)["rows"]
    assigned=read(STAGE/"inputs/train_pairs.json")["rows"]
    assert {r["image"] for r in assigned}.isdisjoint({r["image"] for r in probe})
    assert {(r["cohort"],r["pair"]) for r in assigned}.isdisjoint({(r["cohort"],r["pair"]) for r in probe})
    assert {r["rgb_sha256"] for r in assigned}.isdisjoint({r["rgb_sha256"] for r in probe})
    sampling=read(POOL/"sampling/b_complete.json")
    scores=read(POOL/"scoring/results/b_scores_sha256.json")
    prepared=[]
    for row in probe:
        name=row["image"]
        assert row["source_seed"]=="b" and sampling["eligibility"][name]
        assert row["raw_sha256"]==sampling["files"][name]
        assert row["score_sha256"]==scores[name]==sha(POOL/"scores/b"/(name+".json"))
        slots=sorted([row["r2_winner"],row["r2_loser"]])
        assert len(set(slots))==2 and set(slots)=={row["r3_winner"],row["r3_loser"]}
        assert row["kind"] in ("joint","anchor") and (row["r2_winner"]!=row["r3_winner"])==(row["kind"]=="joint")
        prepared.append({**row,"slots":slots,"preferences":{arm:slots.index(row[arm+"_winner"]) for arm in ("r2","r3")}})
    return {"train":training,"probe":prepared}


def main():
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"]==":4096:8"
    setup()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    cases=inputs()
    proof=read(STAGE/"training"/("reference_"+SEED+".json"))
    if ARM=="ref":
        src="R0"
        expected=proof["initial"]
    else:
        src=STAGE/"models"/f"{SEED}_{ARM}"
        trained=read(src/"complete.json")
        expected=trained["final"]
        assert trained["initial"]==proof["initial"]
    path=STAGE/"probe"/f"{SEED}_{ARM}.json"
    path.parent.mkdir(exist_ok=True)
    signature=dict(model=expected,initial=proof["initial"],probe=sha(STAGE/"inputs/probe_pairs.json"),train=sha(STAGE/"inputs/train_pairs.json"),
                   implementation=sha(Path(__file__)),policy_code=sha(STAGE/"training/dpo.py"))
    if path.exists():
        assert read(path)["signature"]==signature
        return
    assert sha((SOURCE/src)/"adapter/adapter_model.safetensors")==expected
    model,proc=policy(src)
    for name,param in model.named_parameters():
        if ".lora_" in name:
            param.requires_grad_(True)
    tb=table()
    records={}
    for panel,rows in cases.items():
        records[panel]=[]
        for j,case in enumerate(rows):
            if panel=="train" and ARM=="ref":
                recorded=proof["pairs"][j]
                assert recorded["image"]==case["image"] and recorded["slots"]==case["slots"]
                responses=recorded["responses"]
            else:
                exact,paths=build_input(proc,tb,case,source_seed="a" if panel=="train" else "b")
                responses=log_snapshot(model,exact,paths)
            records[panel].append(dict(image=case["image"],pair=case["pair"],kind=case["kind"],slots=case["slots"],
                                       target_position=case["preferences"]["r3"],responses=responses))
    dump(path,dict(signature=signature,records=records))
    print(SEED,ARM,"preferences",len(records["train"]),len(records["probe"]),flush=True)


if __name__=="__main__":
    main()

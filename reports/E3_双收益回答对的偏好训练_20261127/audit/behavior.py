"""Match first-tool contexts and deployed cold-start behavior by RGB and real first caption."""
from pathlib import Path
import hashlib
import json
import torch
from collections import Counter

HERE=Path(__file__).resolve().parent
REPORTS=HERE.parents[1]
SOURCE=REPORTS/"E3_回填后双收益训练图筛选_20261127/sampling"
FORM=REPORTS/"E3_R3改选收益校准与双批首句验证_20261126/targeted/start_diagnostic"
def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return torch.load(path,map_location="cpu",weights_only=True)
def main():
    architecture=read(HERE/"起点权重.json")
    assets={s:read(SOURCE/(s+"_complete.json")) for s in "ab"}
    anchors=read(FORM/"archives/index.json")
    for i,s in enumerate("ab"):
        assert assets[s]["adapter"]==architecture["sha256"][i]==anchors[s]["adapter"]
    counts=Counter()
    for name in assets["a"]["files"]:
        cases=[]
        for s in "ab":
            p=SOURCE/"raw"/s/(name+".pt")
            assert sha(p)==assets[s]["files"][name]
            cases.append(load(p))
        a,b=cases
        assert all(a[k]==b[k] for k in ["image","rgb","query","first","feedback","eligible"])
        counts["same_first_action"]+=1
        if a["eligible"]:
            assert torch.equal(a["context"]["prefix"],b["context"]["prefix"])
            assert a["context"]["answer"]==b["context"]["answer"]
            counts["same_reached_context"]+=1
            counts["same_eight_paths"]+=int(a["paths"]==b["paths"])
            counts["rng_distinct"]+=int(a["rng"]!=b["rng"])
    sets={s:FORM/"evaluate"/(s+"_R0") for s in "ab"}
    proofs={s:read(sets[s]/"complete.json") for s in "ab"}
    for name in proofs["a"]["images"]:
        cases=[]
        for s in "ab":
            p=sets[s]/(name+".pt")
            assert sha(p)==proofs[s]["images"][name]
            cases.append(load(p))
        a,b=cases
        assert all(a[k]==b[k] for k in ["image","rgb","query","feedback","actions","calls"])
        assert torch.equal(a["prior"],b["prior"]) and torch.equal(a["v"],b["v"])
        counts["identical_full_greedy_rollout_prior_visual"]+=1
    assert counts["same_first_action"]==270 and counts["identical_full_greedy_rollout_prior_visual"]==210
    (HERE/"起点行为.json").write_text(json.dumps(dict(**counts,source={
        "a_rollouts":sha(SOURCE/"a_complete.json"),"b_rollouts":sha(SOURCE/"b_complete.json"),
        "a_formal":sha(sets["a"]/"complete.json"),"b_formal":sha(sets["b"]/"complete.json"),
        "architecture":sha(HERE/"起点权重.json")}),indent=2)+"\n")
    print(counts)
if __name__=="__main__":
    torch.set_num_threads(1)
    main()

"""Verify the original deployed R2 reward and audit the additive term on bound old candidates."""
from pathlib import Path
import ast
import hashlib
import json

import numpy as np
import torch

from detail_reward import graded_bonus

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT.parent
HIST = REPORTS/"E3-2_UCM回填后回答训练_20261124"
ARCH = REPORTS/"E3-1_奖励一致性分析_20261112/空间关系候选奖励核验_20261120"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def rr(value, neg):
    above = np.count_nonzero(neg > value)
    ties = np.count_nonzero(neg == value)
    return (1/np.arange(1+above, 2+above+ties)).mean()


def advantage(values):
    return (values - values.mean())/(values.std(ddof=1)+1e-8)


def original_mapping():
    # Isolate the exact archived function from the CUDA-dependent training module.
    code = HIST/"code/train.py"
    tree = ast.parse(code.read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name=="matched_scores")
    env = {"np":np}
    exec(compile(ast.Module([fn],[]), str(code),"exec"), env)
    return env["matched_scores"]


def abi_checks():
    original = original_mapping()
    checked = 0
    for prefix in ("records","repeat/records"):
        for step in read(HIST/prefix/"R2/steps.json"):
            comp = original(step["native_margin"], 1/(1+np.array(step["proxy_rank"])))
            total = np.array(step["format"]) + comp
            assert np.array_equal(comp, step["retrieval"]) and np.array_equal(total,step["total"])
            np.testing.assert_array_equal(advantage(total),step["advantage"])
            checked += 1
    scores = np.linspace(0,7,8)
    margins = np.array([[-1.,2.],[0.,1.],[3.,1.],[1.,1.],[1.,4.],[1.,7.],[8.,8.],[8.,8.]])
    bonus = graded_bonus(scores, margins)
    assert bonus[0]==bonus[1]==0 and 0<bonus[2]==bonus[3]==bonus[4]==bonus[5]<bonus[6]==bonus[7]==1.
    assert not graded_bonus(np.ones(8), margins).any()
    assert not graded_bonus(scores, np.ones((8,2))).any()
    assert not graded_bonus(scores, np.zeros((8,2))).any()
    perm = [4,6,1,7,0,3,2,5]
    assert np.array_equal(graded_bonus(scores[perm],margins[perm]),bonus[perm])
    assert np.allclose(graded_bonus(2*scores+5,margins),2*bonus)
    assert np.array_equal(graded_bonus(scores,margins[:,::-1]),bonus)
    return checked


def feasible_winner_sets(record):
    # For a proposed winning set, the highest common reachable score is the
    # smallest upper endpoint of its members. All outsiders must be strictly below it.
    base = np.array(record["retrieval"])
    cap = np.ptp(base) / 7.
    detail = np.array(record["held_C"])
    caption = np.array(record["other_MRR"])
    reference = base == base.max()
    current_detail = detail[reference].mean()
    current_caption = caption[reference].mean()
    unbounded = bounded = False
    for mask in range(1, 1 << len(base)):
        slots = np.array([bool(mask & (1 << i)) for i in range(len(base))])
        if detail[slots].mean() <= current_detail or caption[slots].mean() < current_caption-1e-12:
            continue
        unbounded = True
        common = (base[slots]+cap).min()
        if common >= base.max() and (slots.all() or base[~slots].max() < common):
            bounded = True
    achieved = record["C_top_delta"] > 0 and record["MRR_top_delta"] >= -1e-12
    assert not achieved or bounded
    return dict(unbounded=unbounded, bounded=bounded, achieved=bool(achieved))


def main():
    checked = abi_checks()
    roster = read(REPORTS/"E3-2_UCM同起点三奖励短训_20261120/inputs/train.json")
    native = ARCH/"native/features.pt"
    info = read(ARCH/"native/index.json")
    held = ARCH/"heldout/text.pt"
    held_info = read(ARCH/"heldout/index.json")
    bind = read(ARCH/"scoring/binding.json")
    assert sha(native)==info["sha256"] and sha(held)==held_info["text_sha256"]
    assert sha(ARCH/"scoring/vectors.pt")==bind["bank"] and sha(ARCH/"scoring/rows.json")==bind["rows"]
    src = torch.load(ARCH/"scoring/vectors.pt",weights_only=True)
    native_bank = torch.load(native,weights_only=True)
    cbank = torch.load(held,weights_only=True)["features"].double().numpy()
    hf = torch.load(ROOT/"inputs/hard.pt",weights_only=True)
    assert sha(ROOT/"inputs/hard.pt")==read(ROOT/"results/训练细节标注核对.json")["archive_sha256"]
    hbank = hf["features"].double().numpy()
    nbank = native_bank["features"].double().numpy()
    even=np.array([int(hashlib.sha256(",".join(map(str,row)).encode()).hexdigest()[-1],16)%2==0
                   for row in native_bank["tokens"].tolist()])
    v = src["visual"].double().numpy()
    entries = read(ARCH/"scoring/rows.json")
    assert src["order"]==[(x["image"],x["mode"],x["k"]) for x in entries]
    position = {key:i for i,key in enumerate(src["order"])}
    original = original_mapping()
    notices = {}
    for mode in ("original","scene"):
        allgroups=[]
        for case in roster:
            name=case["image"]
            ix=[position[(name,mode,k)] for k in range(8)]
            im=v[ix]
            proof=[]
            for i,row in zip(ix,[entries[j] for j in ix]):
                p=ARCH/"samples"/mode/Path(name).stem/("sample"+str(row["k"])+".pt")
                saved=read(p.with_suffix(".json"))
                assert saved["sha256"]==sha(p) and saved["rgb_sha256"]==case["rgb_sha256"] and saved["row"]==case["row"]
                ht=torch.load(p,weights_only=True)
                assert hashlib.sha256(ht.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==saved["vector_sha256"]==row["native_tensor"]
                assert saved["seed"]==row["seed"] and saved["query"]==case["orig_query" if mode=="original" else "scene_query"]
                proof.append(saved["vector_sha256"])
            ids=info["maps"][name]
            positive=ids["positive"];first=ids["first"];alternatives=sorted(set(positive)-{first})
            assert alternatives and ids["captions"][0]==case["orig_query"]
            fit=even.copy();fit[positive]=False
            test=~even;test[positive]=False
            similarities=im@nbank.T
            native_margin=similarities[:,first]-similarities[:,fit].max(1)
            primary=original(native_margin,1/(1+np.array([entries[i]["rank0"] for i in ix])))
            cols=np.array(hf["maps"][name])
            hard=im@(hbank[cols[:,0]]-hbank[cols[:,1]]).T
            archived_hard=np.array([[entries[i][k] for k in "AB"] for i in ix])
            assert np.abs(hard-archived_hard).max()<2e-4
            bonus=graded_bonus(primary,hard);after=primary+bonus
            a,b=(advantage(x) for x in (primary,after))
            wa,wb=(np.flatnonzero(x==x.max()) for x in (primary,after))
            cc=held_info["columns"][name]
            held_margin=im@(cbank[cc[0]]-cbank[cc[1]])
            detail=(held_margin>0).astype(float)
            caption=np.array([np.mean([rr(s[k],s[test]) for k in alternatives]) for s in similarities])
            # An identical format constant cancels in the within-group gradient and in winner selection.
            allgroups.append(dict(image=name,pair=case["pair"],max_reward=primary.max(),max_bonus=bonus.max(),
                changed_adv=bool(np.max(np.abs(a-b))>1e-10), changed_top=set(wa)!=set(wb),
                top_before=wa.tolist(),top_after=wb.tolist(),retrieval=primary.tolist(),bonus=bonus.tolist(),
                archives=proof, hard=hard.tolist(), held_C=detail.tolist(),other_MRR=caption.tolist(),
                C_top_delta=detail[wb].mean()-detail[wa].mean(),MRR_top_delta=caption[wb].mean()-caption[wa].mean(),
                C_adv_weight_change=np.mean((b-a)*detail),MRR_adv_weight_change=np.mean((b-a)*caption)))
        notices[mode]=dict(groups=len(allgroups),changed_adv=sum(r["changed_adv"] for r in allgroups),
                  changed_top=sum(r["changed_top"] for r in allgroups),
                  means={k:float(np.mean([r[k] for r in allgroups])) for k in ("C_top_delta","MRR_top_delta","C_adv_weight_change","MRR_adv_weight_change")},
                  groups_raw=allgroups)
    for panel in notices.values():
        rows = panel["groups_raw"]
        pairs = sorted({r["pair"] for r in rows})
        pair_sizes = np.array([sum(r["pair"] == key for r in rows) for key in pairs])
        draws = np.random.default_rng(20261127).multinomial(len(pairs), np.full(len(pairs),1/len(pairs)), size=20000)
        panel["natural_pairs"] = len(pairs)
        panel["intervals"] = {}
        options = [feasible_winner_sets(r) for r in rows]
        panel["feasible_detail_with_caption_protection"] = {key:sum(v[key] for v in options) for key in ("unbounded","bounded","achieved")}
        for metric in panel["means"]:
            sums = np.array([sum(r[metric] for r in rows if r["pair"] == key) for key in pairs])
            panel["intervals"][metric] = np.quantile((draws @ sums) / (draws @ pair_sizes), [.025,.975]).tolist()
    output=dict(historical_on_policy_reward_reconstructed=checked, archive_rows=384, archive_scoring_device=bind["device"],
                encoder=read(ROOT/"results/训练细节标注核对.json"), panels=notices)
    (ROOT/"results/历史24图固定候选.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    for mode,values in notices.items():
        print(mode,{k:v for k,v in values.items() if k!="groups_raw"})


if __name__=="__main__":
    torch.set_num_threads(1)
    main()

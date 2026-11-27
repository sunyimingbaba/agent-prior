"""Independently rebuild the R2 and bounded hard scores and inspect counterfactual changes."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
DEPLOY=ROOT.parent/"E3-2_UCM回填后回答训练_20261124"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def standardized(x):
    return (x-x.mean())/(x.std(ddof=1)+1e-8)


def audit(seed,arm,lock,old):
    folder=ROOT/"models"/f"{seed}_{arm}"
    cp=read(folder/"complete.json")
    assert cp["scores"]==sha(folder/"steps.json")
    assert cp["lock"]==sha(ROOT/"inputs/train_lock.json") and cp["initial"]==lock[seed]["R0/adapter/adapter_model.safetensors"]
    steps=read(folder/"steps.json")
    assert len(steps)==cp["steps"]==24 and sum(s["norm"]>0 for s in steps)==cp["nonzero"]
    reconstruct=[]
    for j,(s,hist) in enumerate(zip(steps,old),1):
        assert (s["step"],s["seed"],s["row"],s["image"])==(j,hist["seed"],hist["row"],hist["image"])
        assert len(s["trajectories"])==8 and len(s["continuations"])==8
        assert np.isfinite([s["replay_mean_abs_logp_difference"],s["norm"],s["replay_max_abs_logp_difference"]]).all()
        proxy=1/(1+np.array(s["proxy_rank"]))
        margin=np.array(s["native_margin"])
        sorted_proxy=np.sort(proxy)
        base=np.array([np.mean(sorted_proxy[np.sum(margin<m):np.sum(margin<=m)]) for m in margin])
        b=np.array(s["hard"])
        u=np.maximum(0,b.min(1))
        ranks=rankdata(u,method="average")
        width=np.ptp(base)/7
        bonus=np.zeros(8) if np.ptp(ranks)==0 else width*((ranks-ranks.min())/np.ptp(ranks))
        assert len(set(s["format"]))==1
        r2=base+np.array(s["format"]);r3=r2+bonus
        a2=standardized(r2);a3=standardized(r3)
        for k,exp in (("retrieval",base),("bonus",bonus),("r2",r2),("r3",r3),("adv2",a2),("adv3",a3),("advantage",a2 if arm=="r2" else a3)):
            np.testing.assert_allclose(s[k],exp,rtol=0,atol=1e-13)
        assert np.all(bonus>=0) and np.all(bonus<=width+1e-15)
        top2=r2==r2.max();top3=r3==r3.max()
        order2=np.sign(r2[:,None]-r2);order3=np.sign(r3[:,None]-r3)
        rr=np.array(s["other_caption_rr"]).mean(1)
        assert np.isfinite(rr).all()
        delta=a3-a2;units=np.maximum(delta,0).sum()
        changed=bool(np.linalg.norm(delta)>1e-10)
        reconstruct.append(dict(step=j,image=s["image"],varying_bonus=bool(np.ptp(bonus)>0),different_advantage=changed,
                         different_winners=not np.array_equal(top2,top3),different_order=not np.array_equal(order2,order3),
                         different_positive_slots=not np.array_equal(a2>1e-10,a3>1e-10),max_advantage_move=float(np.abs(delta).max()),
                         mean_absolute_advantage_move=float(np.abs(delta).mean()),other_caption_rr_winner_delta=float(rr[top3].mean()-rr[top2].mean()),
                         hard_correct_winner_delta=float((b[top3]>0).mean()-(b[top2]>0).mean()),
                         shifted_rr_advantage_contrast=float(np.dot(delta,rr)/units) if changed else None))
    flags={k:sum(r[k] for r in reconstruct) for k in ("varying_bonus","different_advantage","different_winners","different_order","different_positive_slots")}
    first=steps[0];previous=old[0]
    reproduction={
      "first_original_trajectories_match":[t["ids"] for t in first["trajectories"]]==[t["ids"] for t in previous["trajectories"]],
      "first_original_proxy_ranks_match":first["proxy_rank"]==previous["proxy_rank"],
      "first_original_native_margins_maxdiff":float(np.max(np.abs(np.array(first["native_margin"])-previous["native_margin"]))),
      "first_original_r2_scores_maxdiff":float(np.max(np.abs(np.array(first["r2"])-previous["total"])))}
    return steps,dict(nonzero=cp["nonzero"],**flags,original_reference=reproduction,
                      mean_absolute_advantage_move=float(np.mean([r["mean_absolute_advantage_move"] for r in reconstruct])),
                      max_replay_error=max(r["replay_max_abs_logp_difference"] for r in steps),groups=reconstruct)


def main():
    report={}
    lock=read(ROOT/"inputs/train_lock.json")
    for s in "ab":
        old=read((DEPLOY if s=="a" else DEPLOY/"repeat")/"records/R2/steps.json")
        steps={};stats={}
        for arm in ("r2","r3"):
            steps[arm],stats[arm]=audit(s,arm,lock,old)
        diverged=None
        for x,y in zip(steps["r2"],steps["r3"]):
            assert all(x[k]==y[k] for k in ("image","seed","proxy_rank","native_margin","hard","other_caption_rr","trajectories","continuations"))
            if x["advantage"]!=y["advantage"]:
                diverged=x["step"]
                break
            assert x["norm"]==y["norm"]
        report[s]={"arms":stats,"first_different_advantage":diverged}
    (ROOT/"results/逐候选训练核查.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()

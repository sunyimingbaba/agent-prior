"""Recompute direction, gradient multiplier and exact action counts for logged DPO updates."""
from pathlib import Path
import json
import math
from pairs import HERE,read,sha,save

def main():
    root=HERE.parent
    pick=read(HERE/"独立A选对.json")["ordered"]
    bases=[read(root/"training"/f"reference_{s}.json")["pairs"] for s in "ab"]
    assert bases[0]==bases[1]
    total=sum(len(w["tokens"]) for z in bases[0] for w in z["responses"])
    checked=0; err=0.;nonzero={};references={}
    for seed in "ab":
        for arm in ["r2","r3"]:
            d=root/"models"/f"{seed}_{arm}"
            sc=read(d/"complete.json")
            assert sha(d/"steps.json")==sc["scores"]
            refs=root/"training"/f"reference_{seed}.json"
            assert sha(refs)==sc["reference"]
            references[seed]=sha(refs)
            logs=read(d/"steps.json")
            assert len(logs)==len(pick)
            nonzero[seed+"_"+arm]=sum(z["norm"]>0 for z in logs)
            for n,(p,z,ref) in enumerate(zip(pick,logs,bases[0])):
                assert z["step"]==n+1 and z["image"]==p["image"]==ref["image"]
                win,lose=(p["preferred"],p["rejected"]) if arm=="r3" else (p["baseline"],p["alternative"])
                assert (z["chosen_slot"],z["rejected_slot"])==(win,lose)
                j=z["slots"].index(win); k=z["slots"].index(lose)
                for item,origin in zip(z["responses"],ref["responses"]):
                    assert item["tokens"]==origin["tokens"] and len(item["tokens"])==len(item["per_token"])
                cur=z["responses"][j]["logp"]-z["responses"][k]["logp"]
                base=ref["responses"][j]["logp"]-ref["responses"][k]["logp"]
                t=0.1*(cur-base)
                loss=math.log1p(math.exp(-t))
                scale=.1/(1+math.exp(t))
                err=max(err,abs(loss-z["loss"]),abs(scale-z["gradient_scale"]))
                if n==0:
                    assert cur==base and abs(scale-.05)<1e-14
                checked+=1
    assert checked==96 and err<1e-12
    save(HERE/"四臂偏好更新验算.json",dict(steps=checked,response_tokens_per_arm=total,
        reference_scores_and_tokens_identical_across_seeds=True,reference_files=references,
        max_error_loss_and_gradient_scale=err,nonzero_updates=nonzero))
if __name__=="__main__":main()

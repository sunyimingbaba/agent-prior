"""On identical sequences, compare what actions win under the two training rewards."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
def read(p):
    return json.loads(p.read_bytes())
def top_weights(vals):
    mask = vals == np.max(vals)
    return mask / mask.sum()

def measure(record):
    acts = record["actions"]
    assert len(acts)==8
    first_end = np.array([a[0]["ids"]==[151645] for a in acts])
    called = np.array([len(a)>1 for a in acts])
    fmt = np.asarray(record["format"])
    assert not np.any(first_end & called)
    r2,r3 = (np.asarray(record[k]) for k in ("r2","r3"))
    indicators = dict(immediate_end=first_end,tool_followup=called,format_score=fmt,
                      AB_accuracy=(np.array(record["hard"])>0).mean(axis=1),
                      first_caption_rr=1/(1+np.asarray(record["proxy_rank"])),
                      other_caption_rr=np.array([np.mean(x) for x in record["other_caption_rr"]]))
    levels = {key:{m:float(np.dot(w,x)) for m,x in indicators.items()} for key,w in (
        ("draw",np.ones(8)/8),("r2_winner",top_weights(r2)),("r3_winner",top_weights(r3)))}
    return dict(image=record["image"],step=record["step"],levels=levels,
                winner_changed=not np.array_equal(top_weights(r2),top_weights(r3)),
                has_eos_and_alternative=bool(first_end.any() and not first_end.all()))

def main():
    rows = []
    for seed in "ab":
        for arm in ("r2","strict"):
            folder = HERE.parent/"models"/(seed+"_"+arm)
            lock = read(folder/"complete.json")
            p = folder/"steps.json"
            assert hashlib.sha256(p.read_bytes()).hexdigest() == lock["scores"] and lock["steps"]==24
            rows.extend(dict(seed=seed,arm=arm,**measure(v)) for v in read(p))
    assert len(rows)==96 and len({r["image"] for r in rows})==24
    cells = {}
    for name,subset in [("all",rows)] + [(s+"_"+a,[r for r in rows if r["arm"]==a and r["seed"]==s])
                                          for s in "ab" for a in ("strict","r2")]:
        levels = {k:{m:float(np.mean([r["levels"][k][m] for r in subset])) for m in subset[0]["levels"][k]}
                  for k in ("draw","r2_winner","r3_winner")}
        cells[name] = dict(groups=len(subset),mixed_eos_groups=sum(r["has_eos_and_alternative"] for r in subset),
                           winner_changed=sum(r["winner_changed"] for r in subset),levels=levels,
                           paired_change={m:levels["r3_winner"][m]-v for m,v in levels["r2_winner"].items()})
    (HERE/"结果.json").write_text(json.dumps(dict(summary=cells,groups=rows),indent=2,ensure_ascii=False)+"\n")

if __name__=="__main__":
    main()

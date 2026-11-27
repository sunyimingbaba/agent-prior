"""Count better candidates that are actually present during after-tool RFPO training."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
out = {}
for seed in "ab":
    rows = json.loads((ROOT/"models"/(seed+"_r3")/"steps.json").read_text())
    groups = []
    for r in rows:
        score=np.array(r["r2"])
        top=score==score.max()
        detail=(np.array(r["hard"])>0).mean(1)
        native=np.array(r["other_caption_rr"]).mean(1)
        improved=detail>detail[top].mean()
        useful=improved & (native>=native[top].mean())
        groups.append(dict(image=r["image"],step=r["step"],different_detail=bool(np.ptp(detail)>0),
                           better_detail=bool(improved.any()),joint_slots=np.flatnonzero(useful).tolist(),
                           r2_slots=np.flatnonzero(top).tolist(),correct=detail.tolist(),other_MRR=native.tolist()))
    out[seed]=dict(groups=groups,vary=sum(g["different_detail"] for g in groups),
                   any_detail_gain=sum(g["better_detail"] for g in groups),
                   detail_gain_with_native_held=sum(bool(g["joint_slots"]) for g in groups))
(ROOT/"results/训练组可选空间.json").write_text(json.dumps(out,indent=2)+"\n")

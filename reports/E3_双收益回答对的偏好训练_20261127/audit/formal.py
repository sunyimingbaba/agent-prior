"""Replay the blinded two-arm formal contrasts from final image and text features."""
from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
OLD=STAGE.parent/"E3_回填后双收益训练图筛选_20261127"
src=OLD/"evaluation/code/independent.py"
spec=spec_from_file_location("previous_independent",src)
m=module_from_spec(spec)
spec.loader.exec_module(m)

def run():
    gal=m.read(m.FORM/"inputs/ucm.json")
    ids={r["rgb_sha256"]:r["image"] for r in gal}
    panel=[r for r in m.read(m.FINE/"inputs/panel.json") if r["rgb_sha256"] in ids]
    assert len(panel)==106 and len(gal)==210
    bank=m.load(m.FORM/"results/ucm_text.pt"); hard=m.load(m.FINE/"records/text.pt")
    data,proof=m.collect(STAGE/"evaluation",gal,panel,bank,hard)
    ref=m.read(STAGE/"evaluation/results/单次检索对照.json")
    values={}
    for key in ("DE","mrr","mR"):
        rows=gal if key=="mR" else panel
        groups=[r["image" if key=="mR" else "pair"] for r in rows]
        levels={tag:m.summarize(data[tag,key],groups) for tag in ("r2","r3")}
        delta=m.summarize(data["r3",key]-data["r2",key],groups)
        target=ref["official" if key=="mR" else "first106"]["contrasts"]["r3-r2"][key]
        assert np.allclose([delta["mean"],*delta["interval"],*[v["mean"] for v in delta["seeds"].values()]],
                           [target["mean"],*target["interval"],*target["seeds"]],rtol=0,atol=1e-12)
        values[key]=dict(levels=levels,r3_minus_r2=delta)
    criteria=dict(detail_lower_positive=values["DE"]["r3_minus_r2"]["interval"][0]>0,
                  mR_preserved=values["mR"]["r3_minus_r2"]["mean"]>=0,
                  other_caption_MRR_preserved=values["mrr"]["r3_minus_r2"]["mean"]>=0)
    assert criteria==ref["criteria"]
    audit=dict(criteria=criteria,passes=all(criteria.values()),metrics=values,feature_proof=proof,
               calculation_source=m.sha(src),code_sha256=m.sha(Path(__file__)))
    (HERE/"正式指标独立复算.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n")
    print(criteria)
if __name__=="__main__":run()

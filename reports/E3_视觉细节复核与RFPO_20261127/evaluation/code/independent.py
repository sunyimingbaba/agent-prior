"""Verify the three endpoints from the raw embeddings and the equal-size paired units."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json
import numpy as np

EVAL=Path(__file__).resolve().parents[1]
POOL=EVAL.parents[1]/"E3_回填后双收益训练图筛选_20261127/evaluation"
path=POOL/"code/independent.py"
spec=spec_from_file_location("pool_independent",path)
verify=module_from_spec(spec)
spec.loader.exec_module(verify)
read,sha,load=verify.read,verify.sha,verify.load

def main():
    formal=read(EVAL/"results/单次检索对照.json")
    gallery=read(verify.FORM/"inputs/ucm.json")
    names={r["rgb_sha256"]:r["image"] for r in gallery}
    subset=[r for r in read(verify.FINE/"inputs/panel.json") if r["rgb_sha256"] in names]
    assert len(gallery)==210 and len(subset)==106
    bank=load(verify.FORM/"results/ucm_text.pt")
    hard=load(verify.FINE/"records/text.pt")
    raw_file=EVAL/"results/逐图指标.json"
    assert sha(raw_file)==read(EVAL/"results/complete.json")["files"]["results/逐图指标.json"]
    raw={(r["seed"],r["arm"],r["image"]):r for r in read(raw_file)}
    observed={}
    max_error=0
    for arm,base in (("strict",EVAL),("r2",POOL),("r3",POOL)):
        for seed in "ab":
            f=base/"records"/(seed+"_"+arm)/"features.pt"
            assert sha(f)==read(f.with_name("complete.json"))["features"]
            z=load(f)
            assert z["images"]==[r["image"] for r in gallery]
            res=verify.independently_score(z["features"].double().numpy(),gallery,subset,bank,hard)
            for key,obs in res.items():
                metric="official" if key=="mR" else "first106"
                rows=gallery if key=="mR" else subset
                values=[raw[seed,arm,r["image"]][metric][key] for r in rows]
                max_error=max(max_error,float(np.abs(obs-values).max()))
                assert max_error<1e-12
                observed[arm,seed,key]=obs
    checks={}
    for key in ("mR","DE","mrr"):
        if key=="mR":
            units=[r["image"] for r in gallery]; metric="official"
        else:
            units=[r["pair"] for r in subset]; metric="first106"
        checks[key]={}
        for arm in ("r2","r3"):
            delta=np.stack([observed["strict",s,key]-observed[arm,s,key] for s in "ab"])
            tab=verify.summarize(delta,units)
            expected=formal[metric]["contrasts"]["strict-"+arm][key]
            np.testing.assert_allclose([tab["mean"],*tab["interval"],*[v["mean"] for v in tab["seeds"].values()]],
                                       [expected["mean"],*expected["interval"],*expected["seeds"]],rtol=0,atol=1e-12)
            checks[key]["strict-"+arm]=tab
    out=dict(per_image_max_abs_error=max_error,raw_sha256=sha(raw_file),reference_code_sha256=sha(path),metrics=checks)
    (EVAL/"results/独立验算.json").write_text(json.dumps(out,indent=2)+"\n")

if __name__=="__main__":
    main()

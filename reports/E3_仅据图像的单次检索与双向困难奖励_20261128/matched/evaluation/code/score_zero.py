"""Compute absolute R0 results and an analytically identical reward contrast after zero updates."""
from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
import gzip,json,sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parents[1]/"image_only"
sys.path.insert(0,str(OLD/"code"))
spec=spec_from_file_location("common_image_score",OLD/"code/score.py")
core=module_from_spec(spec);spec.loader.exec_module(core)
read,sha,dump=core.read,core.sha,core.dump

def main():
    lock=read(ROOT/"inputs/manifest.json");server=read(ROOT/"inputs/server_lock.json");roster=read(ROOT/"inputs/roster.json")
    assert server["manifest"]==sha(ROOT/"inputs/manifest.json") and len(lock["no_eligible"])==4
    fb=read(ROOT/"feedback_verified.json")["checks"]
    assert sha(ROOT/"feedback_verified.json")==server["feedback"]
    gal=read(core.FORM/"inputs/ucm.json");ids={r["rgb_sha256"]:i for i,r in enumerate(gal)}
    fine=[r for r in read(core.FINE/"inputs/panel.json") if r["rgb_sha256"] in ids]
    ext=read(core.OLD/"extra164/inputs/rows.json");gr={"official":gal,"first106":fine,"extra164":ext}
    labels={p:[r.get("pair",r["image"]) for r in rows] for p,rows in gr.items()}
    native=torch.load(core.FORM/"results/ucm_text.pt",weights_only=True);hard=torch.load(core.FINE/"records/text.pt",weights_only=True)
    et=torch.load(core.OLD/"extra164/inputs/texts.pt",weights_only=True);cols=native["caption_order"].numpy().reshape(210,5)
    assets={};values={p:{"r2":{},"strict":{}} for p in gr};observations=[]
    for s in "ab":
        tag=s+"_R0";vec={}
        for panel in ("official","extra"):
            rs=[r for r in roster if r["panel"]==panel]
            vec[panel],assets[tag+"_"+panel]=core.inspect(ROOT,tag,panel,rs,lock,server,fb)
        metrics={"official":core.calculate("official",vec["official"],gal),"first106":core.details(vec["official"],fine,ids,cols,native,hard),
                 "extra164":core.metrics_extra(ext,et,vec["extra"])}
        for arm in ("r2","strict"):
            cp=lock["policies"][s+"_"+arm]
            assert cp["actual_output"]==tag and cp["steps"]==0 and cp["adapter"]==lock["policies"][tag]["adapter"]
            for p,z in metrics.items():
                values[p][arm][s]=z
                for i,r in enumerate(gr[p]):
                    obs={k:float(v[i]) if np.isfinite(v[i]) else None for k,v in z.items()}
                    observations.append(dict(seed=s,arm=arm,actual_policy=tag,panel=p,image=r["image"],rgb=r["rgb_sha256"],metrics=obs))
    assert len({k for z in assets.values() for k in z["first_input_token_hashes"]})==len({z["template"] for z in assets.values()})==1
    result=core.summarize(values,labels)
    assert all(v["mean"]==0 and v["interval"]==[0.,0.] for p in gr for v in result[p]["strict_minus_r2"].values())
    result.update(status="0 of 24 images in each initialization requested a second round; no trained checkpoints",
        training_steps=0,trained_checkpoints=0,actual_R0_predictions=748,zero_difference_reason="identical R0 within each seed on each image and identical decoder",
        formal_pass=False,joint_pass=False)
    dump(ROOT/"archives/index.json",assets);dump(ROOT/"results/单次检索.json",result)
    with gzip.open(ROOT/"results/逐图指标.jsonl.gz","wt",encoding="utf8") as stream:
        for r in observations:stream.write(json.dumps(r,allow_nan=False)+"\n")
    files=["inputs/manifest.json","inputs/server_lock.json","inputs/evaluation_lock.json","inputs/roster.json","feedback_verified.json",
            "archives/index.json","results/单次检索.json","results/逐图指标.jsonl.gz","code/zero_steps.py","code/zero_steps.sh","code/score_zero.py"]
    dump(ROOT/"results/complete.json",dict(images=374,predictions=748,trained_models=0,initial_models=2,files={f:sha(ROOT/f) for f in files},
        common_scorer=sha(OLD/"code/score.py")))
if __name__=="__main__":
    torch.set_num_threads(1);main()

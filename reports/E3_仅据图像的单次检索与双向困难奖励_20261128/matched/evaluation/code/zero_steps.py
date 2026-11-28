"""Verify empty training arms, then infer each distinct initial policy once."""
import sys,shutil
from pathlib import Path
from run import ROOT,TRAIN,OLD,common,sha,read,dump

def source(seed):
    base=common.BASE/"second_answer_5090_20261124"
    return (base if seed=="a" else base/"repeat")/"R0"

def verify():
    assert (TRAIN/"training.done").is_file()
    pin=read(ROOT/"inputs/evaluation_lock.json")
    assert all(sha(OLD/f)==h for f,h in pin["common_files"].items())
    assert all(sha(ROOT/f)==h for f,h in pin["matched_files"].items())
    lock=read(OLD/"inputs/manifest.json");policies={};proofs={}
    for s in "ab":
        f=source(s)/"adapter/adapter_model.safetensors";init=sha(f)
        ctx=TRAIN/"contexts"/s/"complete.json"
        for arm in ("r2","strict"):
            tag=s+"_"+arm;d=TRAIN/"models"/tag;no=d/"no_eligible.json"
            z=read(no)
            assert z["initial"]==init==lock["policies"][tag]["initial"] and z["steps"]==0 and z["scheduled_slots"]==24
            assert z["contexts"]==sha(ctx) and z["seed"]==s and z["arm"]==arm
            assert not (d/"complete.json").exists() and not (d/"adapter/adapter_model.safetensors").exists()
            policies[tag]=dict(adapter=init,initial=init,steps=0,actual_output=s+"_R0")
            proofs[tag]=sha(no)
        policies[s+"_R0"]=dict(adapter=init,initial=init,steps=0)
    manifest=dict(query=pin["query"],policies=policies,evaluation_lock=sha(ROOT/"inputs/evaluation_lock.json"),
                  no_eligible=proofs,source=sha(OLD/"inputs/manifest.json"),scoring_inputs=lock["score_inputs"],actual_models=2,fresh_predictions=748)
    f=ROOT/"inputs/manifest.json"
    if f.exists():assert read(f)==manifest
    else:dump(f,manifest)
    feedback=ROOT/"feedback_verified.json";old=read(OLD/"inputs/server_lock.json")
    if not feedback.exists():shutil.copyfile(OLD/feedback.name,feedback)
    assert sha(feedback)==old["feedback"] and sha(ROOT/"inputs/roster.json")==sha(OLD/"inputs/roster.json")
    z=dict(old,manifest=sha(f),common_server=sha(OLD/"inputs/server_lock.json"),zero_step_code=sha(Path(__file__)))
    f=ROOT/"inputs/server_lock.json"
    if f.exists():assert read(f)==z
    else:dump(f,z)
    return manifest,read(ROOT/"inputs/roster.json")
if __name__=="__main__":
    common.ROOT=ROOT;common.locked=verify;common.weights=lambda tag:source(tag.split("_")[0])
    if sys.argv[1]=="verify":verify()
    else:common.generate(sys.argv[1])

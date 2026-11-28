"""Attach the common image-only decoding implementation to the preselected matched checkpoints."""
from pathlib import Path
from importlib.util import module_from_spec,spec_from_file_location
import shutil,sys
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parents[1]/"image_only"; TRAIN=ROOT.parent
spec=spec_from_file_location("common_image_predict",OLD/"code/predict.py")
common=module_from_spec(spec);spec.loader.exec_module(common)
read,dump,sha=common.read,common.dump,common.sha
def prepare():
    pin=read(ROOT/"inputs/evaluation_lock.json")
    assert all(sha(ROOT/f)==h for f,h in pin["matched_files"].items())
    assert all(sha(OLD/f)==h for f,h in pin["common_files"].items())
    assert (TRAIN/"training.done").is_file()
    orig=read(OLD/"inputs/manifest.json")
    assert orig["query"]==pin["query"] and sha(ROOT/"inputs/roster.json")==sha(OLD/"inputs/roster.json")
    lockfile=TRAIN/"inputs/train_lock.json";roster=TRAIN/"inputs/roster.json"; policies={}
    for s in "ab":
        orders={}
        for a in ("r2","strict"):
            tag=s+"_"+a;d=TRAIN/"models"/tag;f=d/"complete.json";cp=read(f)
            assert cp["lock"]==sha(lockfile) and cp["train"]==sha(roster) and cp["scheduled_slots"]==24
            assert 1<=cp["steps"]<=24 and cp["scores"]==sha(d/"steps.json")
            assert cp["final"]==sha(d/"adapter/adapter_model.safetensors") and cp["initial"]==orig["policies"][tag]["initial"]
            steps=read(d/"steps.json")
            orders[a]=[r["image"] for r in steps]
            assert len(set(orders[a]))==len(orders[a])==cp["steps"]
            policies[tag]=dict(adapter=cp["final"],proof=sha(f),steps=cp["steps"],initial=cp["initial"])
        assert orders["r2"]==orders["strict"]
    manifest=dict(query=pin["query"],policies=policies,evaluation_lock=sha(ROOT/"inputs/evaluation_lock.json"),
                 training_lock=sha(lockfile),common=sha(OLD/"inputs/manifest.json"),score_inputs=orig["score_inputs"])
    f=ROOT/"inputs/manifest.json"
    if f.exists():assert manifest==read(f)
    else:dump(f,manifest)
    oldserver=read(OLD/"inputs/server_lock.json")
    assert oldserver["manifest"]==sha(OLD/"inputs/manifest.json")
    feedback=ROOT/"feedback_verified.json"
    if not feedback.exists():shutil.copyfile(OLD/"feedback_verified.json",feedback)
    assert sha(feedback)==oldserver["feedback"]
    server=dict(oldserver,manifest=sha(f),common_server=sha(OLD/"inputs/server_lock.json"))
    f=ROOT/"inputs/server_lock.json"
    if f.exists():assert read(f)==server
    else:dump(f,server)
    return manifest,read(ROOT/"inputs/roster.json")
if __name__=="__main__":
    common.ROOT=ROOT
    common.weights=lambda tag:TRAIN/"models"/tag
    common.locked=prepare
    if sys.argv[1]=="verify":prepare()
    else:common.generate(sys.argv[1])

"""Greedy first-caption DPO inference with original image, tool feedback, and frozen embedding readout."""
from pathlib import Path
import inspect
import os
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]
FORM=BASE/"after_tool_holdout_20261124"
DPO=BASE/"joint_answer_pairs_20261127"
sys.path.insert(0,str(BASE/"dual_input_rfpo_20261126/code"))
from model_run import setup,read,dump,sha,savept,rgb,policy,reader,greedy,TRANS
from cross_route import dataset

def check_inputs():
    lock=read(ROOT/"inputs/manifest.json")
    for f,h in lock["files"].items():
        assert sha(ROOT/f)==h,f
    sources=dict(roster=FORM/"inputs/rsitmd.json",gallery_text=FORM/"results/rsitmd_text.pt",
        arrow=FORM/"inputs/rsitmd_caption_karpathy_label_test.arrow",tool_cache=FORM/"tools.json",
        train_pairs=DPO/"inputs/train_pairs.json", historical_scores=FORM/"results/paired_results.json")
    assert all(sha(f)==lock[k] for k,f in sources.items())
    assert all(sha(BASE/"source_transfer_2h_20261125/code"/n)==h for n,h in lock["scorer_sources"].items())
    assert read(FORM/"environment.json")["scene_head"]==lock["scene_head"]
    for tag,v in lock["policies"].items():
        folder=DPO/"models"/tag
        assert sha(folder/"complete.json")==v["train_proof"] and sha(folder/"steps.json")==v["scores"]
        assert sha(folder/"adapter/adapter_model.safetensors")==v["adapter"]
    trace=BASE/"source_transfer_2h_20261125/scratch/export"
    proof=read(trace/"proof.json")
    assert sha(trace/"cup_native_trace.pt")==proof["trace"]
    impl={f.__name__:sha(Path(inspect.getfile(f))) for f in (policy,greedy,rgb,reader,dataset,setup)}
    env=dict(lock=sha(ROOT/"inputs/manifest.json"),reader=proof["trace"],implementation=impl)
    sig=ROOT/"inputs/server_lock.json"
    if sig.exists():assert env==read(sig)
    else:dump(sig,env)
    return lock

@torch.inference_mode()
def run(seed,arm):
    assert seed in "ab" and arm in ("r2","r3")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"]==":4096:8"
    setup()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(2)
    lock=check_inputs()
    tag=seed+"_"+arm; folder=DPO/"models"/tag; sig=lock["policies"][tag]["adapter"]
    dest=ROOT/"records"/tag;dest.mkdir(parents=True,exist_ok=True)
    policy_model,proc=policy(folder); native=reader()
    table,rows,feedback=dataset()
    assert len(rows)==448
    vectors=[];hashes={}
    lk=sha(ROOT/"inputs/server_lock.json")
    for j,row in enumerate(rows):
        n=row["image"];q=row["orig_query"];r=row["row"];fb=feedback[row["rgb_sha256"]][0]
        assert table["path"][r].as_py()==n and row["captions"]==table["caption"][r].as_py() and q==row["captions"][0]
        im=rgb(table,row)
        f=dest/(n+".pt")
        key=dict(image=n,rgb=row["rgb_sha256"],query=q,feedback=fb,adapter=sig,lock=lk)
        if f.exists():
            z=torch.load(f,map_location="cpu",weights_only=True)
            assert all(z[k]==v for k,v in key.items())
        else:
            h,act,calls,_=greedy(policy_model,proc,im,q,fb)
            v=native(TRANS(im)[None].cuda(),h.float()[None].cuda())[0].cpu()
            z=dict(**key,prior=h,v=v,actions=act,calls=calls)
            savept(f,z)
        assert z["v"].shape==(768,) and z["prior"].shape==(3584,) and z["actions"] and all(x==fb for x in z["calls"])
        assert torch.isfinite(z["v"]).all() and torch.isfinite(z["prior"]).all()
        vectors.append(z["v"]);hashes[n]=sha(f)
        if (j+1)%100==0:print(tag,j+1,flush=True)
    f=dest/"features.pt"
    savept(f,dict(images=[r["image"] for r in rows],features=torch.stack(vectors),adapter=sig,lock=lk))
    dump(dest/"complete.json",dict(features=sha(f),images=hashes,adapter=sig,lock=lk,torch=str(torch.__version__),
        device=torch.cuda.get_device_name(0),cuda=torch.version.cuda))
    print(tag,len(rows),flush=True)

if __name__=="__main__":
    if sys.argv[1]=="lock":check_inputs()
    else:run(*sys.argv[1:])

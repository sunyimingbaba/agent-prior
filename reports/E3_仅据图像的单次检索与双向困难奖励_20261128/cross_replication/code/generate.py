"""Use the same image-only greedy policy, but with the fixed cross-domain images and options."""
import os
import sys
from pathlib import Path
import torch
import pyarrow as pa

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT.parent;BASE=STAGE.parent;FORM=BASE/"after_tool_holdout_20261124"
sys.path.insert(0,str(STAGE/"image_only/code"))
from predict import PromptCheck,greedy,sha,read,dump,policy,reader,pixels,setup,TRANS,savept,digest,ASSETS


def sources():
    lock=read(ROOT/"inputs/manifest.json"); gate=read(ROOT/"inputs/gate.json")
    assert gate["lock"]==sha(ROOT/"inputs/manifest.json") and gate["branch"]!="skip"
    assert all(sha(ROOT/n)==h for n,h in lock["files"].items())
    assert sha(STAGE/"image_only/code/predict.py")==lock["external"]["query_check"]
    assert sha(FORM/"inputs/rsitmd.json")==lock["prior_gallery"] and sha(FORM/"tools.json")==lock["tool_cache"]
    rows=read(ROOT/"inputs/roster.json"); donors=read(ROOT/"inputs/donors.json")
    assert len(rows)==448 and all(set(r)=={"image","rgb","row","feedback"} for r in rows)
    return lock,gate,rows,donors


def arrow(lock):
    f=FORM/"inputs/rsitmd_caption_karpathy_label_test.arrow"
    assert sha(f)==lock["arrow"]
    return pa.ipc.open_file(pa.memory_map(str(f))).read_all()


@torch.inference_mode()
def verify():
    from prompt_clip import clip
    from tools import scene_classify_with_conf
    lock,gate,rows,donors=sources();setup();tb=arrow(lock)
    head=ASSETS/"ucm_scene_cls.pt";clip_path=ASSETS/"models/ViT-L-14.pt"
    assert sha(head)==lock["head"] and sha(clip_path)==read(STAGE/"image_only/feedback_verified.json")["clip"]
    model,pre=clip.load(str(clip_path),device="cpu",jit=False);model=model.float().eval().cuda()
    ck=torch.load(head,map_location="cpu",weights_only=True);labels=ck["classes"]
    classifier=torch.nn.Linear(ck["feat_dim"],len(labels));classifier.load_state_dict(ck["head"]);classifier=classifier.eval().cuda()
    results={}
    for r in rows:
        im=pixels(tb,r);fb,conf=scene_classify_with_conf(im,model=model,preprocess=pre,cls_head=classifier,cls_names=labels)
        assert fb==r["feedback"]
        results[r["rgb"]]=dict(feedback=fb,confidence=float(conf))
    trace=BASE/"source_transfer_2h_20261125/scratch/export";sig=read(trace/"proof.json")["trace"]
    assert sha(trace/"cup_native_trace.pt")==sig
    dump(ROOT/"inputs/verified.json",dict(items=results,clip=sha(clip_path),head=sha(head),reader=sig,gate=sha(ROOT/"inputs/gate.json")))


@torch.inference_mode()
def generate(tag):
    lock,gate,rows,donors=sources()
    assert tag in gate["tags"]
    seed,arm,cond=tag.split("_");policy_tag=seed+"_"+arm
    expected=gate["policies"][tag]
    source=(STAGE/"first_round" if gate["branch"]=="first_round" else BASE/"visual_detail_rfpo_20261127")/"models"/policy_tag
    assert sha(source/"complete.json")==expected["proof"] and sha(source/"adapter/adapter_model.safetensors")==expected["adapter"]
    proof=read(ROOT/"inputs/verified.json");assert proof["gate"]==sha(ROOT/"inputs/gate.json")
    feedback=proof["items"];tb=arrow(lock)
    setup();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"]==":4096:8"
    model,proc=policy(source);native=reader()
    folder=ROOT/"records"/tag;folder.mkdir(parents=True,exist_ok=True)
    files={};vectors=[]
    for j,r in enumerate(rows):
        n=r["image"];rgb=r["rgb"];own=feedback[rgb]["feedback"];assert own==r["feedback"]
        donor=(donors[rgb] if cond=="shuffled" else rgb) if cond!="none" else None
        question=lock["query"] if donor is None else lock["query"]+"\n"+lock["label"]+feedback[donor]["feedback"]
        target=folder/(n+".pt");check=dict(image=n,rgb=rgb,query=question,evidence_rgb=donor,feedback=own,
                 gate=sha(ROOT/"inputs/gate.json"),adapter=expected["adapter"],label=cond)
        if target.exists():
            z=torch.load(target,weights_only=True,map_location="cpu")
            assert all(z[k]==v for k,v in check.items())
        else:
            im=pixels(tb,r);checked=PromptCheck(proc,question);orig=model.generate;actual=[]
            def observe(**kw):
                if not actual:
                    ids=kw["input_ids"].cpu();assert torch.equal(ids,checked.first_ids)
                    actual.append(digest(ids.contiguous().numpy().tobytes()))
                return orig(**kw)
            model.generate=observe
            try:h,actions,calls,_=greedy(model,checked,im,question,own)
            finally:model.generate=orig
            assert len(actual)==1 and checked.actual_query==question and all(v==own for v in calls)
            v=native(TRANS(im)[None].cuda(),h.float()[None].cuda())[0].cpu()
            z=dict(**check,tokens=actual[0],template=checked.template,actions=actions,calls=calls,prior=h.cpu(),v=v)
            savept(target,z)
        assert z["v"].shape==(768,) and z["prior"].shape==(3584,) and torch.isfinite(z["v"]).all()
        vectors.append(z["v"]);files[n]=sha(target)
        if (j+1)%100==0:print(tag,j+1,flush=True)
    fp=folder/"features.pt";savept(fp,dict(images=[r["image"] for r in rows],features=torch.stack(vectors)))
    dump(folder/"complete.json",dict(images=files,features=sha(fp),gate=sha(ROOT/"inputs/gate.json"),
         adapter=expected["adapter"],verified=sha(ROOT/"inputs/verified.json"),code=sha(Path(__file__))))
if __name__=="__main__":
    if sys.argv[1]=="verify":verify()
    else:generate(sys.argv[1])

"""One greedy prior per frozen image / caption with the standard native reader."""
from pathlib import Path
import gc,os,sys
import torch
ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT.parent
BASE=STAGE.parent
sys.path.insert(0,str(BASE/"dual_input_rfpo_20261126/code"))
from model_run import setup, read, dump, sha, savept, table, rgb, policy, reader, greedy, TRANS

@torch.inference_mode()
def run(seed,arm):
    variants={"dpo_r2":("joint_answer_pairs_20261127","r2"),"dpo_r3":("joint_answer_pairs_20261127","r3"),
              "rfpo_r2":("aftertool_joint_pool_20261127","r2"),"rfpo_strict":(STAGE.name,"strict")}
    variant,suffix=variants[arm]
    w=BASE/variant/"models"/f"{seed}_{suffix}"
    proof=read(w/"complete.json");digest=sha(w/"adapter/adapter_model.safetensors")
    assert digest==proof["final"]
    rows=read(ROOT/"inputs/rows.json"); manifest=read(ROOT/"inputs/manifest.json")
    assert len(rows)==164 and manifest["rows"]==sha(ROOT/"inputs/rows.json")
    out=ROOT/"records";out.mkdir(exist_ok=True)
    tag=f"{seed}_{arm}"
    if (out/(tag+".json")).is_file():
        assert sha(out/(tag+".pt"))==read(out/(tag+".json"))["sha256"]
        return
    setup();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    m,proc=policy(w);native=reader();tables={k:table(k) for k in {r["split"] for r in rows}}
    vectors=[];priors=[];traces=[]
    for r in rows:
        tbl=tables[r["split"]]; im=rgb(tbl,r)
        assert tbl["caption"][r["row"]].as_py()==r["captions"] and r["orig_query"]==r["captions"][0]
        h,actions,calls,_=greedy(m,proc,im,r["orig_query"],r["feedback"])
        v=native(TRANS(im)[None].cuda(),h.float()[None].cuda())[0].cpu()
        priors.append(h.cpu());vectors.append(v);traces.append(dict(image=r["image"],rgb=r["rgb_sha256"],actions=actions,calls=calls))
    file=out/(tag+".pt")
    savept(file,dict(visual=torch.stack(vectors),prior=torch.stack(priors),traces=traces,adapter=digest))
    dump(out/(tag+".json"),dict(sha256=sha(file),adapter=digest,inputs=manifest["rows"],images=len(rows),policy=sha(Path(__file__))))
    print(tag,len(rows),flush=True)


if __name__=="__main__":run(sys.argv[1],sys.argv[2])

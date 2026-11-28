"""Greedy image-only prediction; verify the actual user message before the processor renders it."""
import hashlib,inspect,io,os,sys
from pathlib import Path
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]
sys.path.insert(0,str(BASE/"dual_input_rfpo_20261126/code"))
from model_run import read,dump,sha,savept,setup,table,policy,reader,greedy,TRANS
from shared import ASSETS
def digest(data):return hashlib.sha256(data).hexdigest()
def weights(tag):
    return BASE/("visual_detail_rfpo_20261127" if tag.endswith("_strict") else "aftertool_joint_pool_20261127")/"models"/tag
def locked():
    lock=read(ROOT/"inputs/manifest.json")
    assert all(sha(ROOT/f)==h for f,h in lock["files"].items())
    rows=read(ROOT/"inputs/roster.json")
    assert all(set(r)=={"panel","image","row","rgb","split","feedback"} for r in rows)
    assert len(rows)==len({r["rgb"] for r in rows})==374
    return lock,rows
def pixels(tbl,r):
    assert tbl["path"][r["row"]].as_py()==r["image"]
    im=Image.open(io.BytesIO(tbl["image"][r["row"]].as_py())).convert("RGB")
    assert digest(im.tobytes())==r["rgb"]
    return im

@torch.inference_mode()
def verify():
    from prompt_clip import clip
    from tools import scene_classify_with_conf
    lock,rows=locked(); setup()
    head=ASSETS/"ucm_scene_cls.pt"; path=ASSETS/"models/ViT-L-14.pt"
    assert sha(head)==lock["scene_head"]
    m,pre=clip.load(str(path),device="cpu",jit=False);m=m.float().eval().cuda()
    ck=torch.load(head,map_location="cpu",weights_only=True);labels=list(ck["classes"])
    clf=torch.nn.Linear(ck["feat_dim"],len(labels));clf.load_state_dict(ck["head"]);clf=clf.eval().cuda()
    tables={sp:table(sp) for sp in {r["split"] for r in rows}}
    rec={}
    for r in rows:
        im=pixels(tables[r["split"]],r)
        result,conf=scene_classify_with_conf(im,model=m,preprocess=pre,cls_head=clf,cls_names=labels)
        assert result==r["feedback"],(r["image"],result,r["feedback"])
        rec[r["rgb"]]=dict(feedback=result,confidence=float(conf))
    dump(ROOT/"feedback_verified.json",dict(checks=rec,head=sha(head),clip=sha(path)))
    for tag,c in lock["policies"].items():
        f=weights(tag); assert sha(f/"adapter/adapter_model.safetensors")==c["adapter"] and sha(f/"complete.json")==c["proof"]
    trace=BASE/"source_transfer_2h_20261125/scratch/export"
    assert sha(trace/"cup_native_trace.pt")==read(trace/"proof.json")["trace"]
    env=dict(manifest=sha(ROOT/"inputs/manifest.json"),feedback=sha(ROOT/"feedback_verified.json"),reader=read(trace/"proof.json")["trace"],
        functions={f.__name__:sha(Path(inspect.getfile(f))) for f in (policy,reader,greedy,setup,scene_classify_with_conf)},
        arrows={s:sha(ASSETS/"ucm"/f"ucm_caption_karpathy_label_{s}.arrow") for s in tables})
    f=ROOT/"inputs/server_lock.json"
    if f.exists():assert read(f)==env
    else:dump(f,env)
    print("verified",len(rec),flush=True)

class PromptCheck:
    def __init__(self,processor,query):
        self.processor=processor; self.query=query; self.template=None; self.first_ids=None
    def __getattr__(self,key):return getattr(self.processor,key)
    def apply_chat_template(self,messages,*args,**kw):
        assert self.template is None
        assert messages[-1]=={"role":"user","content":[{"type":"image"},{"type":"text","text":"查询文本: "+self.query}]}
        self.actual_query=messages[-1]["content"][1]["text"].removeprefix("查询文本: ")
        text=self.processor.apply_chat_template(messages,*args,**kw)
        self.template=digest(text.encode())
        return text
    def __call__(self,*args,**kw):
        z=self.processor(*args,**kw)
        if self.first_ids is None:
            assert kw["images"] and len(kw["text"])==1 and digest(kw["text"][0].encode())==self.template
            self.first_ids=z["input_ids"].cpu().contiguous().clone()
        return z

@torch.inference_mode()
def generate(tag):
    lock,rows=locked(); proof=read(ROOT/"inputs/server_lock.json")
    assert sha(ROOT/"inputs/manifest.json")==proof["manifest"] and sha(ROOT/"feedback_verified.json")==proof["feedback"]
    verified=read(ROOT/"feedback_verified.json")["checks"]
    sig=lock["policies"][tag]["adapter"];src=weights(tag);assert sha(src/"adapter/adapter_model.safetensors")==sig
    setup();torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.set_num_threads(2)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"]==":4096:8"
    m,proc=policy(src);native=reader();tables={s:table(s) for s in {r["split"] for r in rows}}
    template=None
    for panel in ("official","extra"):
        output=ROOT/"records"/tag/panel;output.mkdir(parents=True,exist_ok=True)
        selected=[r for r in rows if r["panel"]==panel];vectors=[];files={}
        for j,r in enumerate(selected):
            name=r["image"];f=output/(name+".pt");key=dict(rgb=r["rgb"],image=name,query=lock["query"],feedback=r["feedback"],adapter=sig,lock=sha(ROOT/"inputs/server_lock.json"))
            assert verified[r["rgb"]]["feedback"]==r["feedback"]
            if f.exists():
                z=torch.load(f,map_location="cpu",weights_only=True);assert all(z[k]==v for k,v in key.items())
            else:
                im=pixels(tables[r["split"]],r);checked=PromptCheck(proc,lock["query"])
                gen=m.generate;actual=[]
                def observe(**kw):
                    if not actual:
                        tokens=kw["input_ids"].cpu()
                        assert checked.first_ids is not None and torch.equal(checked.first_ids,tokens)
                        actual.append(digest(tokens.contiguous().numpy().tobytes()))
                    return gen(**kw)
                m.generate=observe
                try:h,rounds,calls,_=greedy(m,checked,im,lock["query"],verified[r["rgb"]]["feedback"])
                finally:m.generate=gen
                assert actual and checked.template and checked.actual_query==key["query"]
                v=native(TRANS(im)[None].cuda(),h.float()[None].cuda())[0].cpu()
                z=dict(**key,actual_query=checked.actual_query,template=checked.template,first_input_ids_sha256=actual[0],
                       prior=h,v=v,actions=rounds,calls=calls)
                savept(f,z)
            if template is None:template=z["template"]
            assert z["template"]==template and z["actual_query"]==lock["query"] and all(v==r["feedback"] for v in z["calls"])
            assert z["v"].shape==(768,) and z["prior"].shape==(3584,) and torch.isfinite(z["v"]).all() and torch.isfinite(z["prior"]).all()
            files[name]=sha(f);vectors.append(z["v"])
            if (j+1)%50==0:print(tag,panel,j+1,flush=True)
        f=output/"features.pt"; savept(f,dict(adapter=sig,images=[r["image"] for r in selected],features=torch.stack(vectors)))
        dump(output/"complete.json",dict(adapter=sig,images=files,features=sha(f),template=template,query=lock["query"],lock=key["lock"],
               torch=str(torch.__version__),device=torch.cuda.get_device_name(),cuda=torch.version.cuda))
    print(tag,len(rows),flush=True)

if __name__=="__main__":
    if sys.argv[1]=="verify":verify()
    else:generate(sys.argv[1])

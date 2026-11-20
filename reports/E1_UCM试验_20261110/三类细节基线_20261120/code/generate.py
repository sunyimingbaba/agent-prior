"""Use the archived, original rollout entry for both fixed query policies."""
from common import ASSETS, ROOT, QUERY, read, dump, tables, image_for, digest
import contextlib
import io
import sys
import random
import numpy as np
import torch
from rollout import PLAIN_PROMPT, SYSTEM_PROMPT, load_mllm, rollout_one
from tools import scene_classify_with_conf
from prompt_clip import clip

method=sys.argv[1]
assert method in ("ASPF","AGSE")
out=ROOT/"priors"/method
out.mkdir(exist_ok=True,parents=True)
images=read(ROOT/"inputs"/(sys.argv[2] if len(sys.argv)>2 else "images.json"))
data=tables()
torch.set_num_threads(4)
model,processor=load_mllm(str(ASSETS/"models/Qwen2.5-VL-7B-Instruct"),"cuda",
                          str(ASSETS/"ucm_main_lora") if method=="AGSE" else "")
tools={}
events=[]
if method=="AGSE":
    tool_model,preprocess=clip.load(str(ASSETS/"models/ViT-L-14.pt"),device="cpu",jit=False)
    tool_model=tool_model.float().eval().cuda()
    cp=torch.load(ASSETS/"ucm_scene_cls.pt",map_location="cpu",weights_only=True)
    head=torch.nn.Linear(cp["feat_dim"],len(cp["classes"]))
    head.load_state_dict(cp["head"],strict=True)
    head=head.eval().cuda()
    def scene(im):
        t,c=scene_classify_with_conf(im,model=tool_model,preprocess=preprocess,cls_head=head,cls_names=list(cp["classes"]))
        events.append(dict(text=t,confidence=float(c)))
        return t if c>=0.0 else None
    tools["scene_classify"]=scene
for i,r in enumerate(images):
    stem=r["image"].split(".")[0];info=out/(stem+".json");tensor=out/(stem+".pt")
    if info.exists():
        old=read(info)
        assert old["input"]==r and old["query"]==QUERY and old["method"]==method and digest(tensor)==old["sha256"]
        continue
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    events.clear()
    trace=io.StringIO()
    with contextlib.redirect_stdout(trace), torch.inference_mode():
        h,used,filtered=rollout_one(model,processor,image_for(r,data),QUERY,tools,3,64,"cuda",
                                    verbose=True,injected_h_end=False,
                                    system_prompt=PLAIN_PROMPT if method=="ASPF" else SYSTEM_PROMPT,temperature=0.)
    assert list(h.shape)==[3584] and h.dtype==torch.bfloat16 and torch.isfinite(h).all()
    tmp=tensor.with_suffix(".part");torch.save(h,tmp);tmp.replace(tensor)
    dump(info,dict(method=method,input=r,query=QUERY,used=int(used),filtered=int(filtered),events=events,
                   trace=trace.getvalue(),sha256=digest(tensor)))
    if (i+1)%10==0:print(method,i+1,len(images),flush=True)
print("COMPLETE",method,len(images),flush=True)

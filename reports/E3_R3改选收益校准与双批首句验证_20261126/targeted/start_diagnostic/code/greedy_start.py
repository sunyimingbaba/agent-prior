"""Evaluate the two original adapters under the same deterministic first-caption protocol."""
from pathlib import Path
import os
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT.parent
sys.path.insert(0,str(TARGET/"code"))
from eval_single import setup, read, dump, sha, savept, table, rgb, policy, reader, greedy, datasets, TRANS
from runtime import initial


@torch.inference_mode()
def main(seed):
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and os.environ["PYTHONHASHSEED"] == "0"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    setup()
    torch.set_num_threads(2)
    source = initial(seed)
    sig = sha(source/"adapter/adapter_model.safetensors")
    assert sig == read(TARGET/"inputs/pilot/decision.json")["starts"][seed]
    for arm in ("ordinary", "old", "old3"):
        cp = read(TARGET/"models"/(seed+"_"+arm)/"complete.json")
        assert cp["initial"] == sig == read(TARGET/"inputs/pilot/decision.json")["starts"][seed]
    m,proc = policy(source)
    native = reader()
    rows, feedback, field = datasets()["official"]
    assert len(rows) == 210 and field == "orig_query"
    tables = {s:table(s) for s in {r["split"] for r in rows}}
    out = ROOT/"evaluate"/(seed+"_R0")
    out.mkdir(parents=True,exist_ok=True)
    protocol = sha(ROOT/"诊断说明.md")
    impl = sha(Path(__file__))
    vectors = []
    files = {}
    for r in rows:
        n=r["image"]; p=out/(n+".pt"); q=r[field]
        expected = dict(image=n, adapter=sig, query=q, rgb=r["rgb_sha256"], feedback=feedback[n], decoder=impl, protocol=protocol)
        if p.exists():
            z=torch.load(p,map_location="cpu",weights_only=True)
            assert all(z[k] == v for k,v in expected.items())
        else:
            im=rgb(tables[r["split"]],r)
            h,act,calls,_=greedy(m,proc,im,q,feedback[n])
            v=native(TRANS(im)[None].cuda(),h.float()[None].cuda())[0].cpu()
            z=dict(**expected,actions=act,calls=calls,prior=h,v=v)
            savept(p,z)
        vectors.append(z["v"])
        files[n]=sha(p)
    f=out/"features.pt"
    savept(f,dict(images=[r["image"] for r in rows],features=torch.stack(vectors),adapter=sig))
    dump(out/"complete.json",dict(adapter=sig,features=sha(f),images=files,decoder=impl,protocol=protocol,
        torch=str(torch.__version__),device=torch.cuda.get_device_name(0),formal_eval_code=sha(TARGET/"code/eval_single.py")))


if __name__ == "__main__":
    main(sys.argv[1])

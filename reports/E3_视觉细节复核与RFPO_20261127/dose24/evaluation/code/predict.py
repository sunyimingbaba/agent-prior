"""Generate one deterministic native first-caption prior per photo and verify external anchors."""
from pathlib import Path
import inspect
import os
import sys
import torch

EVAL = Path(__file__).resolve().parents[1]
STAGE = EVAL.parent
PREVIOUS = STAGE.parents[1] / "r3_switch_gain_20261126/targeted"
sys.path.insert(0, str(STAGE.parents[1] / "dual_input_rfpo_20261126/code"))
from model_run import setup, read, dump, sha, savept, table, rgb, policy, reader, greedy, datasets, weights, TRANS


def resolve(seed,arm):
    source=STAGE/"models"/f"{seed}_{arm}"
    cp=read(source/"complete.json")
    assert cp["steps"]==24 and cp["train"]==sha(STAGE/"inputs/roster.json")
    assert cp["lock"]==sha(STAGE/"inputs/train_lock.json")
    return source,cp["final"]


@torch.inference_mode()
def run(seed, arm):
    assert seed in ("a","b") and arm in ("strict","r2")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    setup()
    torch.set_num_threads(2)
    source, digest = resolve(seed, arm)
    assert sha(source/"adapter/adapter_model.safetensors") == digest
    model, proc = policy(source)
    native = reader()
    gallery, feedback, field = datasets()["official"]
    assert len(gallery)==210 and field=="orig_query"
    tables = {v:table(v) for v in {r["split"] for r in gallery}}
    decoder, protocol = sha(Path(__file__)), sha(EVAL/"验收口径.md")
    env = dict(torch=str(torch.__version__), device=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
               source_hashes={f.__name__:sha(Path(inspect.getfile(f))) for f in (policy,reader,greedy,datasets,rgb)})
    tag = seed+"_"+arm
    folder = EVAL/"records"/tag
    folder.mkdir(parents=True,exist_ok=True)
    rows=gallery
    signatures, vectors = {}, []
    for row in rows:
        name = row["image"]
        path = folder/(name+".pt")
        expected = dict(image=name,rgb=row["rgb_sha256"],query=row[field],feedback=feedback[name],
                        adapter=digest,protocol=protocol,decoder=decoder)
        if path.exists():
            record = torch.load(path,map_location="cpu",weights_only=True)
            assert all(record[k]==v for k,v in expected.items())
        else:
            im = rgb(tables[row["split"]],row)
            h, actions, calls, _ = greedy(model,proc,im,row[field],feedback[name])
            vector = native(TRANS(im)[None].cuda(),h.float()[None].cuda())[0].cpu()
            record = dict(**expected,prior=h,actions=actions,calls=calls,v=vector)
            savept(path,record)
        signatures[name] = sha(path)
        vectors.append(record["v"])
    target = folder/"features.pt"
    savept(target,dict(adapter=digest,images=[r["image"] for r in rows],features=torch.stack(vectors)))
    dump(folder/"complete.json",dict(**env,decoder=decoder,protocol=protocol,adapter=digest,features=sha(target),
                                     images=signatures,reference=None))
    print(seed,arm,len(rows),flush=True)


if __name__=="__main__":
    run(*sys.argv[1:])

"""Generate one deterministic native first-caption prior per photo and verify external anchors."""
from pathlib import Path
import inspect
import os
import sys
import torch

EVAL = Path(__file__).resolve().parents[1]
STAGE = EVAL.parent
PREVIOUS = STAGE.parent / "r3_switch_gain_20261126/targeted"
sys.path.insert(0, str(STAGE.parent / "dual_input_rfpo_20261126/code"))
from model_run import setup, read, dump, sha, savept, table, rgb, policy, reader, greedy, datasets, weights, TRANS


def resolve(seed, arm):
    if arm in ("r2","r3"):
        source = STAGE/"models"/f"{seed}_{arm}"
        cp = read(source/"complete.json")
        roster = STAGE/"scoring/locks/training.json"
        assert cp["steps"] == len(read(roster)["rows"]) and 2 <= cp["steps"] <= 24
        assert cp["train"] == sha(roster) and cp["lock"] == sha(STAGE/"training/train_lock.json")
        return source, cp["final"], None
    ref_dir = PREVIOUS if arm=="retained" else PREVIOUS/"start_diagnostic"
    name = seed + ("_retained" if arm=="retained" else "_R0")
    source, digest = weights(seed + ("_deployed" if arm=="retained" else "_R0"))
    return source, digest, ref_dir/"evaluate"/name


@torch.inference_mode()
def run(seed, arm):
    assert seed in ("a","b") and arm in ("r2","r3","retained","R0")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    setup()
    torch.set_num_threads(2)
    source, digest, reference = resolve(seed, arm)
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
    reference_proof = read(reference/"complete.json") if reference else None
    if reference:
        assert all(reference_proof[k]==env[k] for k in ("torch","device"))
        assert reference_proof["adapter"]==digest
        rows = [gallery[i] for i in (0,105,209)]
    else:
        rows = gallery
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
        if reference:
            past = reference/path.name
            assert sha(past)==reference_proof["images"][name]
            earlier = torch.load(past,map_location="cpu",weights_only=True)
            assert all(earlier[k]==record[k] for k in ("image","rgb","query","feedback","adapter","actions","calls"))
            assert torch.equal(earlier["prior"],record["prior"]) and torch.equal(earlier["v"],record["v"])
        signatures[name] = sha(path)
        vectors.append(record["v"])
    target = folder/"features.pt"
    savept(target,dict(adapter=digest,images=[r["image"] for r in rows],features=torch.stack(vectors)))
    dump(folder/"complete.json",dict(**env,decoder=decoder,protocol=protocol,adapter=digest,features=sha(target),
                                     images=signatures,reference=sha(reference/"complete.json") if reference else None))
    print(seed,arm,len(rows),flush=True)


if __name__=="__main__":
    run(*sys.argv[1:])

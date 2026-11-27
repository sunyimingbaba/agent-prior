"""One deterministic greedy first-caption observation per UCM gallery image."""
from pathlib import Path
import os
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent / "dual_input_rfpo_20261126/code"))
from model_run import setup, read, dump, sha, savept, table, rgb, policy, reader, greedy, datasets, weights, TRANS


@torch.inference_mode()
def evaluate(seed, arm):
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    setup()
    torch.set_num_threads(2)
    if arm == "retained":
        source, signature = weights(seed + "_deployed")
    else:
        source = ROOT / "models" / (seed + "_" + arm)
        signature = sha(source / "adapter/adapter_model.safetensors")
        assert signature == read(source / "complete.json")["final"]
    model, processor = policy(source)
    native = reader()
    rows, feedback, field = datasets()["official"]
    assert field == "orig_query" and len(rows) == 210
    tables = {split:table(split) for split in {row["split"] for row in rows}}
    folder = ROOT / "evaluate" / (seed + "_" + arm)
    folder.mkdir(parents=True, exist_ok=True)
    protocol = sha(ROOT / "单次推理验收.md")
    code = sha(Path(__file__))
    vectors, signatures = [], {}
    for row in rows:
        name, query = row["image"], row[field]
        path = folder / (name + ".pt")
        expected = dict(adapter=signature, rgb=row["rgb_sha256"], query=query, feedback=feedback[name], protocol=protocol, decoder=code)
        if path.exists():
            record = torch.load(path, weights_only=True, map_location="cpu")
            assert all(record[k] == v for k,v in expected.items())
        else:
            image = rgb(tables[row["split"]], row)
            prior, actions, calls, _ = greedy(model, processor, image, query, feedback[name])
            vector = native(TRANS(image)[None].cuda(),prior.float()[None].cuda())[0].cpu()
            record = dict(**expected, image=name, prior=prior, v=vector, actions=actions, calls=calls)
            savept(path, record)
        vectors.append(record["v"])
        signatures[name] = sha(path)
    bundle = folder / "features.pt"
    savept(bundle, dict(images=[row["image"] for row in rows], features=torch.stack(vectors), adapter=signature))
    dump(folder / "complete.json", dict(adapter=signature, features=sha(bundle), images=signatures,
         protocol=protocol, decoder=code, torch=str(torch.__version__), device=torch.cuda.get_device_name(0)))
    print(seed,arm,len(rows),flush=True)


if __name__ == "__main__":
    evaluate(*sys.argv[1:])

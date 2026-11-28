"""Generate one prior per held-out image using the final, matched dose24 policies."""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
DOSE = ROOT.parent
BASE = DOSE.parents[1]
sys.path.insert(0, str(BASE/"dual_input_rfpo_20261126/code"))
from model_run import setup, read, dump, sha, savept, table, rgb, policy, reader, greedy, TRANS


@torch.inference_mode()
def run(seed, arm):
    assert seed in ("a", "b") and arm in ("r2", "strict")
    model_dir = DOSE/"models"/f"{seed}_{arm}"
    proof = read(model_dir/"complete.json")
    adapter = sha(model_dir/"adapter/adapter_model.safetensors")
    assert adapter == proof["final"] and proof["steps"] == 24
    manifest = read(ROOT/"inputs/manifest.json")
    assert sha(ROOT/"inputs/rows.json") == manifest["rows"]
    assert sha(ROOT/"事前口径.md") == manifest["protocol"]
    rows = read(ROOT/"inputs/rows.json")
    output = ROOT/"records"/f"{seed}_{arm}.pt"
    output.parent.mkdir(exist_ok=True)
    info = dict(inputs=manifest["rows"], adapter=adapter, policy=sha(Path(__file__)),
                protocol=manifest["protocol"], images=len(rows))
    if output.with_suffix(".json").exists():
        assert read(output.with_suffix(".json")) == dict(**info, sha256=sha(output))
        return
    setup()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model, processor = policy(model_dir)
    native = reader()
    tables = {k: table(k) for k in {r["split"] for r in rows}}
    features, priors, traces = [], [], []
    for row in rows:
        samples = tables[row["split"]]
        image = rgb(samples, row)
        assert samples["caption"][row["row"]].as_py() == row["captions"]
        assert row["orig_query"] == row["captions"][0]
        hidden, actions, calls, _ = greedy(model, processor, image, row["orig_query"], row["feedback"])
        features.append(native(TRANS(image)[None].cuda(), hidden.float()[None].cuda())[0].cpu())
        priors.append(hidden.cpu())
        traces.append(dict(image=row["image"], rgb=row["rgb_sha256"], actions=actions, calls=calls))
    savept(output, dict(visual=torch.stack(features), prior=torch.stack(priors), traces=traces, adapter=adapter))
    dump(output.with_suffix(".json"), dict(**info, sha256=sha(output)))
    print(seed, arm, len(rows), flush=True)


if __name__ == "__main__":
    run(*sys.argv[1:])

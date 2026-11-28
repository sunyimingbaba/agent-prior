"""Add a verified real or permuted scene hint before the same standard single-prior rollout."""
from pathlib import Path
import os
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent / "image_only"
sys.path.insert(0, str(BASELINE/"code"))
from predict import read, dump, sha, weights, setup, policy, reader, table, pixels, PromptCheck, greedy, digest, TRANS, savept


@torch.inference_mode()
def run(tag, condition):
    assert condition in ("real", "shuffled") and tag in ("a_r2","b_r2","a_strict","b_strict")
    assert condition == "real" or tag.endswith("strict")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    manifest = read(ROOT/"inputs/manifest.json")
    mapping = read(ROOT/"inputs/assignments.json")
    assert sha(ROOT/"inputs/assignments.json") == manifest["assignment"]
    assert sha(ROOT/"检验口径.md") == manifest["protocol"]
    assert sha(BASELINE/"inputs/manifest.json") == manifest["source"]
    assert sha(BASELINE/"inputs/roster.json") == manifest["roster"]
    assert sha(BASELINE/"feedback_verified.json") == manifest["verified_tool"]
    verified = read(BASELINE/"feedback_verified.json")["checks"]
    rows = read(BASELINE/"inputs/roster.json")
    source = weights(tag)
    sig = sha(source/"adapter/adapter_model.safetensors")
    assert sig == manifest["policies"][tag]["adapter"] and sha(source/"complete.json") == manifest["policies"][tag]["proof"]
    setup()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model, proc = policy(source)
    native = reader()
    tables = {s: table(s) for s in {r["split"] for r in rows}}
    out = ROOT/"records"/(tag+"_"+condition)
    out.mkdir(parents=True, exist_ok=True)
    evidence_proof = dict(manifest=sha(ROOT/"inputs/manifest.json"), model=sig, generation=sha(Path(__file__)),
                          common=sha(BASELINE/"code/predict.py"))
    for panel in ("official","extra"):
        path = out/(panel+".pt")
        if path.with_suffix(".json").exists():
            assert read(path.with_suffix(".json")) == dict(**evidence_proof, sha256=sha(path))
            continue
        priors, vectors, traces = [], [], []
        for r in [r for r in rows if r["panel"] == panel]:
            source_rgb = r["rgb"] if condition == "real" else mapping[r["rgb"]]
            hint = verified[source_rgb]["feedback"]
            question = manifest["query"] + "\n" + manifest["label"] + hint
            own = verified[r["rgb"]]["feedback"]
            assert own == r["feedback"]
            image = pixels(tables[r["split"]], r)
            checked = PromptCheck(proc, question)
            generate = model.generate
            token_hashes = []
            def observe(**kw):
                if not token_hashes:
                    ids = kw["input_ids"].cpu()
                    assert torch.equal(checked.first_ids, ids)
                    token_hashes.append(digest(ids.contiguous().numpy().tobytes()))
                return generate(**kw)
            model.generate = observe
            try:
                hidden, actions, calls, _ = greedy(model, checked, image, question, own)
            finally:
                model.generate = generate
            assert checked.actual_query == question and len(token_hashes) == 1 and all(f == own for f in calls)
            features = native(TRANS(image)[None].cuda(), hidden.float()[None].cuda())[0].cpu()
            priors.append(hidden); vectors.append(features)
            traces.append(dict(name=r["image"], rgb=r["rgb"], evidence_rgb=source_rgb, query=checked.actual_query,
                               tokens=token_hashes[0], template=checked.template, actions=actions, calls=calls))
        savept(path, dict(prior=torch.stack(priors), visual=torch.stack(vectors), traces=traces, adapter=sig))
        dump(path.with_suffix(".json"), dict(**evidence_proof, sha256=sha(path)))
        print(tag,condition,panel,len(traces),flush=True)


if __name__ == "__main__":
    run(*sys.argv[1:])

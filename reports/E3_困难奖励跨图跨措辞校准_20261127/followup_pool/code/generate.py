"""Frozen cold-start second-round actions on the original first caption for each image."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[1]
SEED = sys.argv[1]
assert SEED in ("d", "feedback")
SOURCE = BASE / "second_answer_5090_20261124"
sys.path.insert(0, str(SOURCE / "code"))
from shared import ASSETS, dump, read, rgb, setup, sha, table
from policy import policy
from train import finish, sample
from rollout import SYSTEM_PROMPT, parse_tool_call, rollout_one


def signature_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def verify_roster(rows):
    signature = sha(ROOT / "inputs/roster.json")
    assert signature == read(ROOT / "inputs/roster_lock.json")["roster"]
    assert len(rows) == len({r["image"] for r in rows}) == len({r["rgb_sha256"] for r in rows}) == read(ROOT/"inputs/roster_lock.json")["images"]
    assert all(r["split"] == "train" and r["captions"][0] == r["orig_query"] for r in rows)
    return signature


@torch.inference_mode()
def verify_feedback(rows, tb, signature):
    from prompt_clip import clip
    from tools import scene_classify_with_conf

    model_path = ASSETS / "models/ViT-L-14.pt"
    head_path = ASSETS / "ucm_scene_cls.pt"
    tool, preprocess = clip.load(str(model_path), device="cpu", jit=False)
    tool = tool.float().eval().cuda()
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
    labels = list(checkpoint["classes"])
    head = torch.nn.Linear(checkpoint["feat_dim"], len(labels))
    head.load_state_dict(checkpoint["head"])
    head = head.eval().cuda()
    verified = {}
    for row in rows:
        image = rgb(tb, row)
        assert tb["caption"][row["row"]].as_py() == row["captions"]
        answer, confidence = scene_classify_with_conf(image, model=tool, preprocess=preprocess,
                                                     cls_head=head, cls_names=labels)
        assert answer == row["feedback"], (row["image"], answer, row["feedback"])
        verified[row["image"]] = dict(rgb=row["rgb_sha256"], feedback=answer, confidence=float(confidence))
    dump(ROOT / "feedback_verified.json", dict(roster=signature, classifier=sha(head_path),
         clip=sha(model_path), images=verified, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__)))


class CapturedRound(Exception):
    pass


def natural_context(model, processor, image, row):
    original = model.generate
    first = []
    prefix = None
    returned = []

    def feedback(tool_image):
        assert tool_image is image and not returned
        returned.append(row["feedback"])
        return returned[0]

    def capture(**kw):
        nonlocal prefix
        if first:
            assert returned == [row["feedback"]]
            prefix = kw["input_ids"].cpu().clone()
            raise CapturedRound
        out = original(**kw)
        first.extend(out.sequences[0, kw["input_ids"].shape[1]:].tolist())
        return out

    model.generate = capture
    try:
        rollout_one(model, processor, image, row["orig_query"], {"scene_classify": feedback},
                    3, 64, "cuda", temperature=0., system_prompt=SYSTEM_PROMPT)
    except CapturedRound:
        pass
    finally:
        model.generate = original
    assert first and (prefix is not None) == bool(returned)
    text = processor.decode(first, skip_special_tokens=True)
    assert (prefix is not None) == (parse_tool_call(text) == "scene_classify")
    if prefix is not None:
        suffix = processor(text=[text + f'\n<result>{row["feedback"]}</result>\n'],
                           return_tensors="pt").input_ids
        assert torch.equal(prefix[:, -suffix.shape[1]:], suffix)
    return first, text, prefix


def validate(record, row, rng, adapter, lock):
    assert (record["image"], record["row"], record["seed"], record["rng"], record["adapter"],
            record["rgb"], record["query"], record["feedback"], record["lock"]) == (
        row["image"], row["row"], SEED, rng, adapter, row["rgb_sha256"], row["orig_query"], row["feedback"], lock)
    assert record["first"] and record["eligible"] == (record["context"] is not None)
    if record["eligible"]:
        assert record["hidden"].shape == (8, 3584) and torch.isfinite(record["hidden"]).all()
        assert len(record["paths"]) == len(record["extras"]) == len(record["texts"]) == len(record["logps"]) == 8
        assert all(len(path) == len(logp) > 0 for path, logp in zip(record["paths"], record["logps"]))
        assert record["context"]["first"] == record["first"] and record["context"]["answer"] == row["feedback"]
        assert record["context"]["prefix"].dtype == torch.int64 and record["context"]["prefix"].shape[0] == 1
    else:
        assert record["hidden"] is None and record["paths"] == []


@torch.inference_mode()
def run():
    setup()
    torch.set_num_threads(1)
    rows = read(ROOT / "inputs/roster.json")
    signature = verify_roster(rows)
    tb = table("train")
    if SEED == "feedback":
        return verify_feedback(rows, tb, signature)
    feedback_path = ROOT / "feedback_verified.json"
    checked = read(feedback_path)
    assert checked["roster"] == signature and len(checked["images"]) == len(rows)
    adapter = sha(SOURCE / "R0/adapter/adapter_model.safetensors")
    assert adapter == read(SOURCE / "initial.json")["adapter"]
    assert adapter == read(BASE / "r3_switch_gain_20261126/inputs/pilot/decision.json")["starts"]["a"]
    dependencies = [SOURCE / "code" / name for name in ("policy.py", "train.py", "shared.py")]
    import rollout
    dependencies += [Path(rollout.__file__), Path(__file__), feedback_path]
    lock = dict(roster=signature, dependencies={str(p): sha(p) for p in dependencies})
    directory = ROOT / "raw" / SEED
    directory.mkdir(parents=True, exist_ok=True)
    model, processor = policy("R0")
    files = {}
    eligible = {}
    for row in rows:
        name = row["image"]
        image = rgb(tb, row)
        assert tb["caption"][row["row"]].as_py() == row["captions"]
        assert (checked["images"][name]["rgb"], checked["images"][name]["feedback"]) == (row["rgb_sha256"], row["feedback"])
        rng = 261000000 + row["row"] * 1000
        path = directory / (name + ".pt")
        if path.exists():
            record = torch.load(path, map_location="cpu", weights_only=True)
        else:
            first, first_text, prefix = natural_context(model, processor, image, row)
            record = dict(image=name, row=row["row"], seed=SEED, rng=rng, adapter=adapter, lock=lock,
                          rgb=row["rgb_sha256"], query=row["orig_query"], feedback=row["feedback"],
                          captions_sha256=signature_text(json.dumps(row["captions"], ensure_ascii=False)),
                          first=first, first_text=first_text, eligible=prefix is not None,
                          context=None, hidden=None, paths=[], extras=[], texts=[], logps=[], extra_texts=[])
            if prefix is not None:
                context = dict(first=first, prefix=prefix, answer=row["feedback"])
                torch.manual_seed(rng)
                single, traces = sample(model, processor, image, row["orig_query"], context)
                states, extra = finish(model, processor, single, traces, row["feedback"])
                record.update(context=context, hidden=torch.stack(states), paths=[t[0] for t in traces],
                              texts=[t[4] for t in traces], logps=[t[2] for t in traces], extras=extra,
                              extra_texts=[processor.decode(ids, skip_special_tokens=True) for ids in extra])
                del single, traces, states
            validate(record, row, rng, adapter, lock)
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as out:
                torch.save(record, out)
                tmp = Path(out.name)
            tmp.replace(path)
        validate(record, row, rng, adapter, lock)
        files[name] = sha(path)
        eligible[name] = record["eligible"]
        print(SEED, len(files), name, record["eligible"], flush=True)
    assert set(directory.glob("*.pt")) == {directory / (name + ".pt") for name in files}
    dump(ROOT / (SEED + "_eligibility.json"), eligible)
    dump(ROOT / (SEED + "_complete.json"), dict(seed=SEED, adapter=adapter, lock=lock, files=files,
         eligibility=eligible, rows=len(files), reached=sum(eligible.values()), gpu=torch.cuda.get_device_name(),
         torch=str(torch.__version__), cuda=torch.version.cuda))


if __name__ == "__main__":
    run()

"""Force the prescribed tokens with the same eight-row cached decoder as the original sampler."""
from pathlib import Path
import sys

import torch
from transformers import LogitsProcessor, LogitsProcessorList

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parent
BASE = STAGE.parent
FRESH = STAGE / "fresh98"
sys.path.insert(0, str(BASE / "second_answer_5090_20261124/code"))
from shared import dump, read, rgb, setup, sha, table
from policy import policy, prompt, cut


class FixedTokens(LogitsProcessor):
    def __init__(self, paths, prefix, eos):
        self.paths = paths
        self.prefix = prefix
        self.eos = eos

    def __call__(self, input_ids, scores):
        idx = input_ids.shape[1] - self.prefix
        forced = [p[idx] if idx < len(p) else self.eos for p in self.paths]
        scores.fill_(float("-inf"))
        scores.scatter_(1, torch.tensor(forced, device=scores.device)[:, None], 0.)
        return scores


def prepared(proc, im, row, raw):
    inp = proc(text=[prompt(proc, row["orig_query"])], images=[im], return_tensors="pt").to("cuda")
    prefix = raw["context"]["prefix"].cuda()
    assert torch.equal(prefix[:, :inp.input_ids.shape[1]], inp.input_ids)
    inp["input_ids"] = prefix
    inp["attention_mask"] = torch.ones_like(prefix)
    return {k:v.repeat(8, *([1] * (v.ndim - 1))) for k, v in inp.items()}


def force_model(model, inp, paths, eos):
    n = inp["input_ids"].shape[1]
    assert len(paths) == 8 and all(0 < len(ids) <= 64 for ids in paths)
    result = model.generate(**inp, max_new_tokens=64, do_sample=True, temperature=1.,
                            logits_processor=LogitsProcessorList([FixedTokens(paths, n, eos)]),
                            output_scores=True, output_hidden_states=True, return_dict_in_generate=True)
    states = []
    for i, ids in enumerate(paths):
        actual = cut(result.sequences[i, n:].tolist(), model.generation_config.eos_token_id)
        assert actual == ids, (i, ids, actual)
        states.append(result.hidden_states[len(ids)-1][-1][i, -1].detach().cpu().clone())
    return torch.stack(states)


def savept(path, obj):
    tmp = path.with_suffix(".pt.part")
    torch.save(obj, tmp)
    tmp.replace(path)


@torch.inference_mode()
def main():
    setup()
    torch.set_num_threads(1)
    proof = read(FRESH / "c_complete.json")
    rows = read(FRESH / "inputs/rows.json")
    assert len(rows) == proof["rows"] == proof["reached"] == 98
    assert proof["adapter"] == sha(BASE / "second_answer_5090_20261124/R0/adapter/adapter_model.safetensors")
    model, proc = policy("R0")
    eos = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert eos == 151645 and eos in model.generation_config.eos_token_id
    tbl = table("train")
    out = ROOT / "raw"
    out.mkdir(exist_ok=True)
    calibration = {}
    probes = {}
    counts = {True:0, False:0}
    for row in sorted(rows, key=lambda r:r["rgb_sha256"]):
        name = row["image"]
        path = FRESH / "raw/c" / (name + ".pt")
        assert sha(path) == proof["files"][name]
        raw = torch.load(path, weights_only=True, map_location="cpu")
        eligible = [i for i in range(8) if raw["paths"][i][-1] == eos and not raw["extras"][i]]
        counts[True] += sum(x[-1] == eos for x in raw["paths"])
        counts[False] += sum(x[-1] != eos for x in raw["paths"])
        if len(calibration) < 4 and eligible:
            i = eligible[0]
            inp = prepared(proc, rgb(tbl, row), row, raw)
            torch.manual_seed(raw["rng"])
            repro = force_model(model, inp, raw["paths"], eos)
            sel = repro[i]
            d = sel.float() - raw["hidden"][i].float()
            comp = dict(exact=bool(torch.equal(sel, raw["hidden"][i])), max_abs=float(d.abs().max()),
                        mean_abs=float(d.abs().mean()), l2=float(d.norm()), slot=i, original_sha=sha(path),
                        block_check_slots=len(eligible),
                        exact_for_all_eligible=all(torch.equal(repro[j], raw["hidden"][j]) for j in eligible))
            calibration[name] = comp
            probes[name] = dict(slot=i, replay=sel, stored=raw["hidden"][i], eligible_slots=eligible,
                                replays=repro[eligible], stored_all=raw["hidden"][eligible])
    assert len(calibration) == 4
    savept(ROOT / "calibration.pt", probes)
    dump(ROOT / "calibration.json", dict(checks=calibration, eos=eos, natural_ended=counts[True],
         capped_without_eos=counts[False], archived_source=sha(FRESH / "c_complete.json"),
         probe_tensors=sha(ROOT / "calibration.pt"), exact_all=all(d["exact_for_all_eligible"] for d in calibration.values())))
    # Force exactly one true and one false text, replicated within the same batch shape as the source.
    files = {}
    for row in rows:
        name = row["image"]
        path = FRESH / "raw/c" / (name + ".pt")
        dest = out / (name + ".pt")
        if not dest.exists():
            raw = torch.load(path, weights_only=True, map_location="cpu")
            assert raw["seed"] == "c" and raw["rgb"] == row["rgb_sha256"]
            im = rgb(tbl, row)
            inp = prepared(proc, im, row, raw)
            texts = ["<answer>" + q + "</answer>" for q in row["A"]]
            ids = [proc.tokenizer.encode(t, add_special_tokens=False) + [eos] for t in texts]
            assert all(proc.tokenizer.decode(p[:-1], skip_special_tokens=False) == t for p, t in zip(ids, texts))
            assert ids[0] != ids[1] and all(not set(p[:-1]) & set(model.generation_config.eos_token_id) for p in ids)
            torch.manual_seed(raw["rng"])
            h = force_model(model, inp, ids * 4, eos)
            assert all(torch.equal(h[k], h[k % 2]) for k in range(8))
            savept(dest, dict(image=name, row=row["row"], pair=row["pair"], rgb=row["rgb_sha256"], source=sha(path),
                              states=h[:2].clone(), texts=texts, ids=ids, eos=eos,
                              context_prefix=raw["context"]["prefix"], adapter=proof["adapter"]))
        z = torch.load(dest, weights_only=True, map_location="cpu")
        assert z["source"] == sha(path) and z["rgb"] == row["rgb_sha256"] and z["states"].shape == (2,3584)
        files[name] = sha(dest)
        print(len(files), name, flush=True)
    dump(ROOT / "generation.json", dict(files=files, model=proof["adapter"], code=sha(Path(__file__)),
         calibration=sha(ROOT/"calibration.json"), gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
         fresh=sha(FRESH/"c_complete.json"), protocol=sha(ROOT/"口径.md")))


if __name__ == "__main__":
    main()

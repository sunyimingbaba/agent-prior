"""Use the original four-thread training reader, then verify every saved candidate."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REBUILD = ROOT.parents[1]
sys.path.insert(0, str(REBUILD / "aftertool_joint_pool_20261127/scoring/code"))
import pool_reader as original

original.SCORING = ROOT
read, load, save, sha = original.read, original.load, original.save, original.sha


@torch.inference_mode()
def main():
    assert str(original.ORIGINAL) == str(REBUILD / "second_answer_5090_20261124/code")
    model = original.Reader()
    assert torch.get_num_threads() == 4
    bank = load(REBUILD / "relation_candidates_20261120/native/features.pt")
    lookup = read(REBUILD / "relation_candidates_20261120/native/index.json")["columns"]
    fits = torch.tensor([int(hashlib.sha256(",".join(map(str, q)).encode()).hexdigest()[-1], 16) % 2 == 0 for q in bank["tokens"].tolist()])
    words = bank["features"].double()
    proj_dir = ROOT / "readouts"
    proj_dir.mkdir(exist_ok=True)
    score_dir = ROOT / "scores"
    score_dir.mkdir(exist_ok=True)
    signatures = {}
    projected = {}
    for name, row in model.rows.items():
        path = ROOT / "raw/d" / (name + ".pt")
        while not path.is_file():
            time.sleep(2)
        raw = load(path)
        output = score_dir / (name + ".json")
        features = proj_dir / (name + ".pt")
        if not (output.exists() and features.exists()):
            assert raw["row"] == row["row"] and raw["rgb"] == row["rgb_sha256"] and raw["query"] == row["orig_query"]
            assert raw["image"] == name and raw["feedback"] == row["feedback"] and raw["seed"] == "d"
            data = dict(image=name, rgb=raw["rgb"], eligible=raw["eligible"], source=sha(path), seed="d",
                        pair=row["pair"], cohort="independent282", alt_ids=row["alt_ids"])
            states = raw["hidden"]
            capture = {}
            if raw["eligible"]:
                original_img = model.native.img_embeds
                def track(*args, **kwargs):
                    result = original_img(*args, **kwargs)
                    key = kwargs["agent_hidden"].float().contiguous().numpy().tobytes()
                    capture[key] = result["image_feats"][0].detach().cpu().clone()
                    return result
                model.native.img_embeds = track
                try:
                    scores = model.score(name, states)
                finally:
                    model.native.img_embeds = original_img
                data.update(scores)
                visual = torch.stack([capture[h.float().contiguous().numpy().tobytes()] for h in states])
                pos = [lookup[q] for q in row["captions"]]
                alts = sorted(set(pos) - {pos[0]})
                assert alts == row["alt_ids"] and pos == row["native_ids"]
                fit, test = fits.clone(), ~fits
                fit[pos] = test[pos] = False
                cache = {}
                results = []
                for state, v in zip(states.float().contiguous(), visual.double()):
                    state_key = state.numpy().tobytes()
                    if state_key not in cache:
                        rank = int(model.proxy.score(state.tolist(), row["row"])[0])
                        sims = words @ v
                        margin = float(sims[pos[0]] - sims[fit].max())
                        h = model.hard[name].double() @ v
                        signed = (h[::2] - h[1::2]).tolist()
                        negative = sims[test]
                        rr = []
                        for j in alts:
                            ahead = int((negative > sims[j]).sum())
                            tied = int((negative == sims[j]).sum())
                            rr.append(float(np.mean(1 / np.arange(ahead + 1, ahead + tied + 2))))
                        cache[state_key] = dict(rank=rank, margin=margin, hard_margins=signed[:2],
                                                DE=signed[2:4], FG=signed[4:] or None, other_caption_rr=rr)
                    results.append(cache[state_key])
                assert results == data["items"], name
                assert all(data[k] == v for k, v in original.grade(results, model.match, model.bonus).items())
                data["all_readouts_verified"] = True
            else:
                visual = None
                data["all_readouts_verified"] = None
            tmp = features.with_suffix(".pt.part")
            torch.save(dict(image=name, rgb=raw["rgb"], source=sha(path), visual=visual), tmp)
            tmp.replace(features)
            data["projections"] = sha(features)
            save(output, data)
        data = read(output)
        assert data["source"] == sha(path) and data["projections"] == sha(features)
        signatures[name] = sha(output)
        projected[name] = sha(features)
        print(len(signatures), name, data["eligible"], flush=True)
    while not (ROOT / "d_complete.json").exists():
        time.sleep(1)
    sample_proof = read(ROOT / "d_complete.json")
    assert set(signatures) == set(sample_proof["files"])
    for name in signatures:
        assert read(score_dir / (name + ".json"))["source"] == sample_proof["files"][name]
    save(ROOT / "scoring_complete.json", dict(files=signatures, readouts=projected, threads=torch.get_num_threads(),
         reached=sample_proof["reached"], verified_slots=8*sample_proof["reached"], roster=sha(ROOT/"inputs/roster.json"),
         sampling=sha(ROOT/"d_complete.json"), original_reader=sha(Path(original.__file__)), code=sha(Path(__file__)),
         compiled=sha(ROOT/"inputs/manifest.json")))


if __name__ == "__main__":
    main()

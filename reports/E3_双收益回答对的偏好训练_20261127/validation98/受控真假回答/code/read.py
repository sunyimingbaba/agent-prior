"""Project the forced-response and calibration states using the original four-thread reader."""
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parent
BASE = STAGE.parent
FRESH = STAGE / "fresh98"
sys.path.insert(0, str(BASE / "aftertool_joint_pool_20261127/scoring/code"))
import pool_reader as original

original.SCORING = FRESH
read, save, load, sha = original.read, original.save, original.load, original.sha


def wait(path):
    while not path.exists():
        time.sleep(3)


def repeated(reader, name, states):
    n = len(states)
    projected = {}
    fn = reader.native.img_embeds
    def track(*args, **kwargs):
        obj = fn(*args, **kwargs)
        key = kwargs["agent_hidden"].float().contiguous().numpy().tobytes()
        projected[key] = obj["image_feats"][0].detach().cpu().clone()
        return obj
    reader.native.img_embeds = track
    try:
        slots = torch.stack([states[j % n] for j in range(8)])
        items = reader.score(name, slots)["items"]
    finally:
        reader.native.img_embeds = fn
    assert all(items[j] == items[j % n] for j in range(8))
    vectors = torch.stack([projected[h.float().contiguous().numpy().tobytes()] for h in states])
    return items[:n], vectors


def capture(path, obj):
    tmp = path.with_suffix(".pt.part")
    torch.save(obj, tmp)
    tmp.replace(path)


@torch.inference_mode()
def main():
    wait(ROOT / "calibration.json")
    reader = original.Reader()
    assert torch.get_num_threads() == 4
    logger = read(ROOT / "calibration.json")
    probes = load(ROOT / "calibration.pt")
    assert sha(ROOT / "calibration.pt") == logger["probe_tensors"]
    calibrated = {}
    scorer_receipt = read(FRESH / "scoring_complete.json")
    for name, probe in probes.items():
        path = FRESH / "scores" / (name + ".json")
        assert sha(path) == scorer_receipt["files"][name]
        stored = read(path)["items"]
        slots = probe["eligible_slots"]
        items, _ = repeated(reader, name, probe["replays"])
        records = []
        for i, current in zip(slots, items):
            prior = stored[i]
            def differences(k):
                return [float(a) - float(b) for a, b in zip(current[k], prior[k])]
            records.append(dict(slot=i, rank_same=current["rank"] == prior["rank"],
                                rr_same=current["other_caption_rr"] == prior["other_caption_rr"],
                                ab_signs_equal=np.array_equal(np.array(current["hard_margins"]) > 0, np.array(prior["hard_margins"]) > 0),
                                de_signs_equal=np.array_equal(np.array(current["DE"]) > 0, np.array(prior["DE"]) > 0),
                                fg_signs_equal=np.array_equal(np.array(current["FG"]) > 0, np.array(prior["FG"]) > 0),
                                max_hard_delta=max(abs(v) for k in ("hard_margins", "DE", "FG") for v in differences(k)),
                                margin_delta=current["margin"] - prior["margin"], all_equal=current == prior))
        calibrated[name] = records
    success = all(r["rank_same"] and r["rr_same"] and r["ab_signs_equal"] and r["de_signs_equal"] and r["fg_signs_equal"]
                  for records in calibrated.values() for r in records)
    save(ROOT / "calibration_readouts.json", dict(images=calibrated, checked=sum(map(len,calibrated.values())),
         all_equal=all(r["all_equal"] for value in calibrated.values() for r in value), calibration_ok=success))
    assert success
    out = ROOT / "scores"
    out.mkdir(exist_ok=True)
    proj_dir = ROOT / "readouts"
    proj_dir.mkdir(exist_ok=True)
    files, projections = {}, {}
    for name, row in reader.rows.items():
        path = ROOT / "raw" / (name + ".pt")
        wait(path)
        p = out / (name + ".json")
        v = proj_dir / (name + ".pt")
        if not (p.exists() and v.exists()):
            raw = load(path)
            assert raw["rgb"] == row["rgb_sha256"] and raw["texts"] == ["<answer>"+q+"</answer>" for q in row["A"]]
            items, features = repeated(reader, name, raw["states"])
            capture(v, dict(visual=features, source=sha(path), image=name, rgb=raw["rgb"]))
            save(p, dict(image=name,rgb=raw["rgb"],pair=row["pair"],texts=raw["texts"],alt_ids=row["alt_ids"],
                         true=items[0], false=items[1], source=sha(path), readout=sha(v)))
        obj = read(p)
        assert obj["source"] == sha(path) and obj["readout"] == sha(v)
        files[name], projections[name] = sha(p), sha(v)
        print(len(files), name, flush=True)
    wait(ROOT / "generation.json")
    generated = read(ROOT / "generation.json")
    assert set(generated["files"]) == set(files)
    for name in files:
        assert read(out / (name + ".json"))["source"] == generated["files"][name]
    save(ROOT / "scoring.json", dict(files=files, readouts=projections, generation=sha(ROOT / "generation.json"),
         calibration_readouts=sha(ROOT/"calibration_readouts.json"), code=sha(Path(__file__)),
         threads=torch.get_num_threads(), frozen_reader=sha(Path(original.__file__)), compiled=sha(FRESH / "inputs/manifest.json")))


if __name__ == "__main__":
    main()

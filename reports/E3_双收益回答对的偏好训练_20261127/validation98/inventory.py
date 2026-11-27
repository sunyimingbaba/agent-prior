"""Audit overlap and sampling conditions of potential independent candidate archives."""
from collections import Counter
from pathlib import Path
import json

import torch
from validate import ARCH, OLD, OUT, PREVIOUS, REPORTS, STAGE, inventory, read, sha, save

def main():
    torch.set_num_threads(1)
    cold = REPORTS / "E3_R3改选收益校准与双批首句验证_20261126/cold"
    roster = read(cold / "inputs/real_roster.json")
    starts = read(cold / "inputs/real_lock.json")
    assert starts["roster"] == sha(cold / "inputs/real_roster.json")
    other = {"history98":OLD / "inputs/real_roster.json",
             "training270":PREVIOUS / "sampling/inputs/roster.json",
             "formalUCM":REPORTS / "E3-2_UCM与RSITMD独立复验_20261124/inputs/ucm.json",
             "formalRSITMD":REPORTS / "E3-2_UCM与RSITMD独立复验_20261124/inputs/rsitmd.json"}
    pixels = {r["rgb_sha256"] for r in roster}
    overlap = {n:len(pixels & {r["rgb_sha256"] for r in read(p)}) for n,p in other.items()}
    assert all(v == 0 for v in overlap.values()) and len(pixels) == 26
    modes = {}
    for s in "ab":
        cnt, reaches, total, same, complete_blocks = Counter(), 0, 0, 0, []
        base = cold / "records/real" / s / "raw"
        lock = read(base / "complete.json")
        for r in roster:
            name = r["image"]
            path = base / (name + ".pt")
            assert sha(path) == lock["files"][name]
            z = torch.load(path, weights_only=True, map_location="cpu")
            assert z["rgb"] == r["rgb_sha256"] and z["query"] == r["orig_query"] and z["feedback"] == r["feedback"]
            assert z["adapter"] == starts["starts"][s]
            reaches += bool(z["calls"])
            for block in (0, 1):
                start = 8 * block
                paths = z["paths"][start:start + 8]
                assert len(paths) == 8
                total += 8
                matches = sum(bool(z["calls"]) and len(p)>1 and p[0] == z["greedy"][0] for p in paths)
                same += matches
                cnt[matches] += 1
                if matches == 8:
                    complete_blocks.append(dict(image=name, block=block))
        modes[s] = dict(greedy_reach=reaches, candidate_paths=total, same_prefix_reached=same,
                        matching_counts_per_block=dict(sorted(cnt.items())), all_eight=complete_blocks)
    multi = REPORTS / "E3_UCM多描述奖励同预算RFPO_20261124"
    scene = {}
    for s in "ab":
        ctx = torch.load(multi / "records" / (s + "_contexts.pt"), weights_only=True, map_location="cpu")
        one = torch.load(multi / "records" / (s + "_first_batch.pt"), weights_only=True, map_location="cpu")
        scene[s] = dict(roster=len(ctx["rows"]), first_batch_image=one["image"],
                        source=sha(multi / "code/prefixes.py"), initial=ctx["adapter"])
    save(OUT / "inventory.json",dict(history98=inventory(read(OLD / "inputs/real_roster.json")),
         natural_cold26=dict(overlap=overlap, seeds=modes, roster=starts["roster"], old_targeted=True),
         scene_prompt_cold_multicap=scene))


if __name__ == "__main__":
    main()

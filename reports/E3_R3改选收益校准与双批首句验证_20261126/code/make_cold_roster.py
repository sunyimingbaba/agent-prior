"""Match screened development images to controls before cold-start sampling."""
import hashlib
import json
import numpy as np
import torch
from data import ROOT, PREV, read, sha, stream
from training_opportunities import winners


def main():
    model = read(PREV / "锁定评分参数.json")["model"]
    weights = np.array(model["coef"]) / np.array(model["scale_x"])[:,None]
    catalog, marked = {}, set()
    for row, seed, view, bank, visual in stream("dev"):
        wanted_seed = "ab"[int(row["rgb_sha256"][-1], 16) % 2]
        if view != "first" or seed != wanted_seed:
            continue
        name = row["image"]
        catalog[name] = dict(row, screening_seed=seed)
        if winners(row, bank, visual, weights)[0]["old3-old"]:
            marked.add(name)
    assert len(catalog) == 204 and 0 < len(marked) < 102
    available = set(catalog) - marked
    rows = []
    for name in sorted(marked, key=lambda n:catalog[n]["rgb_sha256"]):
        row = catalog[name]
        def preference(other):
            candidate = catalog[other]
            attribute = row["attribute"] == candidate["attribute"]
            score = (attribute and row["side"] == candidate["side"], attribute, row["scene_id"] == candidate["scene_id"])
            tie = hashlib.sha256((row["rgb_sha256"] + candidate["rgb_sha256"]).encode()).hexdigest()
            return (*score, tie)
        control = max((n for n in available if catalog[n]["pair"] != row["pair"]), key=preference)
        available.remove(control)
        for n, status in ((name, "selected"), (control, "control")):
            entry = dict(catalog[n], status=status, match=name)
            for s, base in (("a", 137000000), ("b", 147000000)):
                entry["rng_"+s] = base + 1000*entry["row"]
            rows.append(entry)
    folder = ROOT/"cold/inputs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/"real_roster.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False)+"\n")
    receipt = {"roster":sha(path), "development":len(catalog), "pairs":len(marked),
               "protocol":sha(ROOT/"冷启动训练图预筛复核.md"), "starts":read(ROOT/"inputs/pilot/decision.json")["starts"]}
    (folder/"real_lock.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print("matched pairs", len(marked))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

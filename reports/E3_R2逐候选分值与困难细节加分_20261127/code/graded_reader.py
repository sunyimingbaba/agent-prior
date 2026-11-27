"""Use the original CPU readouts and append a held-caption diagnostic and A/B hard margins."""
import base64
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(sys.argv[1]) / "code"))
from shared import REBUILD, read, rgb, setup, sha, table


@torch.inference_mode()
def main():
    output = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    setup()
    from readouts import load_model, scorer_for

    reader = load_model(False)
    proxy = scorer_for(load_model(True), REBUILD/"relation_candidates_20261120/gallery/train.pt")
    folder = REBUILD/"relation_candidates_20261120/native"
    index = read(folder/"index.json")
    assert sha(folder/"features.pt") == index["sha256"]
    bank = torch.load(folder/"features.pt", map_location="cpu", weights_only=True)
    features = bank["features"].double()
    from hashlib import sha256
    even = np.array([int(sha256(",".join(map(str,q)).encode()).hexdigest()[-1],16)%2 == 0
                     for q in bank["tokens"].tolist()])
    hard = torch.load(STAGE/"inputs/hard.pt", map_location="cpu", weights_only=True)
    cases = {r["image"]:r for r in read(REBUILD/"matched_policy_20261120/inputs/train.json")}
    assert set(hard["maps"]) == set(cases)
    images = table("train")
    print("READY", file=output)
    for line in sys.stdin:
        msg = json.loads(line)
        case = cases[msg["image"]]
        states = np.frombuffer(base64.b64decode(msg["hidden"]), dtype=np.float32).copy().reshape(8,3584)
        pixels = reader.preprocess(rgb(images,case))[None]
        mapping = index["maps"][case["image"]]
        assert mapping["captions"] == images["caption"][case["row"]].as_py()
        negative = even.copy()
        negative[mapping["positive"]] = False
        alternate_neg = ~even
        alternate_neg[mapping["positive"]] = False
        alt = sorted(set(mapping["positive"]) - {mapping["first"]})
        hard_ids = hard["maps"][case["image"]]
        hard_vectors = hard["features"][np.asarray(hard_ids).ravel()].double()
        result, cache = [], {}
        for state in torch.from_numpy(states):
            key = state.numpy().tobytes()
            if key not in cache:
                rank = proxy.score(state.tolist(),case["row"])[0]
                v = reader.img_embeds(pixels,agent_hidden=state[None])["image_feats"][0]
                sims = features @ v.double()
                margin = float(sims[mapping["first"]] - sims[negative].max())
                hard_scores = hard_vectors @ v.double()
                hard_margins = (hard_scores[::2] - hard_scores[1::2]).tolist()
                rank_other = []
                for col in alt:
                    above = int((sims[alternate_neg] > sims[col]).sum())
                    tied = int((sims[alternate_neg] == sims[col]).sum())
                    rank_other.append(float(np.mean(1/np.arange(above+1, above+tied+2))))
                cache[key] = dict(rank=rank, margin=margin, hard_margins=hard_margins,
                                  other_caption_rr=rank_other)
            result.append(cache[key])
        print(json.dumps(result,allow_nan=False), file=output)


if __name__ == "__main__":
    main()

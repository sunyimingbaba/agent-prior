"""Apply the locked two- and three-wording reward to a live group of eight priors."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "code"))
from online_reward import best, old_choices, rr


class TargetReward:
    def __init__(self, seed):
        self.text = np.load(ROOT / "inputs/text.npz")
        self.index = json.loads((ROOT / "inputs/index.json").read_text())

    def run(self, name, vectors):
        assert vectors.shape == (8, 768)
        row = self.index[name]
        bank = str(row["bank"])
        ids = row["native"]
        positives = sorted(set(ids) - {ids[0]})
        odd = self.text["odd"+bank].copy()
        odd[ids] = False
        sims = vectors @ self.text["native"+bank].T
        neg, pos = sims[:,odd], sims[:,positives]
        rr_exact = np.array([[rr(x, competition) for x in vals] for vals, competition in zip(pos, neg)], dtype=object)
        rank = rr_exact.mean(1)
        ref = best(rank, range(8))
        rivals = np.partition(neg, -8, axis=1)[:,-8:]
        gaps = pos - rivals.max(1)[:,None]
        base = np.column_stack((rank.astype(float), rr_exact.min(1).astype(float), pos.mean(1), pos.min(1),
                                gaps.mean(1), gaps.min(1), pos.mean(1)-rivals.mean(1)))
        hard = vectors @ self.text[name].T
        margin = hard[:,::2] - hard[:,1::2]
        estimates = []
        for a,b in ((0,1), (0,2), (1,2)):
            h = margin[:,[a,b]]
            x = np.column_stack((h, h.min(1), h>0, (h>0).all(1), base))
            estimates.append((x-x.mean(0)) @ self.text["coefficients"])
        packet = {"ref":ref, "exact":rank}
        return {"ordinary":ref, "old":old_choices(estimates[0],packet), "old3":old_choices(np.mean(estimates,axis=0),packet)}

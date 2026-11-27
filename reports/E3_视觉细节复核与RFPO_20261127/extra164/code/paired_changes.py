"""Describe paired changes for one prior per image, keeping each initialization visible."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1] / "E3-2_UCM与RSITMD独立复验_20261124/code"))
from metrics import fixed


def read(path):
    return json.loads(path.read_bytes())


def question_outcomes(visual, texts, question_ids):
    scores = fixed(visual, texts)
    selected = scores[np.arange(len(visual))[:, None], question_ids]
    return (selected[:, ::2] > selected[:, 1::2]).astype(int)


def directions(values):
    return dict(higher=int((values > 0).sum()), equal=int((values == 0).sum()),
                lower=int((values < 0).sum()))


def main():
    rows = read(HERE / "inputs/rows.json")
    questions = np.array([row["panel_de_ids"] for row in rows])
    texts = torch.load(HERE / "inputs/texts.pt", weights_only=True)["details"]
    scores = np.load(HERE / "per_image.npz", allow_pickle=False)
    result = {}
    for challenger, base in (("rfpo_strict", "rfpo_r2"), ("dpo_r3", "dpo_r2")):
        result[challenger] = {}
        for index, seed in enumerate("ab"):
            data = []
            for arm in (challenger, base):
                record = HERE / "records" / f"{seed}_{arm}.pt"
                assert hashlib.sha256(record.read_bytes()).hexdigest() == read(record.with_suffix(".json"))["sha256"]
                data.append(torch.load(record, weights_only=True, map_location="cpu"))
            new, old = data
            same_actions = [x["actions"] == y["actions"] for x, y in zip(new["traces"], old["traces"])]
            detail_change = question_outcomes(new["visual"], texts, questions) - question_outcomes(old["visual"], texts, questions)
            per_image = scores[challenger][index] - scores[base][index]
            np.testing.assert_array_equal(detail_change.mean(axis=1), per_image[:, 0])
            result[challenger][seed] = dict(
                images=len(rows), explicit_questions_per_image=2, detail_questions=directions(detail_change),
                details_by_image=directions(per_image[:, 0]), other_caption_MRR=directions(per_image[:, 1]),
                identical_generated_token_sequence=sum(same_actions),
                identical_tool_calls=sum(x["calls"] == y["calls"] for x, y in zip(new["traces"], old["traces"])),
                identical_native_features=int((new["visual"] == old["visual"]).all(dim=1).sum()),
            )
    output = HERE / "逐图变化计数.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

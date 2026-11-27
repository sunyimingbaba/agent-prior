"""Encode the two pre-existing training phrasings with the historical frozen CPU CUP."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
sys.path.insert(0, str(BASE / "second_answer_5090_20261124/code"))
from shared import read, sha, rgb, table, setup
from readouts import load_model, clip
import torch


@torch.inference_mode()
def main():
    setup()
    torch.set_num_threads(3)
    name = BASE/"matched_policy_20261120/inputs/train.json"
    rows = read(name)
    assert len(rows) == 24 and len({r["rgb_sha256"] for r in rows}) == 24
    words, indices, maps = [], {}, {}
    bank = table()
    for row in rows:
        rgb(bank, row)
        assert row["split"] == "train"
        maps[row["image"]] = []
        for k in ("A","B"):
            positive, negative = row[k]
            assert positive != negative
            for word in (positive, negative):
                if word not in indices:
                    indices[word] = len(words)
                    words.append(word)
            maps[row["image"]].append([indices[positive], indices[negative]])
    for folder in ("second_answer_5090_20261124/R2", "second_answer_5090_20261124/repeat/R2"):
        history = read(BASE/folder/"steps.json")
        assert [(r["image"],r["row"]) for r in rows] == [(s["image"],s["row"]) for s in history]
    model = load_model(False)
    tokens = clip.tokenize(words, context_length=77 - model.prompt_length)
    assert len(torch.unique(tokens, dim=0)) == len(words)
    features = torch.cat([model.txt_embeds(batch)["text_feats"] for batch in tokens.split(24)]).cpu()
    assert features.shape == (len(words),768) and torch.isfinite(features).all()
    out = ROOT/"inputs/hard.pt"
    torch.save({"texts": words, "tokens": tokens, "features": features, "maps": maps}, out)
    checks = {"images": len(rows), "different_phrases": len(words), "words_per_image": 4,
              "training_roster_sha256": sha(name), "archive_sha256": sha(out), "encoder_source": sha(BASE/"second_answer_5090_20261124/code/readouts.py"),
              "frozen_reader": "09ef20e0b354efbf8a7d4d49105cd92e1ae9412b3ea58820961265a065a874dc"}
    (ROOT/"results/训练细节标注核对.json").write_text(json.dumps(checks,indent=2)+"\n")
    print(checks)


if __name__ == "__main__":
    main()

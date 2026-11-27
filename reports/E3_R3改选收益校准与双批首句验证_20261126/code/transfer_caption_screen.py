"""Apply the development caption-gain screen unchanged to the two existing image panels."""
import hashlib
import json
import numpy as np
import torch
from screen_caption_gain import EPS, METRICS, PREV, ROOT, evaluate, read, screen, stream, summary


def main():
    model = read(PREV/"锁定评分参数.json")["model"]
    coef = np.array(model["coef"])/np.array(model["scale_x"])[:,None]
    masks, metadata = {}, {}
    for tag in ("v122","v148"):
        for row,seed,view,bank,visual in stream(tag):
            assert view == "first"
            image = row["rgb_sha256"]
            assert image not in metadata or metadata[image]["tag"]==tag
            metadata[image] = {"rgb":image, "image":row["image"], "pair":row["pair"], "tag":tag}
            masks[image,seed] = screen(row,bank,visual,coef)
    assert len(metadata)==270 and len(masks)==540
    preregistered_code = ROOT/"code/screen_caption_gain.py"
    frozen = hashlib.sha256(json.dumps(sorted(masks.items()),sort_keys=True).encode()).hexdigest()
    rows = []
    for tag in ("v122","v148"):
        for row,seed,view,bank,visual in stream(tag):
            image = row["rgb_sha256"]
            source = "a" if seed=="b" else "b"
            m = masks[image,source]
            if m is None:
                continue
            outcome = evaluate(row,seed,bank,visual,coef)
            rows.append({**metadata[image],**m,"direction":int(source=="b"),"all":True,
                        "caption_gain":m["disagreement"] and m["source_caption_gain"]>EPS,
                        "target":outcome.tolist()})
    assert len(rows)==538 and len({x["pair"] for x in rows})==135
    by_panel = {}
    for panel in ("v122","v148","pooled"):
        subset = [x for x in rows if panel=="pooled" or x["tag"]==panel]
        levels, contrasts = summary(subset)
        by_panel[panel] = {"levels":levels, "contrasts":contrasts}
    result = {"selector_code_sha256":hashlib.sha256(preregistered_code.read_bytes()).hexdigest(), "mask_sha256":frozen,
              "target_measure":list(METRICS),"panels":by_panel,"rows":rows}
    (ROOT/"results/配文收益筛图双批复核.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    for panel,sub in by_panel.items():
        print(panel, sub["levels"]["caption_gain"], sub["contrasts"]["caption_gain_minus_disagreement"])


if __name__=="__main__":
    torch.set_num_threads(1)
    main()

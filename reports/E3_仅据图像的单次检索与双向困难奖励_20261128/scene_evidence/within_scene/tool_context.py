"""Check pretrained scene-head provenance, split overlap and real/shuffled label accuracy."""
from collections import Counter
from pathlib import Path
from hashlib import sha256
import hashlib
import io
import json
import re
import sys

from PIL import Image
import pyarrow as pa
import torch

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent.parent
ARCHIVE = Path(sys.argv[1]) if len(sys.argv)>1 else Path("/Users/botlv/cup-deploy-assets")
def read(p):
    return json.loads(p.read_bytes())
def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def main():
    head = ARCHIVE/"experiment_archive_2026-09/models_aux/ucm_scene_cls.pt"
    log = ARCHIVE/"experiment_archive_2026-09/logs/train_scene_cls.log"
    classes = torch.load(head,map_location="cpu",weights_only=True)["classes"]
    assert sha(head)==read(STAGE/"image_only/inputs/manifest.json")["scene_head"]
    names = {}
    for split in ("train","val","test"):
        arrow = ARCHIVE/"arrow/ucm"/("ucm_caption_karpathy_label_"+split+".arrow")
        t = pa.ipc.open_file(pa.memory_map(str(arrow))).read_all()
        pixels = set()
        labels = Counter()
        for file, blob in zip(t["path"],t["image"],strict=True):
            scene_id = (int(Path(file.as_py()).stem)-1)//100
            labels[scene_id] += 1
            pixels.add(sha256(Image.open(io.BytesIO(blob.as_py())).convert("RGB").tobytes()).hexdigest())
        names[split] = (len(t),pixels,[labels[k] for k in range(len(classes))])
    record = log.read_text()
    log_counts = json.loads(re.search(r'标签分布 (\[[^\]]+\])',record).group(1))
    assert names["train"][0] == sum(log_counts)
    assert log_counts == names["train"][2] and "ucm_scene_cls.pt" in record
    rows = read(STAGE/"image_only/inputs/roster.json")
    donors = read(HERE.parent/"inputs/assignments.json")
    byrgb = {r["rgb"]:r for r in rows}
    panels = {}
    for panel in ("official","extra"):
        subset = [r for r in rows if r["panel"]==panel]
        matched = {key:0 for key in ("real_top1","real_top3","permuted_top1","permuted_top3")}
        for r in subset:
            truth = classes[(int(Path(r["image"]).stem)-1)//100]
            for key,image in (("real",r),("permuted",byrgb[donors[r["rgb"]]])):
                predicted = [x.split(" (")[0] for x in image["feedback"].split(", ")]
                assert len(set(predicted))==3
                matched[key+"_top1"] += truth==predicted[0]
                matched[key+"_top3"] += truth in predicted
        pix = {r["rgb"] for r in subset}
        panels[panel] = dict(images=len(subset),exact_rgb_in_arrow_split={s:len(pix&z[1]) for s,z in names.items()},
                             **matched)
    output = dict(head_sha256=sha(head),historical_train_log=sha(log),source_train_rows=names["train"][0],
                  class_count_matches_archive_log=True,archive_training_class_counts=log_counts,
                  unique_rgb={s:len(z[1]) for s,z in names.items()},panels=panels)
    (HERE/"分类工具与划分核验.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(panels)

if __name__=="__main__":
    main()

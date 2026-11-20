"""Bind the old quantity pairs and draw an image-only panel before any scoring."""
from pathlib import Path
import csv
import hashlib
import io
import json
import random
import sys
import pyarrow as pa
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
data, pairs = map(Path, sys.argv[1:])
with pairs.open(encoding="utf-8-sig") as f:
    original = list(csv.DictReader(f))
assert len(original)==200 and len({r["图像编号"] for r in original})==100
quantity = [dict(image=r["图像编号"], split=r["数据来源"].split()[-1], dimension="quantity",
                 wording=r["句式"][0], positive=r["正确描述"], negative=r["错误描述"],
                 source=r["用户确认状态"]) for r in original]
rng = random.Random(20261120)
blocks = [2,3,4,8,9,10,11,13,14,15,19,20]
records, panel = [], []
for split in ("train","val"):
    table = pa.ipc.open_file(str(data/f"ucm_caption_karpathy_label_{split}.arrow")).read_all()
    paths=table["path"].to_pylist()
    selected=[]
    for block in blocks:
        rows=[i for i,name in enumerate(paths) if (int(Path(name).stem)-1)//100==block]
        selected.extend(sorted(rng.sample(rows,2)))
    names={r["image"] for r in quantity if r["split"]==split}
    for i,name in enumerate(paths):
        if name not in names and i not in selected:
            continue
        image=Image.open(io.BytesIO(table["image"][i].as_py())).convert("RGB")
        records.append(dict(image=name,split=split,row=i,rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        if i in selected:
            panel.append((name,split,image))
assert len(panel)==48
for offset in range(0,len(panel),8):
    sheet=Image.new("RGB",(4*258,2*278),"white")
    draw=ImageDraw.Draw(sheet)
    for i,(name,split,image) in enumerate(panel[offset:offset+8]):
        x=i%4*258;y=i//4*278
        sheet.paste(image,(x,y));draw.text((x+3,y+257),f"{name}   {split}",fill="black")
    sheet.save(root/"review"/f"sheet_{offset//8+1}.jpg",quality=95)
for name,obj in [("quantity",quantity),("images",records),("panel",[[n,s] for n,s,_ in panel])]:
    (root/"inputs"/(name+".json")).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
print(len(records),len(panel))

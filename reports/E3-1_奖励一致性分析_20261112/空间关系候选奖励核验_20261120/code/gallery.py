"""Build and recheck the full train-only proxy image gallery."""
from pathlib import Path
import io
from PIL import Image
from shared import ROOT,ARROW,dump,sha,table,setup
from readouts import load_model,scorer_for
import torch
setup()
out=ROOT/'gallery';out.mkdir(exist_ok=True)
m=load_model(True)
p=out/'train.pt'
s=scorer_for(m,p)
t=table()
names=t['path'].to_pylist()
assert len(names)==len(set(names))==1680
for idx in [0,839,1679]:
    im=Image.open(io.BytesIO(t['image'][idx].as_py())).convert('RGB')
    with torch.inference_mode():f=s._encode_image(m.preprocess(im).unsqueeze(0).cuda()).cpu()
    assert torch.equal(f/f.norm(dim=1,keepdim=True),s.image_features[idx:idx+1])
dump(out/'index.json',dict(names=names,arrow_sha256=sha(ARROW),gallery_sha256=sha(p),anchor_rows=[0,839,1679]))
print('DONE 1680',flush=True)

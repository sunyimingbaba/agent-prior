"""Bind score-blind candidate images to UCM validation RGB bytes and scene descriptions."""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
import pyarrow as pa
from PIL import Image

root = Path(__file__).resolve().parents[1]
previous = Path(sys.argv[2])
data = pa.ipc.open_file(sys.argv[1]).read_all()
def load(name):
    return json.loads((previous / 'inputs' / name).read_bytes())
def write(name, obj):
    (root/'inputs'/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2))
anchors = [x for x in load('cases.json') if x['split']=='val']
fresh = []
for file in ['extension.tsv', 'extra_pairs.tsv']:
    with (root/'inputs'/file).open() as stream:
        fresh.extend(csv.DictReader(stream, delimiter='\t'))
old = {r['image'] for r in anchors}
new = {r[s]+'.tif' for r in fresh for s in ('a','b')}
assert not old & new
scenes = ['farmland','an airport','a baseball facility','a sandy beach','buildings','scrubland',
          'a residential neighborhood','forest','a freeway','a golf course','a harbor with boats','a road intersection',
          'a residential neighborhood','a mobile home park','a road overpass','a parking lot','a river',
          'a runway','a residential area','a storage facility','a tennis facility']
out = []
for i, name in enumerate(data['path'].to_pylist()):
    if name not in old|new:
        continue
    im = Image.open(io.BytesIO(data['image'][i].as_py())).convert('RGB')
    block = (int(Path(name).stem)-1)//100
    out.append(dict(image=name,split='val',row=i,rgb_sha256=hashlib.sha256(im.tobytes()).hexdigest(),
                    block=block,generic='An aerial view of '+scenes[block]+'.',
                    scene_query='Scene: '+scenes[block]+'.',captions=data['caption'][i].as_py()))
assert len(out)==len(old|new)
write('candidates.json',out)
write('previous_val.json',anchors)
print(len(old),len(new),len(out))

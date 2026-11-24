"""Freeze input rows and audit decoded RGB duplicates before policy inference."""
import hashlib, io, sys
from pathlib import Path
from collections import Counter
import pyarrow as pa
from PIL import Image
from shared import ROOT, dump, sha

def read_data(path):
    t = pa.ipc.open_file(pa.memory_map(str(path))).read_all()
    rows = []
    for i in range(len(t)):
        im = Image.open(io.BytesIO(t['image'][i].as_py())).convert('RGB')
        caps = t['caption'][i].as_py()
        rows.append(dict(row=i,image=t['path'][i].as_py(),rgb_sha256=hashlib.sha256(im.tobytes()).hexdigest(),
                         size=list(im.size),orig_query=caps[0],captions=caps))
    return rows

def main(base):
    files = {}
    rows = {}
    for dn, split in [('ucm','train'),('ucm','val'),('ucm','test'),('rsitmd','test')]:
        path = base/dn/f'{dn}_caption_karpathy_label_{split}.arrow'
        key = dn+'_'+split
        files[key] = sha(path)
        rows[key] = read_data(path)
    key = lambda r: (tuple(r['size']),r['rgb_sha256'])
    used = {key(r) for name in ('ucm_train','ucm_val') for r in rows[name]}
    regions = {}
    for dn in ['ucm','rsitmd']:
        source = rows[dn+'_test']
        prior, seen = used.copy(), set()
        kept, excluded = [], []
        for r in source:
            h = key(r)
            if h in prior or h in seen:
                excluded.append(dict(image=r['image'],reason='previous_split' if h in prior else 'within_split'))
            else:
                kept.append(r)
            seen.add(h)
        used.update(seen)
        assert all(len(r['captions'])==5 for r in kept)
        dump(ROOT/'inputs'/f'{dn}.json',kept)
        regions[dn]=dict(source=len(source),kept=len(kept),excluded=excluded,
                         manifest=sha(ROOT/'inputs'/f'{dn}.json'))
    dump(ROOT/'inputs/audit.json',dict(files=files,regions=regions,source_counts={k:len(v) for k,v in rows.items()}))

if __name__=='__main__':main(Path(sys.argv[1]))

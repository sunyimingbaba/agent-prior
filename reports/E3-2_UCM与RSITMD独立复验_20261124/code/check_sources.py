"""Cross-check source captions and split assignments against the local annotation distributions."""
import hashlib,json,sys,zipfile
from pathlib import Path
from collections import Counter
from shared import ROOT,read,dump
import pyarrow as pa
def main(assets,repo):
    info={}
    for dn in ('ucm','rsitmd'):
        if dn=='ucm':raw=(repo/'UCM_captions/dataset.json').read_bytes()
        else:
            with zipfile.ZipFile(repo/'datasets/RSITMD_original.zip') as z:raw=z.read('dataset_RSITMD.json')
        a=json.loads(raw)['images']; ref={r['filename']:r for r in a if r['split']=='test'}
        f=assets/dn/f'{dn}_caption_karpathy_label_test.arrow'
        t=pa.ipc.open_file(pa.memory_map(str(f))).read_all()
        names=t['path'].to_pylist()
        assert set(ref)==set(names) and len(ref)==len(names)
        exact=0;differences=[]
        for name,caps in zip(names,t['caption'].to_pylist()):
            stored=[s['raw'] for s in ref[name]['sentences']]
            if stored==caps:exact+=1
            else:differences.append(name)
        info[dn]=dict(source_sha256=hashlib.sha256(raw).hexdigest(),split_counts=dict(Counter(r['split'] for r in a)),
           identical_names=len(ref),identical_caption_lists=exact,caption_differences=differences)
    dump(ROOT/'inputs/annotation_audit.json',info)
if __name__=='__main__':main(*map(Path,sys.argv[1:]))

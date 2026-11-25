"""Preserve exact generated single-pass trajectories independently of projection speed."""
import sys,tarfile
from pathlib import Path
from rosters import ROOT,read,sha,dump
STORE=Path('/Users/botlv/cup-deploy-assets/e3-source-transfer-20261125')
def package(sub,seed,panel):
    f=STORE/'local_projection'/sub/seed
    if panel:f=f/panel
    proof=read(f/'complete.json')
    out=ROOT/'records/device/raw';out.mkdir(exist_ok=True,parents=True);name='_'.join(x for x in (sub,seed,panel) if x)
    arc=out/(name+'.tar.gz');receipt=out/(name+'.json')
    if receipt.exists() and sha(arc)==read(receipt)['archive']:return
    with tarfile.open(arc.with_suffix('.part'),'w:gz',compresslevel=3) as tf:
        tf.add(f/'complete.json',arcname='complete.json')
        for nm,digest in proof['files'].items():
            p=f/(nm+'.pt');assert sha(p)==digest;tf.add(p,arcname=p.name)
    arc.with_suffix('.part').replace(arc)
    dump(receipt,dict(images=len(proof['files']),source=sha(f/'complete.json'),archive=sha(arc),adapter=proof['adapter']))
if __name__=='__main__':
    if sys.argv[1]=='route':
        for s in 'ab':
            for tag in ('primary','official'):package('route',s,tag)
    else:package('cross',sys.argv[2],None)

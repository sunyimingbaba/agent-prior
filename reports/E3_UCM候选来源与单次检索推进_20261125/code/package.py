"""Store original trajectories in one checksummed archive per panel and seed."""
import gzip
import io
import json
import tarfile
from pathlib import Path
import sys
from rosters import ROOT,read,sha,dump

def main(tag,seed):
    folder=ROOT/'records'/tag/seed/'raw'
    proof=read(folder/'complete.json');dest=folder.parent/'trajectories.tar.gz';part=dest.with_suffix('.part')
    receipt=folder.parent/'archive.json'
    if receipt.exists() and dest.exists():
        done=read(receipt)
        if done['raw_proof']==sha(folder/'complete.json') and done['sha256']==sha(dest):
            assert done['images']==len(proof['files']);return
    with tarfile.open(part,'w:gz',compresslevel=3) as output:
        output.add(folder/'complete.json',arcname='complete.json')
        for image,h in proof['files'].items():
            p=folder/(image+'.pt');assert sha(p)==h
            output.add(p,arcname=p.name)
    part.replace(dest)
    dump(folder.parent/'archive.json',dict(file=dest.name,sha256=sha(dest),images=len(proof['files']),raw_proof=sha(folder/'complete.json')))
if __name__=='__main__':main(*sys.argv[1:])

"""Serialize evaluation of a particular completed/ongoing arm and part across the GPU queues."""
import fcntl,sys,os
from shared import ROOT,read,sha
def main(arm,part):
    out=ROOT/'eval'/arm
    if part!='score':out=out/part
    out.mkdir(parents=True,exist_ok=True);descriptor=os.open(out,os.O_RDONLY)
    try:
        fcntl.flock(descriptor,fcntl.LOCK_EX)
        receipt=out/('scored.json' if part=='score' else 'complete.json')
        if receipt.is_file():
            r=read(receipt)
            if part=='score':assert r['vectors']==sha(out/'vectors.pt')
            else:assert r['arm']==arm and r['part']==part
            return
        if part=='score':
            from score_eval import main as score
            score(arm)
        else:
            from deploy import run
            run(arm,part)
    finally:os.close(descriptor)
if __name__=='__main__':main(*sys.argv[1:])

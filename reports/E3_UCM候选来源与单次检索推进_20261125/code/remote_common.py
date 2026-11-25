from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
sys.path.insert(0,str(BASE/'multicap_rfpo_20261124/code'))
from runtime import read,sha,table,rgb,setup,savept,dump
def roster(tag):
    lock=read(ROOT/'inputs/lock.json')
    assert sha(ROOT/'实验口径.md')==lock['protocol']
    p=ROOT/'inputs'/(tag+'.json')
    assert sha(p)==lock['panels'][tag]['file']
    return read(p)
def directory(tag,seed,kind):
    out=ROOT/'records'/tag/seed/kind
    out.mkdir(exist_ok=True,parents=True)
    return out

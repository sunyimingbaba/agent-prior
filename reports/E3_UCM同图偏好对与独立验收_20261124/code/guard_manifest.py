import hashlib,json,sys
from pathlib import Path
from dev_metrics import ROOT,REP,HIST,read,sha,dump
OLD=REP/'E3-1_UCM第二轮R3困难负样本复验_20261124';sys.path.insert(0,str(OLD/'code'))
from pairs import BANK
def main():
    r=ROOT/'guard';cf=read(HIST/'inputs/panel.json')
    for c in cf:
        side=c['side'];ab=BANK[c['attribute']][:2]
        c.update({k:[a[side],a[1-side]] for k,a in zip('AB',ab)})
    dump(r/'inputs/panel.json',cf);dump(r/'inputs/lock.json',dict(protocol=sha(r/'实验口径.md'),panel=sha(r/'inputs/panel.json'),train_arrow=read(HIST/'inputs/lock.json')['train_arrow']))
if __name__=='__main__':main()

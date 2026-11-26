from pathlib import Path
import hashlib,json,sys
import torch
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'E3_普通配文连续扣分与独立检验_20261125'
sys.path.insert(0,str(OLD/'code'))
from compare import read,dump,sha,load,boot
from single_scores import FORM,RFPO,calculate,summary_by_seed
from exact_scores import paired
def feature_path(row,seed,view):
    if view=='first':base=OLD/'records/real'
    elif row['tag'] in ('fresh','extension'):base=OLD/'records'/row['tag']
    else:base=OLD/'records/covered'/row['tag']
    return base/seed/'features'/(row['image']+'.pt')
def digest(*values):return hashlib.sha256('|'.join(values).encode()).hexdigest()
def banks():
    return [load(OLD/'records/text.pt'),load(OLD/'records/covered/bank/text.pt')]
def bank_for(row,packs):return packs[int(row['tag'] in ('saved','extra'))]

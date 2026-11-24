import hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
REP=ROOT.parent
LAST=REP/'E3_UCM同图偏好对与独立验收_20261124'
HIST=REP/'E3-2_UCM有效图组定向短训_20261124'
OLD=REP/'E3-1_UCM第二轮R3困难负样本复验_20261124'
FORM=REP/'E3-2_UCM与RSITMD独立复验_20261124'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def steps(tag):return [json.loads(line) for line in (ROOT/'pilot'/tag/'steps.jsonl').read_text().splitlines()]
def trainrows():
    return {r['image']:r for sub in ('','confirm') for r in read(OLD/sub/'inputs/panel.json')}

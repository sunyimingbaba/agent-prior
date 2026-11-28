from pathlib import Path
import json,hashlib,re
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];REP=ROOT.parents[1]
TR=REP/"E3_回填后双收益训练图筛选_20261127/sampling/inputs/roster.json"
FO=REP/"E3_UCM多描述奖励同预算RFPO_20261124/inputs/panel.json"
GALLERY=REP/"E3-2_UCM与RSITMD独立复验_20261124/inputs/ucm.json"
EXT=REP/"E3_视觉细节复核与RFPO_20261127/extra164/inputs/rows.json"
AK=("field_axis","bridge_axis","freeway_axis","harbor_rows","mobile_axis","parking_axis","river_axis","street_axis")
def read(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def axis(sentence):
    text=sentence.lower()
    hor=any(x in text for x in ("horizontal","left to right","left-right","sideways","width","side to side","from left toward right","across the frame from side"))
    ver=any(x in text for x in ("vertical","top to bottom","top-bottom","height","upper edge","top toward bottom","up and down","from top toward bottom"))
    if hor==ver:return None
    return "horizontal" if hor else "vertical"
def main():
    pixels={r['rgb_sha256'] for r in read(GALLERY)}
    pools=dict(train270=read(TR),formal106=[r for r in read(FO) if r['rgb_sha256'] in pixels],extra164=read(EXT))
    selection={};base={}
    for name,rows in pools.items():
        assert len(rows)=={'train270':270,'formal106':106,'extra164':164}[name]
        out=[]
        for r in rows:
            if r['attribute'] not in AK:continue
            expected='horizontal' if r['side']==0 else 'vertical';other='vertical' if expected=='horizontal' else 'horizontal'
            readouts=[]
            for k in ('A','B') if name!='formal106' else ('D','E'):
                label,opposite=r[k]
                if axis(label) is None or axis(opposite) is None:
                    # The historical wording is validated below against the complementary phrasing as well.
                    continue
                assert (axis(label),axis(opposite))==(expected,other),(name,k,r['attribute'],r[k])
                readouts.append(k)
            assert len(readouts)>=(1 if name=='formal106' else 2),(name,r['attribute'],r.get('D'),r.get('E'))
            out.append(dict(image=r['image'],rgb=r['rgb_sha256'],row=r['row'],split=r.get('split','test'),pair=r['pair'],attribute=r['attribute'],
                label=expected,truth_sources=readouts,correct_phrasing={k:r[k][0] for k in readouts}))
        pair={}
        for x in out:pair.setdefault(x['pair'],[]).append(x)
        assert all(len(g)==2 and {r['label'] for r in g}=={'horizontal','vertical'} for g in pair.values())
        selection[name]=out;base[name]=[r['rgb_sha256'] for r in rows]
    assert {x['attribute'] for x in selection['train270']}==set(AK)
    dump(ROOT/'inputs/roster.json',selection)
    paths=[ROOT/'口径.md',ROOT/'code/pixels.py',ROOT/'code/prepare.py',ROOT/'code/check.py',ROOT/'inputs/roster.json']
    out=dict(files={str(p.relative_to(ROOT)):sha(p) for p in paths},sources={str(p.relative_to(REP)):sha(p) for p in (TR,FO,GALLERY,EXT)},
             attributes=AK,full_sizes={k:len(v) for k,v in base.items()},full_overlap={a+'__'+b:len(set(base[a])&set(base[b])) for i,a in enumerate(base) for b in list(base)[i+1:]})
    p=ROOT/'inputs/lock.json'
    if p.exists():assert read(p)==json.loads(json.dumps(out))
    else:dump(p,out)
if __name__=="__main__":main()

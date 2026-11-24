"""Assemble contact sheets from both unused-for-gradient UCM splits before opening model results."""
import hashlib,io,json,sys
from collections import defaultdict
from pathlib import Path
import pyarrow as pa
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];REP=ROOT.parent
ASSET=Path('/Users/botlv/cup-deploy-assets/arrow/ucm')
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def prior_detail_names():
    used=set();sources={}
    for p in REP.glob('E*/**/inputs/**/*.json'):
        if ROOT in p.parents or p.stat().st_size>9_000_000:continue
        if p.name not in {'cases.json','evaluation_cases.json','quantity_cases.json','secondary_eval.json','sft_all_cases.json','panel.json','selected.json','case_inputs.json','train.json'}:continue
        x=read(p)
        if not isinstance(x,list):continue
        n={v['image'] for v in x if isinstance(v,dict) and isinstance(v.get('image'),str) and any(k in v for k in ('dimension','positive','negative','A','C','attribute'))}
        if n:used.update(n);sources[str(p.relative_to(REP))]=len(n)
    return used,sources
def gather():
    used,_=prior_detail_names();rows={}
    for split in ('val','test'):
        t=pa.ipc.open_file(str(ASSET/f'ucm_caption_karpathy_label_{split}.arrow')).read_all()
        for j,nm in enumerate(t['path'].to_pylist()):
            if nm in used:continue
            im=Image.open(io.BytesIO(t['image'][j].as_py())).convert('RGB')
            rows[nm]=dict(im=im,row=j,split=split,scene_id=(int(nm[:-4])-1)//100,orig_query=t['caption'][j].as_py()[0],rgb_sha256=hashlib.sha256(im.tobytes()).hexdigest())
    return rows
def sheets(dest):
    dest.mkdir(exist_ok=True,parents=True);rows=gather()
    for g,scenes in enumerate(([0,2,3],[4,6,8],[10,11,12],[13,14,15],[16,17,18],[20,1,5],[7,9,19])):
        im=Image.new('RGB',(10*116,6*133),'white');draw=ImageDraw.Draw(im)
        for i,sc in enumerate(scenes):
            for j,split in enumerate(('val','test')):
                names=sorted((n for n,r in rows.items() if r['scene_id']==sc and r['split']==split),key=lambda n:int(n[:-4]))
                for k,n in enumerate(names):
                    x,y=k*116,(i*2+j)*133;im.paste(rows[n]['im'].resize((113,113)),(x,y));draw.text((x+2,y+113),n+' '+split,fill='black')
        im.save(dest/f'{g}.jpg',quality=94)
    print(len(rows),'detail-unexposed val/test images')
if __name__=='__main__':sheets(Path(sys.argv[1]))

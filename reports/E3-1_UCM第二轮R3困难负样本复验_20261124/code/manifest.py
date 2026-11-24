"""Make a train-only panel. All splits and exclusions are checked before sampling."""
import csv,hashlib,io,json,sys
from pathlib import Path
import pyarrow as pa
from PIL import Image
from pairs import BANK,PAIRS,SCENE
RUN=Path(__file__).resolve().parents[1]
def main(base):
    repo=RUN.parents[1];reports=repo/'reports'
    train=base/'ucm/ucm_caption_karpathy_label_train.arrow'
    t=pa.ipc.open_file(pa.memory_map(str(train))).read_all()
    banned={r['image'] for r in json.loads((reports/'E3-2_UCM同起点三奖励短训_20261120/inputs/train.json').read_text())}
    with (reports/'E3_UCM候选奖励与跨图复验_20261123/开发逐图.csv').open(encoding='utf-8-sig') as f:
        banned.update(r['image'] for r in csv.DictReader(f))
    banned.update(r['image'] for r in json.loads((reports/'E3-2_UCM双视图方位奖励_20261120/inputs/e1/cases.json').read_text()) if r['split']=='train')
    hold=reports/'E3-2_UCM与RSITMD独立复验_20261124/inputs'
    used_pixels={r['rgb_sha256'] for dn in ['ucm','rsitmd'] for r in json.loads((hold/f'{dn}.json').read_text())}
    names=t['path'].to_pylist();rows=[]
    for label,a,b in PAIRS:
        scene=(a-1)//100
        assert scene==(b-1)//100
        noun=SCENE[scene];article='' if noun in ('buildings','tennis courts') else 'an ' if noun[0] in 'aeiou' else 'a '
        for side,i in enumerate((a,b)):
            name=f'{i}.tif';j=names.index(name);assert name not in banned and t['split'][j].as_py()=='train'
            im=Image.open(io.BytesIO(t['image'][j].as_py())).convert('RGB')
            h=hashlib.sha256(im.tobytes()).hexdigest()
            assert h not in used_pixels
            used_pixels.add(h)
            caps=t['caption'][j].as_py()
            rows.append(dict(image=name,row=j,split='train',rgb_sha256=h,pair=f'{a}-{b}',attribute=label,
                  side=side,scene=noun,scene_id=scene,scene_query=f'Scene: {article}{noun}.',orig_query=caps[0],captions=caps,
                  **{k:[p[side],p[1-side]] for k,p in zip('ABC',BANK[label],strict=True)}))
    assert len(rows)==len(set(r['image'] for r in rows))
    out=RUN/'inputs/panel.json';out.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    def sha(p):
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    (RUN/'inputs/lock.json').write_text(json.dumps(dict(panel=sha(out),train_arrow=sha(train),
         protocol=sha(RUN/'实验口径.md'),excluded_prior_images=len(banned)),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))

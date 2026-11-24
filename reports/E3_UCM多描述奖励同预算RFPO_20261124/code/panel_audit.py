from collections import defaultdict,Counter
import io,hashlib
from pathlib import Path
import pyarrow as pa
from PIL import Image
from report_io import ROOT,LAST,HIST,FORM,OLD,load,read,sha,dump,trainrows
def main():
    folder=Path('/Users/botlv/cup-deploy-assets/arrow/ucm');lock=read(ROOT/'inputs/panel_lock.json')
    p=ROOT/'inputs/panel.json';assert sha(p)==lock['panel'] and sha(ROOT/'code/panel.py')==lock['selection_source'] and sha(ROOT/'实验口径.md')==lock['protocol']
    snapshot=read(ROOT/'inputs/screening_snapshot.json')
    assert all(sha(ROOT.parent/name)==h for name,h in snapshot['files'].items())
    panels=read(p);groups=defaultdict(list)
    earlier=list(trainrows().values())+read(HIST/'inputs/panel.json')+read(LAST/'confirm/inputs/panel.json')+read(LAST/'reward_confirm/inputs/panel.json')
    tables={}
    for sp,h in lock['arrows'].items():
        f=folder/f'ucm_caption_karpathy_label_{sp}.arrow';assert sha(f)==h;tables[sp]=pa.ipc.open_file(str(f)).read_all()
    for row in panels:
        t=tables[row['split']];j=row['row'];n=row['image'];assert t['path'][j].as_py()==n and t['caption'][j].as_py()[0]==row['orig_query']
        im=Image.open(io.BytesIO(t['image'][j].as_py())).convert('RGB');assert hashlib.sha256(im.tobytes()).hexdigest()==row['rgb_sha256']
        groups[row['pair']].append(row)
    for v in groups.values():
        assert len(v)==2 and {k['side'] for k in v}=={0,1}
        l,r=v;assert l['scene_id']==r['scene_id'] and l['split']==r['split'] and l['scene_query']==r['scene_query'] and l['attribute']==r['attribute']
        assert len(set(sum((l[k] for k in 'CDE'),[])))==6 and all(l[k]==r[k][::-1] for k in 'CDE')
    for k in ('image','rgb_sha256'):
        ids={r[k] for r in panels};assert len(ids)==len(panels) and not ids&{r[k] for r in earlier}
    assert not set(snapshot['excluded_images'])&{r['image'] for r in panels}
    out=dict(images=len(panels),pairs=len(groups),classes=len({r['scene_id'] for r in panels}),attributes=len({r['attribute'] for r in panels}),by_split=dict(Counter(r['split'] for r in panels)),prior_geometry_checked=lock['prior_geometric'])
    if (ROOT/'records/text.pt').exists():
        bank=load(ROOT/'records/text.pt');native=load(FORM/'results/ucm_text.pt')['native_tokens']
        if native.shape[1]==bank['tokens'].shape[1]:
            ids=set(sum((bank['maps']['primary'][r['image']][k] for r in panels for k in 'CDE'),[]))
            nt={tuple(v) for v in native.tolist()}
            out['primary_token_sequences']=len(ids)
            out['exact_official_native_token_overlap']=sum(tuple(bank['tokens'][i].tolist()) in nt for i in ids)
    dump(ROOT/'results/panel_audit.json',out)
if __name__=='__main__':main()

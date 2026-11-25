"""Identity-checked data for single generated priors and frozen native CUP."""
from pathlib import Path
import hashlib, json
import pyarrow as pa
import numpy as np
from PIL import Image
import io, torch
from torchvision import transforms as T
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
SOURCE=BASE/'source_transfer_2h_20261125';FORM=BASE/'after_tool_holdout_20261124';RFPO=BASE/'multicap_rfpo_20261124'
PAIR=BASE/'paired_answer_preference_20261124'
ASSETS=Path('/private/AGSE_E1_20261110/legacy/assets')
def read(p):return json.loads(p.read_bytes())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def dump(p,z):
    q=p.with_suffix('.part');q.write_text(json.dumps(z,indent=2,allow_nan=False)+'\n');q.replace(p)
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def save(p,z):
    q=p.with_suffix('.part');torch.save(z,q);q.replace(p)
def reader():
    p=SOURCE/'scratch/export/cup_native_trace.pt';assert sha(p)==read(SOURCE/'scratch/export/proof.json')['trace']
    m=torch.jit.load(str(p),map_location='cpu').eval().cuda()
    for module in m.modules():
        for method in module._c._method_names():
            for node in module._c._get_method(method).graph.findAllNodes('prim::Constant'):
                if str(node.output().type())=='Device' and 'value' in node.attributeNames() and node.s('value')=='cpu':node.s_('value','cuda:0')
    for par in m.parameters():par.requires_grad_(False)
    return m

def pack(tag,seed):
    training=tag in ('first','second','third');fresh=tag in ('fresh','extension')
    if training:
        rows=read(SOURCE/'inputs'/(tag+'.json'));directory={'first':BASE/'vote_confirm_20261125','second':PAIR/'reward_confirm','third':PAIR/'guard'}[tag]
    elif fresh:
        rows=read(ROOT/'inputs'/(tag+'.json'));directory=ROOT
    elif tag=='primary':
        rows=read(RFPO/'inputs/panel.json');directory=RFPO
    else:
        rows=read(FORM/'inputs'/('ucm.json' if tag=='official' else 'rsitmd.json'));directory=None
    bank=None if directory is None else load(directory/'records/text.pt')
    transform=T.Compose([T.Resize(224,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),
                          T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    tables={};images=[];hs=[];hn=[];ab=[];cd=[];positives=[]
    for j,c in enumerate(rows):
        nm=c['image'];sp=c.get('split','test')
        if sp not in tables:
            p=FORM/'inputs/rsitmd_caption_karpathy_label_test.arrow' if tag=='cross' else ASSETS/'ucm'/('ucm_caption_karpathy_label_'+sp+'.arrow')
            tables[sp]=pa.ipc.open_file(pa.memory_map(str(p))).read_all()
        tb=tables[sp];assert tb['path'][c['row']].as_py()==nm
        im=Image.open(io.BytesIO(tb['image'][c['row']].as_py())).convert('RGB')
        assert hashlib.sha256(im.tobytes()).hexdigest()==c['rgb_sha256']
        images.append(transform(im))
        if training:
            source=(SOURCE/'records/prefix/third'/seed if tag=='third' else SOURCE/'records'/tag/seed)/'raw'/(nm+'.pt')
            z=load(source);assert z['rgb']==c['rgb_sha256'] and z['image']==nm
            hz=z['modes']['locked' if tag=='third' else 'scene']['hidden'][-1]
            o=load(ROOT/'records/ordinary'/tag/seed/(nm+'.pt'));assert o['adapter']==z['adapter'] and o['query']==c['orig_query'] and o['rgb']==z['rgb']
            hs.append(hz);hn.append(o['prior'])
        elif fresh:
            z=load(ROOT/'records'/tag/seed/'raw'/(nm+'.pt'));assert z['rgb']==c['rgb_sha256'] and z['query']==c['scene_query'];hs.append(z['hidden'][-1])
        else:
            path=(FORM/'outputs'/(seed+'_R2')/'rsitmd'/('%04d.pt'%c['row'])) if tag=='cross' else (RFPO/'greedy'/(seed+'_deployed')/tag/(nm+'.pt'))
            z=load(path);fld='scene_query' if tag=='primary' else 'orig_query'
            assert z['query']==c[fld] and z['rgb']==c['rgb_sha256'] and z['image']==nm
            (hs if tag=='primary' else hn).append(z['prior'])
        if training or fresh:
            ix=bank['maps'][nm];positives.append(ix['native'])
            ab.append(torch.stack([bank['hard'][ix[k][0]]-bank['hard'][ix[k][1]] for k in 'AB']))
            cd.append(torch.stack([bank['hard'][ix[k][0]]-bank['hard'][ix[k][1]] for k in 'CDE']))
        elif tag=='primary':
            ix=bank['maps']['primary'][nm];cd.append(torch.stack([bank['hard'][ix[k][0]]-bank['hard'][ix[k][1]] for k in 'CDE']))
    return dict(rows=rows,pixels=torch.stack(images),scene=torch.stack(hs).float() if hs else None,ordinary=torch.stack(hn).float() if hn else None,
                ab=torch.stack(ab) if ab else None,cd=torch.stack(cd) if cd else None,pos=torch.tensor(positives) if positives else None,
                native=bank['native'] if training or fresh else None,adapter=z['adapter'])

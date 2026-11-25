"""Independent local replay of frozen single-pass image embeddings, including every historical baseline."""
from pathlib import Path
import hashlib,io,sys,tarfile
from PIL import Image
import numpy as np
import pyarrow as pa
import torch
from torchvision import transforms as T
from rosters import ROOT,read,sha,dump
from single_scores import RFPO,FORM,calculate,summary_by_seed,load
STORE=Path('/Users/botlv/cup-deploy-assets/e3-source-transfer-20261125')
ASSETS=STORE.parent
def arrow(split):
    p=ASSETS/'arrow/ucm'/('ucm_caption_karpathy_label_'+split+'.arrow')
    return pa.ipc.open_file(pa.memory_map(str(p))).read_all()
def load_device():
    proof=read(STORE/'readout/proof.json');file=STORE/'readout/cup_native_trace.pt';assert sha(file)==proof['trace']
    assert proof['reader']=='09ef20e0b354efbf8a7d4d49105cd92e1ae9412b3ea58820961265a065a874dc'
    m=torch.jit.load(str(file),map_location='cpu').eval().to('mps');torch.set_num_threads(2)
    # Trace captured the CPU used for the visual class-token zeros. Relocate that device literal only.
    for module in m.modules():
        for method in module._c._method_names():
            graph=module._c._get_method(method).graph
            for node in graph.findAllNodes('prim::Constant'):
                if str(node.output().type())=='Device' and 'value' in node.attributeNames() and node.s('value')=='cpu':
                    node.s_('value','mps:0')
    convert=T.Compose([T.Resize(224,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop((224,224)),T.ToTensor(),
                       T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    return m,convert,proof
@torch.inference_mode()
def main():
    m,convert,proof=load_device()
    tables={s:arrow(s) for s in ('val','test')};out={};report={}
    for seed in 'ab':
        arc=RFPO/'archives'/(seed+'_deployed.tar.gz')
        with tarfile.open(arc) as ref:
            for sub,rows in [('primary',read(RFPO/'inputs/panel.json')),('official',read(FORM/'inputs/ucm.json'))]:
                old=load(RFPO/'vectors'/(seed+'_deployed_'+sub+'.pt'));folder=STORE/'local_projection/route'/seed/sub
                cp=read(folder/'complete.json');assert cp['adapter']==old['adapter'] and cp['route']==sha(ROOT/'inputs/route.json')
                vectors=[];maxerr=0.
                for j,c in enumerate(rows):
                    nm=c['image'];field='scene_query' if sub=='primary' else 'orig_query'
                    z=load(folder/(nm+'.pt'));assert sha(folder/(nm+'.pt'))==cp['files'][nm]
                    a=torch.load(io.BytesIO(ref.extractfile(f'{seed}_deployed/{sub}/{nm}.pt').read()),weights_only=True)
                    assert a['rgb']==z['rgb']==c['rgb_sha256'] and a['query']==z['original']==c[field] and a['adapter']==z['adapter']==old['adapter']
                    t=tables[c.get('split','test')];assert t['path'][c['row']].as_py()==nm
                    im=Image.open(io.BytesIO(t['image'][c['row']].as_py())).convert('RGB');assert hashlib.sha256(im.tobytes()).hexdigest()==z['rgb']
                    pix=convert(im)[None].to('mps');hs=torch.stack([a['prior'],z['prior']]).float().to('mps')
                    v=m(pix.repeat(2,1,1,1),hs).cpu();assert v.shape==(2,768)
                    maxerr=max(maxerr,float((v[0]-old['features'][j]).abs().max()));vectors.append(v)
                    if (j+1)%60==0:print(sub,seed,j+1,maxerr,flush=True)
                v=torch.stack(vectors);reference=calculate(sub,old['features'],rows)
                before=calculate(sub,v[:,0],rows);after=calculate(sub,v[:,1],rows)
                mismatches={k:int((abs(before[k]-x)>1e-10).sum()) for k,x in reference.items()}
                assert maxerr<3e-4
                out[sub,seed]=dict(before=before,after=after,change={k:after[k]-before[k] for k in before})
                dest=ROOT/'records/device';dest.mkdir(exist_ok=True);torch.save(dict(vectors=v,ids=old['images'],adapter=old['adapter'],source=proof['trace']),dest/(seed+'_'+sub+'.pt'))
                report[sub+'_'+seed]=dict(images=len(rows),max_baseline_embedding_difference=maxerr,baseline_endpoint_disagreements=mismatches)
    for sub,rows in [('primary',read(RFPO/'inputs/panel.json')),('official',read(FORM/'inputs/ucm.json'))]:
        labels=[c.get('pair',c['image']) for c in rows]
        report[sub]={k:summary_by_seed({s:out[sub,s][k] for s in 'ab'},labels) for k in ('before','after','change')}
    dump(ROOT/'results/independent_device.json',report)
if __name__=='__main__':main()

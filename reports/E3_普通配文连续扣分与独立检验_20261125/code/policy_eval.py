"""One real greedy generation per test image and frozen checkpoint."""
from pathlib import Path
import sys,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'source_transfer_2h_20261125/code'))
from remote_common import setup,table,rgb,read,dump,sha,savept
from eval_io import datasets
from policy import policy
from prefixes import greedy
from head_data import reader
from torchvision import transforms as T
@torch.inference_mode()
def evaluate(seed,arm):
    setup();torch.set_num_threads(2);source=ROOT/'policy'/(seed+'_'+arm);proof=read(source/'complete.json');assert sha(source/'adapter/adapter_model.safetensors')==proof['final']
    m,proc=policy(source);native=reader()
    transform=T.Compose([T.Resize(224,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    ds=datasets();tables={s:table(s) for s in ('val','test')}
    for sub in ('primary','official'):
        rows,fb,field=ds[sub];folder=source/'evaluate'/sub;folder.mkdir(exist_ok=True,parents=True);vs=[]
        for c in rows:
            nm=c['image'];f=folder/(nm+'.pt')
            if f.exists():
                z=torch.load(f,weights_only=True);assert z['adapter']==proof['final'] and z['rgb']==c['rgb_sha256'] and z['query']==c[field];vs.append(z['v']);continue
            im=rgb(tables[c['split']],c);h,action,calls,_=greedy(m,proc,im,c[field],fb[nm])
            v=native(transform(im)[None].cuda(),h.float()[None].cuda())[0].cpu()
            savept(f,dict(image=nm,rgb=c['rgb_sha256'],query=c[field],adapter=proof['final'],actions=action,calls=calls,prior=h,v=v));vs.append(v)
        savept(source/'evaluate'/(sub+'.pt'),dict(features=torch.stack(vs),images=[r['image'] for r in rows],adapter=proof['final']))
        dump(folder/'complete.json',dict(files={c['image']:sha(folder/(c['image']+'.pt')) for c in rows},features=sha(source/'evaluate'/(sub+'.pt'))))
        print(seed,arm,sub,len(vs),flush=True)
if __name__=='__main__':evaluate(*sys.argv[1:])

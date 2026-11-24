"""Frozen FP32 readout of each greedy prior; join historical starting evaluations by digest."""
import time,torch
from common import ROOT as RUN,OLD,ARMS,PREVIOUS,REBUILD,sha,read,dump,savept,setup,rgb
import importlib.util
from pathlib import Path
entry=importlib.util.spec_from_file_location('informative_group_deployment',Path(__file__).with_name('evaluate.py'))
deployment=importlib.util.module_from_spec(entry)
entry.loader.exec_module(deployment)
datasets=deployment.datasets
from readouts import load_model
@torch.inference_mode()
def main():
    setup();m=load_model(False);ds=datasets();out=RUN/'pilot/vectors';out.mkdir(exist_ok=True)
    for seed,reward in [(s,a) for s in ('a','b') for a in ('start',*ARMS)]:
        tag=f'{seed}_{reward}'
        for name,(t,rows,answers,qkey) in ds.items():
            f=out/f'{tag}_{name}.pt'
            if f.exists():continue
            features=[];hashes=[];old=name=='val' and reward=='start'
            folder=(PREVIOUS/('repeat' if seed=='b' else '')/'R2/eval') if old else (RUN/'pilot/greedy'/tag/name)
            for j,c in enumerate(rows):
                nm=c['image'];p=(folder/f'{j:03d}.pt') if old else (folder/(nm+'.pt'))
                while not p.is_file():time.sleep(1)
                obj=torch.load(p,weights_only=True,map_location='cpu')
                assert obj['image']==nm and obj['rgb']==c['rgb_sha256'] and obj['query']==c[qkey]
                if old:
                    v=obj['feature']
                else:
                    pix=m.preprocess(rgb(t,c))[None];v=m.img_embeds(pix,agent_hidden=obj['prior'].float()[None])['image_feats'][0]
                assert torch.isfinite(v).all() and abs(v.norm().item()-1)<1e-4
                features.append(v);hashes.append(sha(p))
            while not (folder/'complete.json').is_file():time.sleep(1)
            proof=read(folder/'complete.json');assert obj['adapter']==proof['adapter']
            assert hashes==(proof['rows'] if old else [proof['files'][c['image']] for c in rows])
            savept(f,dict(images=[c['image'] for c in rows],features=torch.stack(features),adapter=proof['adapter'],sources=hashes))
            print(tag,name,len(rows),flush=True)
    dump(out/'index.json',{p.name:sha(p) for p in sorted(out.glob('*.pt'))})
if __name__=='__main__':main()

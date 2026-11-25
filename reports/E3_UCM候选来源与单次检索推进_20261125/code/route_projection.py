import sys,time,tarfile
import torch
from remote_common import ROOT,BASE,setup,table,rgb,read,dump,sha,savept
from eval_io import datasets
from readouts import load_model
@torch.inference_mode()
def main():
    setup();torch.set_num_threads(3);cup=load_model(False)
    ds=datasets();tables={s:table(s) for s in ('val','test')}
    for seed in 'ab':
        for sub in ('primary','official'):
            rows,feedback,field=ds[sub];folder=ROOT/'records/route'/seed/sub
            dest=folder.parent/(sub+'_features.pt');fs=[];replay={};meta=[];sig=None
            for c in rows:
                p=folder/(c['image']+'.pt')
                while not p.is_file():time.sleep(1)
                z=torch.load(p,weights_only=True);assert z['rgb']==c['rgb_sha256'] and z['original']==c[field] and z['feedback']==feedback[c['image']]
                sig=z['adapter'] if sig is None else sig;assert sig==z['adapter']
                pix=cup.preprocess(rgb(tables[c['split']],c))[None]
                def embed(h):return cup.img_embeds(pix,agent_hidden=h.float()[None])['image_feats'][0].clone()
                fs.append(embed(z['prior']))
                if 'baseline' in z:replay[c['image']]=embed(z['baseline']['prior'])
                meta.append(dict(image=c['image'],top=z['top'],attribute=z['attribute'],source=sha(p),calls=len(z['calls'])))
            while not (folder/'complete.json').exists():time.sleep(1)
            cp=read(folder/'complete.json');assert cp['files']=={r['image']:r['source'] for r in meta} and cp['adapter']==sig
            savept(dest,dict(visual=torch.stack(fs),meta=meta,baseline=replay,adapter=sig,route=cp['route']))
            archive=folder.parent/(sub+'_trajectories.tar.gz')
            with tarfile.open(archive.with_suffix('.part'),'w:gz',compresslevel=3) as tar:
                for c in rows:tar.add(folder/(c['image']+'.pt'),arcname=c['image']+'.pt')
                tar.add(folder/'complete.json',arcname='complete.json')
            archive.with_suffix('.part').replace(archive)
            dump(folder.parent/(sub+'_complete.json'),dict(features=sha(dest),archive=sha(archive),raw=sha(folder/'complete.json')))
            print(sub,seed,len(rows),flush=True)
if __name__=='__main__':main()

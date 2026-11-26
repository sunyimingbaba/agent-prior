import hashlib,io,json,sys,tarfile
from pathlib import Path
import torch
from local import ROOT,RFPO,FORM,read,sha,dump,load
def main(backup):
    idx=read(ROOT/'archives/index.json');assert sha(ROOT/'archives/execution_logs.tar.gz')==idx['logs']
    held=read(ROOT/'inputs/holdout.json');rows={t:{r['image']:r for r in v} for t,v in [('primary',read(RFPO/'inputs/panel.json')),('official',read(FORM/'inputs/ucm.json')),('hold_first',held),('hold_scene',held)]}
    report={}
    for key,meta in idx['groups'].items():
        seed,arm=key.split('_');a=ROOT/'archives'/(key+'.tar.gz');assert sha(a)==meta['sha'];archives=ROOT/'evaluate'/key
        if arm!='parent':
            folder=Path(backup)/key
            cp=read(ROOT/'models'/key/'complete.json');sig=sha(folder/'adapter/adapter_model.safetensors');assert sig==cp['final']
        else:sig=read(ROOT/'inputs/lock.json')['starting'][seed]
        mats={t:load(archives/(t+'.pt')) for t in meta['images']};ind={t:{n:i for i,n in enumerate(m['images'])} for t,m in mats.items()}
        manifests={};seen=set()
        with tarfile.open(a,'r|gz') as tf:
            for member in tf:
                t,n=member.name.split('/');raw=tf.extractfile(member).read()
                if n=='complete.json':
                    manifests[t]=json.loads(raw);assert manifests[t]==read(archives/t/n);continue
                z=torch.load(io.BytesIO(raw),weights_only=True)
                name=z['image'];r=rows[t][name];assert z['rgb']==r['rgb_sha256'] and z['adapter']==sig
                field='scene_query' if t in ('hold_scene','primary') else 'orig_query';assert z['query']==r[field]
                assert hashlib.sha256(raw).hexdigest()==manifests[t]['images'][name] and torch.equal(z['v'],mats[t]['features'][ind[t][name]])
                assert z['prior'].shape==(3584,) and torch.isfinite(z['prior']).all() and (t,name) not in seen;seen.add((t,name))
        assert len(seen)==sum(meta['images'].values())
        report[key]=dict(adapter=sig,images=meta['images'],sha=meta['sha'])
    dump(ROOT/'results/backup.json',dict(archive=report,external_models=str(Path(backup).resolve())))
if __name__=='__main__':main(sys.argv[1])

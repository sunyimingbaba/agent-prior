import hashlib,io,json,sys,tarfile
from pathlib import Path
import torch
from study import STAGE,RFPO,FORM,read,sha,dump,load
def main(backup):
    stage=STAGE
    idx=read(stage/'archives/index.json');assert sha(stage/'archives/execution_logs.tar.gz')==idx['logs']
    held=read(stage/'inputs/holdout.json');rows={t:{r['image']:r for r in v} for t,v in [('primary',read(RFPO/'inputs/panel.json')),('official',read(FORM/'inputs/ucm.json')),('hold_first',held),('hold_scene',held)]}
    repeat=read(stage/'inputs/repeat.json')
    rows.update({t:{r['image']:r for r in repeat} for t in ('repeat_scene','repeat_first')})
    report={}
    for key,meta in idx['groups'].items():
        seed,arm=key.split('_');a=stage/'archives'/(key+'.tar.gz');assert sha(a)==meta['sha'];archives=stage/'evaluate'/key
        if arm!='parent':
            folder=Path(backup)/key
            cp=read(stage/'models'/key/'complete.json');sig=sha(folder/'adapter/adapter_model.safetensors');assert sig==cp['final']
        else:sig=read(stage/'inputs/lock.json')['starting'][seed]
        mats={t:load(archives/(t+'.pt')) for t in meta['images']};ind={t:{n:i for i,n in enumerate(m['images'])} for t,m in mats.items()}
        manifests={};seen=set()
        with tarfile.open(a,'r|gz') as tf:
            for member in tf:
                t,n=member.name.split('/');raw=tf.extractfile(member).read()
                if n=='complete.json':
                    manifests[t]=json.loads(raw);assert manifests[t]==read(archives/t/n);continue
                z=torch.load(io.BytesIO(raw),weights_only=True)
                name=z['image'];r=rows[t][name];assert z['rgb']==r['rgb_sha256'] and z['adapter']==sig
                field='scene_query' if t in ('hold_scene','repeat_scene','primary') else 'orig_query';assert z['query']==r[field]
                assert hashlib.sha256(raw).hexdigest()==manifests[t]['images'][name] and torch.equal(z['v'],mats[t]['features'][ind[t][name]])
                assert z['prior'].shape==(3584,) and torch.isfinite(z['prior']).all() and (t,name) not in seen;seen.add((t,name))
        assert len(seen)==sum(meta['images'].values())
        report[key]=dict(adapter=sig,images=meta['images'],sha=meta['sha'])
    dump(stage/'results/backup.json',dict(archive=report,external_models=str(Path(backup).resolve())))
if __name__=='__main__':main(*sys.argv[1:])

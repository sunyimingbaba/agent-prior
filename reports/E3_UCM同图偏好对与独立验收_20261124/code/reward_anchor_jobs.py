"""Observe one original R2 greedy trajectory for each independent reward-panel image."""
import sys, time, tarfile, subprocess
import torch
from env_pair import ROOT, read, sha, setup, table, rgb, savept, dump, weight

def datasets():
    folder = ROOT/'reward_confirm'
    p = folder/'inputs/panel.json'
    assert sha(p) == read(folder/'inputs/lock.json')['panel']
    answers = {n:v[0] for n,v in read(folder/'records/tools.json').items()}
    return {'reward_anchor':(table(),read(p),answers,'scene_query')}

@torch.inference_mode()
def reader(arm='start'):
    assert arm in ('start','bounded')
    stage='reward_anchor' if arm=='start' else 'reward_replica'
    from readouts import load_model
    setup()
    model = load_model(False)
    t, rows, _, field = datasets()['reward_anchor']
    index = {}
    for seed in 'ab':
        tag = seed+'_'+arm
        folder = ROOT/'pilot/greedy'/tag/'reward_anchor'
        p = folder/'complete.json'
        while not p.exists():time.sleep(2)
        evidence = read(p)
        adapter = weight(seed) if arm=='start' else read(ROOT/'inputs/nomination_bounded.json')['adapters'][seed]
        assert evidence['adapter'] == adapter
        names, hashes, features = [], [], []
        for c in rows:
            name = c['image']
            state = folder/(name+'.pt')
            h = sha(state)
            z = torch.load(state,weights_only=True)
            assert h==evidence['files'][name] and z['adapter']==adapter and z['rgb']==c['rgb_sha256'] and z['query']==c[field]
            v = model.img_embeds(model.preprocess(rgb(t,c))[None],agent_hidden=z['prior'].float()[None])['image_feats'][0]
            assert torch.isfinite(v).all() and abs(v.norm().item()-1)<1e-4
            names.append(name);hashes.append(h);features.append(v)
        savept(ROOT/'pilot/vectors'/(tag+'_reward_anchor.pt'),dict(images=names,features=torch.stack(features),adapter=adapter,sources=hashes))
        dest=ROOT/'archives'/(stage+'_'+tag+'.tar.gz')
        with tarfile.open(dest.with_suffix('.part'),'w:gz',compresslevel=3) as tf:tf.add(folder,arcname=tag+'/reward_anchor')
        dest.with_suffix('.part').replace(dest)
        index[dest.name]=sha(dest)
    dump(ROOT/'archives'/(stage+'_index.json'),index)

if __name__=='__main__':
    if sys.argv[1]=='read':reader(sys.argv[2] if len(sys.argv)>2 else 'start')
    elif sys.argv[1]=='deploy':
        arm=sys.argv[2] if len(sys.argv)>2 else 'start'
        for seed in 'ab':subprocess.run([sys.executable,__file__,'one',seed,arm],check=True)
    elif sys.argv[1]=='one':
        import deploy_pair
        seed=sys.argv[2];assert seed in ('a','b')
        deploy_pair.datasets=datasets
        arm=sys.argv[3];assert arm in ('start','bounded')
        if arm=='bounded':assert read(ROOT/'pilot'/(seed+'_'+arm)/'complete.json')['final']==read(ROOT/'inputs/nomination_bounded.json')['adapters'][seed]
        deploy_pair.main(seed,arm)
    else:raise ValueError(sys.argv[1])

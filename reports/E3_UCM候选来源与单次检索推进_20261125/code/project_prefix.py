import sys,time
import torch
from remote_common import ROOT,BASE,setup,table,rgb,read,sha,savept,dump
from readouts import load_model,scorer_for
@torch.inference_mode()
def main(shard):
    setup();torch.set_num_threads(3);data=table();m=load_model(False)
    proxy=scorer_for(load_model(True),BASE/'relation_candidates_20261120/gallery/train.pt')
    for seed in 'ab':
        for tag in ('second','third'):
            rows=read(ROOT/'inputs'/(tag+'.json'))[shard::2];base=ROOT/'records/prefix'/tag/seed;src=base/'raw';out=base/'features';out.mkdir(exist_ok=True,parents=True)
            for j,c in enumerate(rows):
                nm=c['image'];p=src/(nm+'.pt');target=out/p.name
                while not p.exists():time.sleep(1)
                if target.exists():
                    assert torch.load(target,weights_only=True)['source']==sha(p);continue
                z=torch.load(p,weights_only=True);assert z['rgb']==c['rgb_sha256']
                cache={};pix=m.preprocess(rgb(data,c))[None];modes={}
                if tag=='second':
                    p0=ROOT/'records/second'/seed/'raw'/p.name;raw=torch.load(p0,weights_only=True)['modes']['scene']
                    feats=torch.load(ROOT/'records/second'/seed/'features'/p.name,weights_only=True)
                    assert feats['source']==sha(p0)
                    for h,v,r in zip(raw['hidden'],feats['modes']['scene']['visual'],feats['modes']['scene']['ranks']):
                        cache[h.float().numpy().tobytes()]=(v,r)
                for arm,obj in z['modes'].items():
                    fs=[];rk=[]
                    for state in obj['hidden']:
                        h=state.float();key=h.numpy().tobytes()
                        if key not in cache:
                            feature=m.img_embeds(pix,agent_hidden=h[None])['image_feats'][0]
                            assert torch.isfinite(feature).all() and abs(feature.norm().item()-1)<1e-4
                            cache[key]=(feature.clone(),int(proxy.score(h.tolist(),c['row'])[0]))
                        v,r=cache[key];fs.append(v);rk.append(r)
                    modes[arm]=dict(visual=torch.stack(fs),ranks=rk)
                savept(target,dict(image=nm,adapter=z['adapter'],rgb=z['rgb'],source=sha(p),modes=modes))
                if (j+1)%15==0:print(tag,seed,j+1,len(rows),flush=True)
            dump(out/('complete_'+str(shard)+'.json'),dict(files={c['image']:sha(out/(c['image']+'.pt')) for c in rows}))
if __name__=='__main__':main(int(sys.argv[1]))

"""Reuse exact current-run CUP vectors when all fixed-first hidden states are already evaluated."""
import torch
from remote_common import ROOT,read,savept,dump,sha
@torch.inference_mode()
def main():
    torch.set_num_threads(1);stats={};seed='b';rows=read(ROOT/'inputs/second.json')
    base=ROOT/'records/second'/seed;out=ROOT/'records/prefix/second'/seed
    (out/'features').mkdir(exist_ok=True)
    source=read(out/'raw/complete.json');by_file=read(base/'raw/complete.json')['files']
    recorded={k:v for shard in (0,1) for k,v in read(base/'features'/('complete_'+str(shard)+'.json'))['files'].items()}
    for c in rows:
        nm=c['image'];p=out/'raw'/(nm+'.pt');f=base/'features'/p.name
        assert sha(p)==source['files'][nm] and sha(f)==recorded[nm]
        z=torch.load(p,weights_only=True);oldp=base/'raw'/p.name;old=torch.load(oldp,weights_only=True);feature=torch.load(f,weights_only=True)
        assert z['adapter']==feature['adapter']==old['adapter'] and z['rgb']==feature['rgb']==old['rgb']==c['rgb_sha256']
        assert feature['source']==sha(oldp)==by_file[nm]
        cache={}
        for k,record in old['modes'].items():
            for v,h,r in zip(feature['modes'][k]['visual'],record['hidden'],feature['modes'][k]['ranks']):
                key=h.float().numpy().tobytes()
                if key in cache:assert torch.equal(v,cache[key][0]) and r==cache[key][1]
                cache[key]=(v,r)
        keys=[h.float().numpy().tobytes() for h in z['modes']['locked']['hidden']]
        missing=set(keys)-set(cache);stats[nm]=dict(unique=len(set(keys)),missing=len(missing))
        if not missing:
            vals=[cache[key] for key in keys];target=out/'features'/p.name
            pack=dict(image=nm,adapter=z['adapter'],rgb=z['rgb'],source=source['files'][nm],
                      modes=dict(locked=dict(visual=torch.stack([v for v,r in vals]),ranks=[r for v,r in vals])))
            if not target.exists():savept(target,pack)
            else:
                m=torch.load(target,weights_only=True)['modes']['locked'];assert torch.equal(m['visual'],pack['modes']['locked']['visual']) and m['ranks']==pack['modes']['locked']['ranks']
    dump(ROOT/'results/second_exact_reuse.json',dict(images=len(rows),fully_reused=sum(x['missing']==0 for x in stats.values()),rows=stats))
if __name__=='__main__':main()

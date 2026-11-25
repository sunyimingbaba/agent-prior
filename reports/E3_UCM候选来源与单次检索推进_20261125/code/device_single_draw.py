"""Bounded partial read of both distinct single-draw trials on the available seed-a held-out detail panel."""
import time,io,hashlib
import numpy as np
import torch
from PIL import Image
from device_readout import STORE,ROOT,RFPO,read,sha,dump,load,load_device,arrow,calculate
from run_stage import boot
@torch.inference_mode()
def main():
    m,cv,_=load_device();rows=read(RFPO/'inputs/panel.json');ts={k:arrow(k) for k in ('val','test')}
    base=STORE/'local_projection/stochastic/a/primary';ref=load(RFPO/'vectors/a_deployed_primary.pt');vs=[];proof={}
    for c in rows:
        nm=c['image'];p=base/(nm+'.pt')
        while not p.exists():
            if time.time()>=1795593180:raise SystemExit(75)
            time.sleep(1)
        z=load(p);assert z['rgb']==c['rgb_sha256'] and ref['adapter']==z['adapter'] and z['query']==c['scene_query']
        table=ts[c['split']];im=Image.open(io.BytesIO(table['image'][c['row']].as_py())).convert('RGB')
        assert hashlib.sha256(im.tobytes()).hexdigest()==z['rgb']
        x=cv(im)[None].to('mps');hs=torch.stack([z['baseline']]+[r['state'] for r in z['draws']]).float().to('mps')
        assert len(hs)==3
        vs.append(m(x.repeat(3,1,1,1),hs).cpu());proof[nm]=sha(p)
    while not (base/'complete.json').is_file():
        if time.time()>=1795593180:raise SystemExit(75)
        time.sleep(1)
    assert proof==read(base/'complete.json')['files'];v=torch.stack(vs)
    old=calculate('primary',ref['features'],rows);results=[calculate('primary',v[:,i],rows) for i in range(3)]
    assert all(np.array_equal(x,results[0][k]) for k,x in old.items())
    labels=[r['pair'] for r in rows];out={}
    for k in old:
        a,b,c=[z[k] for z in results];delta=(b+c)/2-a
        out[k]=dict(greedy=float(a.mean()),draw1=float(b.mean()),draw2=float(c.mean()),change=boot(delta,np.ones(len(delta)),labels))
    dump(ROOT/'results/device_stochastic_seed_a_primary.json',dict(seed='a',images=len(rows),raw=sha(base/'complete.json'),metrics=out))
    root=ROOT/'records/device';torch.save(dict(vectors=v,images=ref['images']),root/'stochastic_seed_a_primary.pt')
    from package_deploy import package
    package('stochastic','a','primary')
if __name__=='__main__':main()

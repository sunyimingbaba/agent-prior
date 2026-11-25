"""Stream an independent second-device readout for the historical 448-image RSITMD gallery."""
import hashlib,io,time
import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from device_readout import STORE,ASSETS,ROOT,read,sha,dump,FORM,load,load_device
from transfer_scores import counts,summary,scalar_values
def wait(p):
    while not p.is_file():
        if time.time()>=1795593180:raise SystemExit(75)
        time.sleep(2)
@torch.inference_mode()
def main():
    m,convert,proof=load_device();name='rsitmd'
    f=ASSETS/'arrow/rsitmd/rsitmd_caption_karpathy_label_test.arrow'
    assert sha(f)==read(FORM/'inputs/audit.json')['files']['rsitmd_test']
    t=pa.ipc.open_file(pa.memory_map(str(f))).read_all();rows=read(FORM/'inputs/rsitmd.json')
    gal=load(FORM/'results/rsitmd_text.pt');route=read(ROOT/'inputs/route.json');expected=read(FORM/'results/paired_results.json')['rsitmd']
    groups={k:[] for k in ('native','alternatives')};out={};meta={}
    for seed in 'ab':
        folder=STORE/'local_projection/cross'/seed;ref=load(FORM/'results'/('rsitmd_'+seed+'_R2.pt'))
        features=[];hashes={};a0=None
        for j,c in enumerate(rows):
            nm=c['image'];p=folder/(nm+'.pt');wait(p);z=load(p);hashes[nm]=sha(p)
            assert z['adapter']==ref['adapter'] and z['rgb']==c['rgb_sha256'] and z['image']==nm
            entry=route['categories'].get(z['top']);assert z['query']==c['orig_query']+(' '+entry['question'] if entry else '')
            assert t['path'][c['row']].as_py()==nm
            im=Image.open(io.BytesIO(t['image'][c['row']].as_py())).convert('RGB')
            assert hashlib.sha256(im.tobytes()).hexdigest()==z['rgb']
            pix=convert(im)[None].to('mps')
            states=[z['prior']]+([z['baseline']] if 'baseline' in z else [])
            y=m(pix.repeat(len(states),1,1,1),torch.stack(states).float().to('mps')).cpu()
            features.append(y[0])
            if len(states)>1:a0=float((y[1]-ref['features'][j]).abs().max())
            if (j+1)%100==0:print(seed,j+1,len(rows),flush=True)
        cp=folder/'complete.json';wait(cp);assert hashes==read(cp)['files'] and a0 is not None and a0<3e-4
        new=torch.stack(features);meta[seed]=dict(max_baseline_embedding_difference=a0,images=len(rows),raw=sha(cp))
        dest=ROOT/'records/device';dest.mkdir(exist_ok=True);torch.save(dict(vectors=new,ids=[r['image'] for r in rows],adapter=ref['adapter'],trace=proof['trace']),dest/(seed+'_rsitmd.pt'))
        for mode in groups:
            ids,b,ds=counts(ref['features'],gal,mode);ids2,n,ds2=counts(new,gal,mode)
            assert np.array_equal(ids,ids2) and np.array_equal(ds,ds2)
            assert abs(100*scalar_values(b.sum(0),np.asarray(len(ds)),np.asarray(ds.sum()))[-1]-expected['all_'+mode]['levels'][seed+'_R2']['mR'])<1e-10
            groups[mode].append((b,n))
            B=np.stack([v[0] for v in groups[mode]]).mean(0);N=np.stack([v[1] for v in groups[mode]]).mean(0)
            out[mode]=dict(eligible=len(ids),captions=int(ds.sum()),before=summary(B,ds),after=summary(N,ds),change=summary(N-B,ds))
        dump(ROOT/'results'/('device_cross_'+('both' if seed=='b' else 'seed_a')+'.json'),dict(seeds=meta,metrics=out))
if __name__=='__main__':main()

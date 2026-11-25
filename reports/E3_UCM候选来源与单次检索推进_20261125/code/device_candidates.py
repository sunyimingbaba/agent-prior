"""Independent device shadow of the full matched panel, checked against available canonical CPU reads."""
import io,hashlib,time
from PIL import Image
import numpy as np
import torch
from device_readout import STORE,ROOT,read,sha,dump,load,load_device,arrow
from when_to_sample import archive_raw,compare,PRIOR,bank,image_folds
def move_device(path):
    m=torch.jit.load(str(path),map_location='cpu').eval().to('mps')
    for sub in m.modules():
        for method in sub._c._method_names():
            for node in sub._c._get_method(method).graph.findAllNodes('prim::Constant'):
                if str(node.output().type())=='Device' and 'value' in node.attributeNames() and node.s('value')=='cpu':node.s_('value','mps:0')
    return m
@torch.inference_mode()
def main():
    native,convert,_=load_device();root=STORE/'proxy';p=root/'proxy_trace.pt';receipt=read(root/'proof.json')
    assert sha(p)==receipt['trace'] and sha(root/'gallery.pt')==receipt['gallery']
    proxy=move_device(p);gal=load(root/'gallery.pt').float()
    b=bank(PRIOR/'guard');rows=read(ROOT/'inputs/third.json');t=arrow('train');out=[];checks=[]
    dest=ROOT/'records/device/candidates';dest.mkdir(exist_ok=True,parents=True)
    for s in 'ab':
        folder=ROOT/'records/prefix/third'/s
        for j,(c,z) in enumerate(zip(rows,archive_raw(folder,rows),strict=True)):
            nm=c['image'];source=folder/'features'/(nm+'.pt')
            if s=='a' and j%12!=0:
                f=load(source)['modes']
            else:
                cache={};vectors=[];ids={}
                for mode,a in z['modes'].items():
                    ids[mode]=[]
                    for h in a['hidden']:
                        key=h.float().numpy().tobytes()
                        if key not in cache:cache[key]=len(vectors);vectors.append(h.float())
                        ids[mode].append(cache[key])
                im=Image.open(io.BytesIO(t['image'][c['row']].as_py())).convert('RGB')
                assert hashlib.sha256(im.tobytes()).hexdigest()==c['rgb_sha256']
                pix=convert(im)[None].to('mps');vs=[];rs=[]
                for block in torch.stack(vectors).split(4):
                    h=block.to('mps');x=pix.repeat(len(h),1,1,1)
                    vs.extend(native(x,h).cpu());scores=proxy(torch.zeros_like(x),h).cpu()@gal.T
                    rs.extend((scores>scores[:,c['row'],None]).sum(1).tolist())
                f={mode:dict(visual=torch.stack([vs[i] for i in seq]),ranks=[rs[i] for i in seq]) for mode,seq in ids.items()}
                refpath=source if source.is_file() else STORE/'local_projection/partial_third_b'/(nm+'.pt')
                if refpath.is_file():
                    before=load(refpath)['modes']
                    checks.append(dict(seed=s,image=nm,max_visual=max(float((v['visual']-before[k]['visual']).abs().max()) for k,v in f.items()),
                        different_ranks=sum(np.count_nonzero(np.asarray(v['ranks'])!=before[k]['ranks']) for k,v in f.items())))
            item=dict(image=nm,pair=c['pair'],seed=s);anchor=f['locked']['visual'][-1].double().numpy()
            for arm,v in f.items():
                vis=v['visual'].double().numpy();vis[-1]=anchor;folds,_=image_folds(c,s,b,vis,np.asarray(v['ranks'][:16]))
                item[arm]={k:np.mean([y['baseline'][q] for y in folds]) for k,q in [('joint','hit'),('mrr','mrr'),('detail','detail')]}
                item[arm]['opportunity']=np.mean([bool(y['good']) for y in folds])
            out.append(item)
            if (j+1)%30==0:print(s,j+1,flush=True)
    assert max(x['max_visual'] for x in checks)<3e-4
    dump(ROOT/'results/device_timing_third.json',dict(measures=compare(out),checks=checks));dump(dest/'third_rows.json',out)
if __name__=='__main__':main()

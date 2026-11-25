"""Swap only the learned change between photos, retaining each photo's own original prior."""
from collections import defaultdict
import sys,torch
from head_fit import ROOT,OUT,reader,pack,read,load,save,sha,head,params_on_device,key,merge
@torch.inference_mode()
def main(seed):
    torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    l=read(ROOT/'inputs/head_nomination.json');p=OUT/seed/'hard'/('e%d.pt'%l['epoch']);assert sha(p)==l['checkpoints'][seed]['hard']
    base,w=params_on_device(load(p));train=merge(pack('first',seed),pack('second',seed))
    means={}
    for f in ('ordinary','scene'):
        x=train[f].cuda();means[f]=(head(x,base,w)-x).mean(0).cpu()
    del train;m=reader()
    for tag in ('fresh','extension','primary','official'):
        d=pack(tag,seed);field='ordinary' if tag=='official' else 'scene';x=d[field].cuda()
        delta=head(x,base,w)-x;rows=d['rows'];groups=defaultdict(list);mapping=list(range(len(rows)))
        for i,r in enumerate(rows):groups[(int(r['image'][:-4])-1)//100 if tag=='official' else r['pair']].append(i)
        for group in groups.values():
            assert len(group)>1
            ids=sorted(group,key=lambda i:key('single-shift',rows[i]['image']));order=ids[1:]+ids[:1]
            for i,j in zip(ids,order):mapping[i]=j
        assert sorted(mapping)==list(range(len(rows))) and all(i!=v for i,v in enumerate(mapping))
        out=OUT/'test'/tag/seed
        for name,shift in [('swapped',delta[mapping]),('constant',means[field].cuda()[None].expand_as(x))]:
            vec=[]
            for j in range(0,len(x),12):vec.append(m(d['pixels'][j:j+12].cuda(),(x+shift)[j:j+12]).cpu())
            save(out/(name+'.pt'),dict(images=[r['image'] for r in rows],mapping=mapping,adapter=d['adapter'],nomination=sha(ROOT/'inputs/head_nomination.json'),**{field:torch.cat(vec)}))
        print(tag,seed,flush=True)
if __name__=='__main__':main(sys.argv[1])

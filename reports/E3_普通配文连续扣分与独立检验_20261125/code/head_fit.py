"""Train only a bounded residual on the output of one real greedy generation."""
import sys,hashlib
from collections import defaultdict
import numpy as np,torch
import torch.nn.functional as F
from head_data import ROOT,reader,pack,save,load,read,dump,sha
OUT=ROOT/'head';OUT.mkdir(exist_ok=True)
def key(*x):return hashlib.sha256('|'.join(map(str,x)).encode()).hexdigest()
def head(x,base,w):
    z=((x-base['mu'])@base['pc']/base['std']).clamp(-5,5)
    delta=0.5*x.square().mean(1,keepdim=True).sqrt()*torch.tanh(z@w['weight']+w['bias'])
    return x+delta

def compress(h):
    mu=h.mean(0);u,s,v=torch.linalg.svd((h-mu).double(),full_matrices=False)
    return dict(mu=mu,pc=v[:32].T.float().contiguous(),std=(s[:32]/np.sqrt(len(h)-1)).float().clamp_min(1e-5))
def native_loss(v,pos,native):
    # Other positive paraphrases and identical text columns are excluded from negatives in both directions.
    logits=40*(v@native.T);alt=pos[:,1:];ok=alt!=pos[:,0,None];none=~ok.any(1);ok[none,0]=True
    valid=torch.ones_like(logits,dtype=torch.bool);valid.scatter_(1,pos,False)
    neg=logits.masked_fill(~valid,float('-inf')).logsumexp(1)
    target=logits.gather(1,alt)
    fwd=F.softplus(neg[:,None]-target)
    # The all-image denominator is computed on this minibatch. All other true owners are masked.
    rev=logits.masked_fill(~valid,float('-inf')).logsumexp(0)
    both=(fwd+F.softplus(rev[alt]-target))*.5
    return (both*ok).sum(1).div(ok.sum(1)).mean()
def merge(a,b):
    for k in ('scene','pixels','ordinary','ab','cd','pos'):a[k]=torch.cat([a[k],b[k]])
    assert torch.equal(a['native'],b['native']) and a['adapter']==b['adapter']
    a['rows']+=b['rows'];return a

@torch.no_grad()
def embeddings(m,d,params,device='cuda',size=12):
    base,w=params if params else (None,None)
    out={}
    for field in ('scene','ordinary'):
        x=d[field]
        if x is None:continue
        vec=[]
        for j in range(0,len(x),size):
            h=x[j:j+size].to(device);p=d['pixels'][j:j+size].to(device)
            if w is not None:h=head(h,base,w)
            vec.append(m(p,h).cpu())
        out[field]=torch.cat(vec)
    return out
def params_on_device(z):
    return [{k:v.cuda() for k,v in item.items()} for item in (z['base'],z['w'])]
def fit(seed):
    torch.manual_seed(20261125);torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=reader();data=merge(pack('first',seed),pack('second',seed));dev=pack('third',seed);assert data['adapter']==dev['adapter']
    assert {r['rgb_sha256'] for r in data['rows']}.isdisjoint(r['rgb_sha256'] for r in dev['rows'])
    base={k:v.cuda() for k,v in compress(torch.cat([data['scene'],data['ordinary']])).items()}
    groups=defaultdict(list)
    for j,r in enumerate(data['rows']):groups[r['pair']].append(j)
    assert all(len(v)==2 for v in groups.values())
    pub=OUT/seed;pub.mkdir(exist_ok=True)
    prefix=pub/'base_dev.pt'
    if not prefix.exists():save(prefix,embeddings(m,dev,None))
    gallery=data['native'].cuda();logs=[]
    for arm in ('ordinary','hard'):
        folder=pub/arm;folder.mkdir(exist_ok=True)
        if (folder/'done.json').exists():continue
        w={'weight':torch.zeros(32,3584,device='cuda',requires_grad=True),'bias':torch.zeros(3584,device='cuda',requires_grad=True)}
        opt=torch.optim.AdamW(list(w.values()),lr=.015)
        logs=[]
        for epoch in range(1,7):
            pairs=sorted(groups,key=lambda p:key('single-prior',seed,epoch,p))
            for start in range(0,len(pairs),4):
                ids=sum((groups[k] for k in pairs[start:start+4]),[]);p=data['pixels'][ids].cuda()
                x=torch.cat([data['ordinary'][ids],data['scene'][ids]]).cuda();hs=head(x,base,w)
                v=m(torch.cat([p,p]),hs);o,s=v.chunk(2);cs=data['pos'][ids].cuda()
                ordinary=(native_loss(o,cs,gallery)+native_loss(s,cs,gallery))*.5
                fine=F.softplus(-50*torch.einsum('bd,bkd->bk',s,data['ab'][ids].cuda())).mean()
                loss=ordinary+(.75*fine if arm=='hard' else 0.)
                opt.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(list(w.values()),5.,error_if_nonfinite=True);opt.step()
                logs.append(dict(epoch=epoch,pairs=pairs[start:start+4],native=ordinary.item(),fine=fine.item(),norm=norm.item(),
                                 relative_shift=((hs-x).norm(dim=1)/x.norm(dim=1)).mean().item()))
            if epoch in (1,3,6):
                cp=folder/('e%d.pt'%epoch)
                save(cp,dict(base={k:v.cpu() for k,v in base.items()},w={k:v.detach().cpu() for k,v in w.items()},adapter=data['adapter'],epoch=epoch,arm=arm))
                save(folder/('e%d_dev.pt'%epoch),embeddings(m,dev,(base,w)))
            print(seed,arm,epoch,logs[-1],flush=True)
        dump(folder/'steps.json',logs);dump(folder/'done.json',dict(steps=len(logs),adapter=data['adapter'],checkpoints={e:sha(folder/('e%d.pt'%e)) for e in (1,3,6)}))
def evaluate(seed):
    torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    nomination=read(ROOT/'inputs/head_nomination.json');epoch=nomination['epoch'];m=reader()
    for tag in ('fresh','extension','primary','official','cross'):
        folder=OUT/'test'/tag/seed;folder.mkdir(exist_ok=True,parents=True)
        d=pack(tag,seed)
        for arm in ('baseline','ordinary','hard'):
            f=folder/(arm+'.pt')
            if f.exists():continue
            if arm=='baseline':params=None
            else:
                cp=OUT/seed/arm/('e%d.pt'%epoch);assert sha(cp)==nomination['checkpoints'][seed][arm]
                z=load(cp);assert z['adapter']==d['adapter'];params=params_on_device(z)
            save(f,dict(images=[c['image'] for c in d['rows']],adapter=d['adapter'],nomination=sha(ROOT/'inputs/head_nomination.json'),**embeddings(m,d,params)))
        print(seed,tag,len(d['rows']),flush=True)
if __name__=='__main__':{'fit':fit,'test':evaluate}[sys.argv[1]](sys.argv[2])

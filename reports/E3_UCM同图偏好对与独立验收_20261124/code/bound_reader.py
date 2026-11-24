import os,sys,json,hashlib,base64
pipe=os.fdopen(os.dup(1),'w',buffering=1);os.dup2(2,1)
import numpy as np,torch,torch.nn.functional as F
from env_pair import ROOT,OLD,setup,rows,read,dump,sha,table,rgb
from readouts import load_model
def main(mode,seed):
    setup();m=load_model(False);t=table();ps=rows();bank={};data={}
    for sub in ('','confirm'):
        b=torch.load(OLD/sub/'records/text.pt',weights_only=True)
        even=torch.tensor([int(hashlib.sha256(','.join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in b['native_tokens'].tolist()])
        for nm,ix in b['maps'].items():
            sel=even.clone();sel[ix['native']]=False
            data[nm]=(sub,ix,sel)
        bank[sub]=b
    pix={}
    def scores(n,h):
        sub,ix,sel=data[n];b=bank[sub]
        if n not in pix:pix[n]=m.preprocess(rgb(t,ps[n]))[None]
        v=m.img_embeds(pix[n],agent_hidden=h[None])['image_feats'][0]
        s=b['native']@v;hs=b['hard']@v
        ab=torch.stack([hs[ix[k][0]]-hs[ix[k][1]] for k in 'AB'])
        return s[ix['native'][0]],s[sel],ab
    ref=ROOT/'records'/(seed+'_bound_ref.json')
    if mode=='ref':
        p=ROOT/'records'/(seed+'_bound_source.pt');z=torch.load(p,weights_only=True);xs=[];rr={}
        with torch.no_grad():
            for nm,fields in z['rows'].items():
                rr[nm]={}
                for field,h in fields.items():
                    pos,neg,ab=scores(nm,h.float());rank=int((neg>pos).sum())+1;cap=next((k for k in (1,5,10) if rank<=k),rank)
                    rr[nm][field]=dict(rank=rank,cap=cap,AB=ab.tolist())
                    if field=='scene_query':xs+=ab.abs().tolist()
        dump(ref,dict(adapter=z['adapter'],states=sha(p),target=float(np.median(xs)),images=rr));return
    rr=read(ref);print('READY',file=pipe)
    for line in sys.stdin:
        req=json.loads(line);nm=req['image'];field=req['field'];z=torch.tensor(np.frombuffer(base64.b64decode(req['data']),dtype=np.float32).copy(),requires_grad=True)
        pos,neg,ab=scores(nm,z);cap=rr['images'][nm][field]['cap'];rank=int((neg.detach()>pos.detach()).sum())+1
        threshold=neg.topk(min(cap,len(neg))).values[-1] if cap<=len(neg) else pos.detach()
        native=F.relu(50*(threshold-pos));hard=F.relu(50*(rr['target']-ab)).mean();loss=(native+req['hard']*hard)/4
        g=torch.autograd.grad(loss,z)[0];assert torch.isfinite(g).all()
        print(json.dumps(dict(grad=base64.b64encode(g.numpy().tobytes()).decode(),native=float(native.detach()),hard=float(hard.detach()),AB=ab.detach().tolist(),native_gap=float((pos-neg.max()).detach()),rank=rank,cap=cap,gradnorm=float(g.norm()))),file=pipe)
if __name__=='__main__':main(*sys.argv[1:])

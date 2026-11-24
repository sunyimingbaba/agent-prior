import os,sys,json,base64,hashlib
pipe=os.fdopen(os.dup(1),'w',buffering=1);os.dup2(2,1)
import numpy as np,torch,torch.nn.functional as F
from env_pair import ROOT,OLD,setup,rows,table,rgb
from readouts import load_model
def main():
    setup();m=load_model(False);banks={};ps=rows();t=table();px={}
    for sub in ('','confirm'):
        b=torch.load(OLD/sub/'records/text.pt',weights_only=True)
        neg=torch.tensor([int(hashlib.sha256(','.join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in b['native_tokens'].tolist()])
        banks[sub]=(b,neg)
    src={c['image']:s for s in ('','confirm') for c in json.loads((OLD/s/'inputs/panel.json').read_bytes())}
    print('READY',file=pipe)
    for line in sys.stdin:
        req=json.loads(line);nm=req['image'];bank,ne=banks[src[nm]];ix=bank['maps'][nm];neg=ne.clone();neg[ix['native']]=False
        if nm not in px:px[nm]=m.preprocess(rgb(t,ps[nm]))[None]
        z=torch.tensor(np.frombuffer(base64.b64decode(req['data']),dtype='<f4').copy(),requires_grad=True)
        v=m.img_embeds(px[nm],agent_hidden=z[None])['image_feats'][0];sim=bank['native']@v
        top=sim[neg].topk(8).values;nat=F.softplus(torch.logsumexp(50*top,0)-np.log(8.)-50*sim[ix['native'][0]])
        hs=bank['hard']@v;ab=torch.stack([hs[ix[k][0]]-hs[ix[k][1]] for k in 'AB']);hard=F.softplus(-50*ab).mean()
        loss=(nat+float(req['hard'])*hard)/4;grad=torch.autograd.grad(loss,z)[0]
        assert torch.isfinite(grad).all()
        print(json.dumps(dict(grad=base64.b64encode(grad.numpy().tobytes()).decode(),native=float(nat.detach()),hard=float(hard.detach()),AB=ab.detach().tolist(),native_gap=float((sim[ix['native'][0]]-top.max()).detach()),gradnorm=float(grad.norm()))),file=pipe)
if __name__=='__main__':main()

import json,numpy as np,torch
from env_pair import ROOT,OLD,read,dump,setup,rows,table,rgb
from readouts import load_model
def main():
    setup();m=load_model(False);ps=rows();t=table();rs=read(ROOT/'inputs/candidates.json')['a'];out=[]
    for r in [next(r for r in rs if r['pairs'] and r['side']==v) for v in (0,1)]:
        n=r['image'];base=OLD/r['source'];bank=torch.load(base/'records/text.pt',weights_only=True)
        rec=torch.load(base/'records/a'/(n+'.pt'),weights_only=True);h=rec['hidden'][0].float().requires_grad_();ix=bank['maps'][n]
        px=m.preprocess(rgb(t,ps[n]))[None]
        def f(h):
            s=bank['hard']@m.img_embeds(px,agent_hidden=h[None])['image_feats'][0];return torch.stack([torch.nn.functional.softplus(-50*(s[ix[k][0]]-s[ix[k][1]])) for k in 'AB']).mean()
        loss=f(h);g=torch.autograd.grad(loss,h)[0];direction=g/g.norm();eps=1e-3*float(h.detach().norm())
        with torch.no_grad():fd=float((f(h+eps*direction)-f(h-eps*direction))/(2*eps))
        rel=abs(fd-float(g.norm()))/float(g.norm());assert rel<.1 and fd>0
        out.append(dict(image=n,loss=float(loss.detach()),analytic=float(g.norm()),finite_difference=fd,relative_error=rel))
    dump(ROOT/'results/gradient_check.json',out)
if __name__=='__main__':main()

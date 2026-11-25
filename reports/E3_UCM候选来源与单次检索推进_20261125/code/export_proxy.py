"""Independently export the frozen proxy visual encoder for cross-device rank checks."""
import torch
from remote_common import ROOT,BASE,setup,sha,dump
from readouts import load_model,scorer_for
class ProxyVisual(torch.nn.Module):
    def __init__(self,m):
        super().__init__()
        for k in ('agent_proj','cluster_mapping','visual_prompt_proj','visual_prompt_embeddings','visual_prompt_proj_gather'):setattr(self,k,getattr(m,k))
        self.n=m.prompt_length;self.visual=m.model.visual
    def forward(self,x,h):
        p=self.agent_proj(h).unsqueeze(1).repeat(1,self.n,1);r=self.cluster_mapping(p).clamp(0,1)
        rand=self.visual_prompt_proj(self.visual_prompt_embeddings).expand(x.shape[0],-1,-1)
        v=self.visual(x,self.visual_prompt_proj_gather((1-r)*p+r*rand))
        return v/v.norm(dim=1,keepdim=True)
@torch.inference_mode()
def main():
    setup();torch.set_num_threads(2);m=load_model(True);w=ProxyVisual(m).eval()
    root=ROOT/'scratch/proxy_export';root.mkdir(parents=True,exist_ok=True);p=root/'proxy_trace.pt'
    src=BASE/'relation_candidates_20261120/gallery/train.pt';proxy=scorer_for(m,src)
    torch.manual_seed(314);x=torch.zeros(1,3,224,224);h=torch.randn(1,3584);v=w(x,h)
    sims=v@proxy.image_features.T
    assert int((sims>sims[0,15]).sum())==proxy.score(h[0].tolist(),15)[0]
    tr=torch.jit.trace(w,(x,h),check_trace=False);assert torch.allclose(tr(x,h),v,rtol=0,atol=1e-7)
    tr.save(str(p));f=root/'gallery.pt';torch.save(proxy.image_features,f)
    dump(root/'proof.json',dict(trace=sha(p),gallery=sha(f),source=sha(src)))
if __name__=='__main__':main()

"""Export only the frozen CUP image branch for an independent local deployment readout."""
from pathlib import Path
import torch
from remote_common import ROOT,setup,dump,sha
from readouts import load_model
class NativeVisual(torch.nn.Module):
    def __init__(self,m):
        super().__init__()
        for name in ('agent_proj','cluster_mapping','visual_prompt_embeddings','visual_prompt_proj',
                     'visual_prompt_proj_gather','adapter_img','adapter_img_mapping'):
            setattr(self,name,getattr(m,name))
        self.n=m.prompt_length;self.visual=m.model.visual
    def forward(self,x,h):
        p=self.agent_proj(h).unsqueeze(1).repeat(1,self.n,1);r=self.cluster_mapping(p).clamp(0,1)
        rand=self.visual_prompt_proj(self.visual_prompt_embeddings).expand(x.shape[0],-1,-1)
        p=self.visual_prompt_proj_gather((1-r)*p+r*rand)
        v=self.visual(x,p);a=self.adapter_img(v);g=self.adapter_img_mapping(a).clamp(0,.1)
        z=g*a+(1-g)*v
        return z/z.norm(dim=1,keepdim=True)
@torch.inference_mode()
def main():
    setup();torch.set_num_threads(2);m=load_model(False);w=NativeVisual(m).eval()
    box=ROOT/'scratch/export';box.mkdir(exist_ok=True,parents=True);p=box/'cup_native_trace.pt'
    if p.exists() and (box/'proof.json').exists():return
    torch.manual_seed(20261125);x=torch.randn(1,3,224,224);h=torch.randn(1,3584)
    native=m.img_embeds(x,agent_hidden=h)['image_feats'];assert torch.equal(w(x,h),native)
    traced=torch.jit.trace(w,(x,h),check_trace=False)
    assert torch.allclose(traced(x,h),native,rtol=0,atol=1e-7)
    traced.save(str(p.with_suffix('.part')));p.with_suffix('.part').replace(p)
    print('PREPROCESS',repr(m.preprocess),flush=True)
    dump(box/'proof.json',dict(trace=sha(p),transform=repr(m.preprocess),reader='09ef20e0b354efbf8a7d4d49105cd92e1ae9412b3ea58820961265a065a874dc'))
if __name__=='__main__':main()

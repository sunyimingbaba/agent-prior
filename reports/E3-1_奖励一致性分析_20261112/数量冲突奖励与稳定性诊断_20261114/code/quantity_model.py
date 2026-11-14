import json
import sys
import torch
from milestone_common import ROOT,PLAN,ASSETS,SOURCE,private_clip,sha,case_images

def load_final_model():
    clip=private_clip()
    from run_frozen import MyBackboneFinetuning
    from vilt.modules.vilt_module import ViLTransformerSS
    setattr(sys.modules['__main__'],'MyBackboneFinetuning',MyBackboneFinetuning)
    path=ASSETS/'ucm_main_domain_tool.ckpt';assert sha(path)==PLAN['utility_model_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    config=dict(saved['hyper_parameters']['config']);config.update(load_path='',test_only=False)
    model=ViLTransformerSS(config);model.load_state_dict(saved['state_dict'],strict=True)
    model.eval().requires_grad_(False).cuda()
    assert model.use_agent_prior and model.prompt_length==16
    return model,clip


class QuantityScorer:
    def __init__(self,cases):
        self.pl,_=load_final_model()
        features=torch.load(ROOT/'features/tensors.pt',map_location='cpu',weights_only=True)
        self.qfeatures=features['quantity_text_features'];self.qgpu=self.qfeatures.cuda()
        self.vfeatures=features['original_val_text_features'];self.vgpu=self.vfeatures.cuda()
        self.gallery=json.loads((ROOT/'features/gallery.json').read_text())
        self.images={name:self.pl.preprocess(x['image']).unsqueeze(0).cuda() for name,x in case_images(cases).items()}
        self.last_feature=self.last_qsim=self.last_vsim=None

    @torch.no_grad()
    def score(self,hidden,item):
        assert hidden.shape==(3584,) and hidden.dtype==torch.bfloat16 and torch.isfinite(hidden).all()
        visual=self.pl.img_embeds(self.images[item['image']],agent_hidden=hidden.unsqueeze(0).cuda())['image_feats']
        qs=(visual@self.qgpu.T).cpu().flatten();result={}
        for wording,cols in self.gallery['quantity_columns'][item['image']].items():
            cp,cn=[float(qs[i]) for i in cols];margin=cp-cn
            result.update({wording+'_positive':cp,wording+'_negative':cn,'margin_'+wording:margin,'correct_'+wording:margin>0})
        self.last_vsim=None
        if item['split']=='val':
            vs=(visual@self.vgpu.T).cpu().flatten();pos=self.gallery['val_positive_columns'][item['image']]
            best=float(vs[pos].max());neg=[i for i in range(391) if i not in pos];rank=int((vs[:391]>best).sum())
            result.update(retrieval_rank0=rank,retrieval_rr=1/(1+rank),retrieval_margin=best-float(vs[neg].max()))
            self.last_vsim=vs
        else:result.update(retrieval_rank0=None,retrieval_rr=None,retrieval_margin=None)
        self.last_feature=visual.detach().cpu();self.last_qsim=qs
        return result

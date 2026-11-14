"""Explicitly separated training reward and validation evaluation inputs."""
import json
import sys

from milestone_common import ROOT,PLAN,ASSETS,SOURCE,private_clip,sha,table_images
import torch


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


class SplitReward:
    def __init__(self,split,rows):
        assert split in ['train','val'];self.split=split;self.pl,_=load_final_model()
        self.text_features=torch.load(ROOT/'features'/(split+'.pt'),map_location='cpu',weights_only=True)['text_features']
        self.text_gpu=self.text_features.cuda()
        if split=='train':
            data=json.loads((ROOT/'inputs/training_text_gallery.json').read_text())
            self.gallery=data;self.maps=data['maps'];self.n=len(data['text_order'])
        else:
            self.gallery=json.loads((ROOT/'features/val_gallery.json').read_text())
            self.n=self.gallery['unique_val_caption_count'];assert self.n==391
            pos=json.loads((ROOT/'features/val_positive_columns.json').read_text())
            self.maps={row:{'positive':value,'global_negative':[i for i in range(self.n) if i not in value]} for row,value in pos.items()}
        self.images={row:self.pl.preprocess(item['image']).unsqueeze(0).cuda() for row,item in table_images(rows,split).items()}
        self.last_feature=self.last_similarities=None

    @torch.no_grad()
    def score(self,hidden,row):
        assert hidden.shape==(3584,) and hidden.dtype==torch.bfloat16 and torch.isfinite(hidden).all()
        v=self.pl.img_embeds(self.images[row],agent_hidden=hidden.unsqueeze(0).cuda())['image_feats']
        sim=(v@self.text_gpu.T).cpu().flatten();m=self.maps[str(row)]
        positive=float(sim[m['positive']].max());gi=max(m['global_negative'],key=lambda i:float(sim[i]));negative=float(sim[gi])
        rank=int((sim[:self.n]>positive).sum())
        result={'rank0':rank,'rr':1/(1+rank),'r_global':positive-negative,'positive_best':positive,
                'global_negative_best':negative,'global_negative_index':gi}
        if self.split=='train':
            si=max(m['same_scene_negative'],key=lambda i:float(sim[i]))
            result.update(r_scene=positive-float(sim[si]),scene_negative_best=float(sim[si]),scene_negative_index=si,
                          global_negative_is_same_scene=gi in m['same_scene_negative'])
        self.last_feature=v.detach().cpu();self.last_similarities=sim
        return result

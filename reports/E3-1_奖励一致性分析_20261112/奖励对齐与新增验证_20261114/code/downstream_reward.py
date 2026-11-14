"""Frozen final AGSE reward with dataset-derived positive caption columns."""
import json
import sys

from milestone_common import ROOT,PLAN,ASSETS,SOURCE,private_clip,sha,table_images
import torch


class FinalRetrievalReward:
    def __init__(self,rows):
        private_clip()
        from run_frozen import MyBackboneFinetuning
        from vilt.modules.vilt_module import ViLTransformerSS
        # The archived Lightning callback was pickled under the script namespace.
        setattr(sys.modules['__main__'],'MyBackboneFinetuning',MyBackboneFinetuning)
        path=ASSETS/'ucm_main_domain_tool.ckpt'
        assert sha(path)==PLAN['utility_model_sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        config=dict(saved['hyper_parameters']['config']);config.update(load_path='',test_only=False)
        self.pl=ViLTransformerSS(config);self.pl.load_state_dict(saved['state_dict'],strict=True)
        self.pl.eval().requires_grad_(False).cuda()
        assert self.pl.use_agent_prior and self.pl.prompt_length==16
        self.gallery=json.loads((SOURCE/'scoring/text_gallery.json').read_text())
        original=torch.load(SOURCE/'scoring/scoring_tensors.pt',map_location='cpu',weights_only=True)
        self.text_features=original['text_features'];self.text_gpu=self.text_features.cuda()
        self.n=self.gallery['unique_val_caption_count'];assert self.n==391
        lookup={text:i for i,text in enumerate(self.gallery['text_order'][:self.n])}
        assert sha(ROOT/'inputs/dataset_metadata.json')==PLAN['new_validation_selection']['metadata_sha256']
        metadata=json.loads((ROOT/'inputs/dataset_metadata.json').read_text())
        self.positives={item['row']:sorted({lookup[c] for c in item['captions']}) for item in metadata['rows']}
        self.negatives={row:[i for i in range(self.n) if i not in pos] for row,pos in self.positives.items()}
        self.images={row:self.pl.preprocess(item['image']).unsqueeze(0).cuda() for row,item in table_images(rows).items()}
        self.last_feature=self.last_similarities=None

    @torch.no_grad()
    def score(self,hidden,row):
        assert list(hidden.shape)==[3584] and torch.isfinite(hidden).all()
        visual=self.pl.img_embeds(self.images[row],agent_hidden=hidden.unsqueeze(0).cuda())['image_feats']
        sims=(visual@self.text_gpu.T).cpu().flatten()
        assert torch.isfinite(visual).all() and torch.isfinite(sims).all()
        best=float(sims[self.positives[row]].max());negative=float(sims[self.negatives[row]].max())
        rank=int((sims[:self.n]>best).sum())
        self.last_feature=visual.detach().cpu();self.last_similarities=sims
        return {'actual_rank0':rank,'r_actual_rr':1/(1+rank),'r_aligned':best-negative,
                'best_positive_cosine':best,'best_negative_cosine':negative}

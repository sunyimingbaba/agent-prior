"""Cache authentic classifier responses and frozen R0 second-round contexts on this GPU."""
import torch
from shared import ROOT,ASSETS,REBUILD,dump,read,setup,table,rgb,sha
from policy import policy
from prompt_clip import clip
from rollout import rollout_one,SYSTEM_PROMPT
from tools import scene_classify_with_conf

class SecondRound(Exception):
    pass

@torch.inference_mode()
def main():
    setup()
    tool,pre=clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False)
    tool=tool.float().eval().cuda()
    cp=torch.load(ASSETS/'ucm_scene_cls.pt',map_location='cpu',weights_only=True)
    head=torch.nn.Linear(cp['feat_dim'],len(cp['classes']))
    head.load_state_dict(cp['head'])
    head=head.eval().cuda()
    answers={}
    for split,src in [('train','matched_policy_20261120/inputs/train.json'),('val','paired_views_20261120/inputs/validation.json')]:
        t=table(split)
        for row in read(REBUILD/src):
            key=row['rgb_sha256']
            if key not in answers:
                a,c=scene_classify_with_conf(rgb(t,row),model=tool,preprocess=pre,cls_head=head,cls_names=list(cp['classes']))
                answers[key]=[a,float(c)]
    dump(ROOT/'tools.json',answers)
    del tool,head
    torch.cuda.empty_cache()
    m,proc=policy('R0')
    gen=m.generate
    contexts={}
    t=table('train')
    for case in read(REBUILD/'matched_policy_20261120/inputs/train.json'):
        im=rgb(t,case)
        rounds=[]
        def capture(**kw):
            if rounds:
                contexts[case['image']]=dict(first=rounds[0],prefix=kw['input_ids'].cpu(),answer=answers[case['rgb_sha256']][0])
                raise SecondRound
            o=gen(**kw)
            rounds.append(o.sequences[0,kw['input_ids'].shape[1]:].tolist())
            return o
        m.generate=capture
        try:
            rollout_one(m,proc,im,case['scene_query'],{'scene_classify':lambda _:answers[case['rgb_sha256']][0]},3,64,'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
        except SecondRound:
            pass
        finally:
            m.generate=gen
    assert len(contexts)>0
    torch.save(contexts,ROOT/'contexts.pt')
    dump(ROOT/'prepare.json',dict(images=len(contexts),contexts=sha(ROOT/'contexts.pt'),tools=sha(ROOT/'tools.json'),
        torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda))
if __name__=='__main__':main()

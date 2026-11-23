"""Independently replay the eight first-group states with each sequence cut before its last action."""
import torch
from shared import ROOT, REBUILD, setup, read, dump, table, rgb
from policy import policy
from train import sample, SAMPLES

@torch.inference_mode()
def main():
    setup()
    model,proc=policy('R0')
    case=read(SAMPLES)[0]
    log=read(ROOT/'R1/steps.json')[0]
    im=rgb(table(),case)
    torch.manual_seed(log['seed'])
    single,traces=sample(model,proc,im,case['scene_query'])
    rows=[]
    for logged,(ids,h,ref,support,txt) in zip(log['trajectories'],traces,strict=True):
        assert logged['ids']==ids
        # For each sequence reconstruct the causal input predicting its final action, with no trailing pad.
        prefix=torch.cat([single.input_ids,torch.tensor([ids[:-1]],device='cuda',dtype=torch.long)],dim=1)
        kw=dict(single,input_ids=prefix,attention_mask=torch.ones_like(prefix))
        z=model(**kw,output_hidden_states=True,use_cache=False,return_dict=True,logits_to_keep=1)
        replay=z.hidden_states[-1][0,-1].cpu().float()
        cos=float(torch.nn.functional.cosine_similarity(replay,h.float(),dim=0))
        assert cos>0.99 and len(ids)>=1
        rows.append(dict(tokens=len(ids),ended=logged['ended'],cosine=cos))
    assert len(set(x['tokens'] for x in rows))>1
    dump(ROOT/'results/end_state_check.json',dict(image=case['image'],rows=rows))
if __name__=='__main__':
    main()

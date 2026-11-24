"""Intervene only in training trajectories, under the original measured tool feedback."""
import sys,torch
from env_pair import ROOT,OLD,HIST,read,savept,table,rgb,rows,setup,model_path,weight,inputs,captured_prefix
from policy import policy
@torch.inference_mode()
def main(seed):
    setup();m,proc=policy(model_path(seed));t=table();ps=rows();seen=read(ROOT/'inputs/candidates.json')[seed];out={};tok=proc.tokenizer
    known=torch.load(HIST/'records'/(seed+'_contexts.pt'),weights_only=True)
    for r in seen:
        nm=r['image'];c=ps[nm];im=rgb(t,c);inp=inputs(proc,im,c['scene_query']);prefix=captured_prefix(proc,inp,r)
        if nm in known:assert torch.equal(prefix,known[nm]['prefix'])
        inp=inputs(proc,im,c['scene_query'],prefix);v=[];seqs=[]
        for form in 'AB':
            for text in c[form]:
                seq=tok.encode('<answer>'+text+'</answer>',add_special_tokens=False)+[tok.eos_token_id];x=torch.tensor(seq[:-1],device='cuda')[None]
                ids=torch.cat((inp.input_ids,x),1);kw=dict(inp,input_ids=ids,attention_mask=torch.ones_like(ids))
                res=m(**kw,output_hidden_states=True,return_dict=True,use_cache=False,logits_to_keep=1)
                v.append(res.hidden_states[-1][0,-1].cpu());seqs.append(seq)
        assert torch.isfinite(torch.stack(v)).all()
        out[nm]=dict(prefix=prefix,seqs=seqs,hidden=torch.stack(v),rgb=c['rgb_sha256'])
    savept(ROOT/'records'/(seed+'_text_states.pt'),dict(images=out,adapter=weight(seed)))
if __name__=='__main__':main(sys.argv[1])

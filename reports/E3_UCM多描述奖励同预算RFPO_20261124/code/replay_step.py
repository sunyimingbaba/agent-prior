"""Recreate each seed's first sampled group and hand the archived hidden states to a separate scorer."""
import sys,json,torch
from runtime import ROOT,pool,read,initial,table,rgb,setup,savept
from prefixes import greedy
from policy import policy
from train import sample,finish
@torch.inference_mode()
def run(seed):
    setup();ctx=torch.load(ROOT/'records'/(seed+'_contexts.pt'),weights_only=True)['rows']
    nm=read(ROOT/'inputs/roster.json')['order'][0];c=pool()[nm];m,proc=policy(initial(seed));im=rgb(table(),c)
    src=json.loads((ROOT/'pilot'/(seed+'_R2')/'steps.jsonl').open().readline())
    h,gs,calls,_=greedy(m,proc,im,c['scene_query'],c['feedback']);assert gs==src['anchor']['actions'] and calls==src['anchor']['tools'] and torch.equal(h,ctx[nm]['prior'])
    out=[]
    for b,rng in enumerate(src['rng']):
        torch.manual_seed(rng);args,tr=sample(m,proc,im,c['scene_query'],ctx[nm]);states,third=finish(m,proc,args,tr,c['feedback'])
        for i,rec in enumerate(tr):
            row=src['records'][b*8+i];assert rec[0]==row['ids'] and third[i]==row['third'] and rec[2]==row['sampled_logp']
        out+=states
    savept(ROOT/'records'/(seed+'_first_batch.pt'),dict(image=nm,states=torch.stack(out+[h]).cpu(),adapter=read(ROOT/'inputs/roster.json')['initial'][seed]))
if __name__=='__main__':run(sys.argv[1])

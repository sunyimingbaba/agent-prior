"""Independently replay every action and terminal state of the first eight traces on a fixed pair."""
import torch
from remote_common import ROOT,setup,table,rgb,read,sha,dump,roster
from full_traces import trace_batch
from eval_io import weights
from policy import policy
from rollout import rollout_one,SYSTEM_PROMPT
@torch.inference_mode()
def main():
    setup();torch.set_num_threads(1);data=table();checks=[]
    for seed in 'ab':
        path,sig=weights(seed+'_deployed');m,proc=policy(path)
        for tag in ('first','second'):
            rows=roster(tag);first=rows[0]['pair']
            for row in [r for r in rows if r['pair']==first]:
                f=ROOT/'records'/tag/seed/'raw'/(row['image']+'.pt')
                cp=read(f.parent/'complete.json');assert sha(f)==cp['files'][row['image']]
                raw=torch.load(f,weights_only=True);assert raw['adapter']==sig
                im=rgb(data,row)
                for mode,z in raw['modes'].items():
                    _,paths,states=trace_batch(m,proc,im,z['query'],row['feedback'],z['rngs'][0])
                    assert [[rec[0] for _,rec in path] for path in paths]==z['paths'][:8]
                    assert torch.equal(torch.stack(states),z['hidden'][:8])
                    h,n,filtered=rollout_one(m,proc,im,z['query'],{'scene_classify':lambda _:row['feedback']},3,64,
                        'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
                    assert torch.equal(h.cpu(),z['hidden'][-1]) and n==len(z['calls']) and filtered==0
                    checks.append(dict(image=row['image'],mode=mode,panel=tag,seed=seed,source=sha(f),draws=8,greedy=1,
                        sampled_rounds=sum(map(len,paths))))
        del m,proc;torch.cuda.empty_cache()
    dump(ROOT/'results/sampling_replay.json',dict(checks=checks,sampled=sum(r['draws'] for r in checks),greedy=len(checks)))
if __name__=='__main__':main()

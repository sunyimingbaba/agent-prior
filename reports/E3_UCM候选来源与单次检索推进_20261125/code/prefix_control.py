"""Matched first-action ablation. Both arms use the same stochastic continuation and seeds."""
import sys,time
import torch
from collections import defaultdict
from remote_common import ROOT,BASE,setup,table,rgb,savept,dump,read,sha
from eval_io import weights
from policy import policy,prompt,cut
from rollout import parse_tool_call
from prefixes import greedy
from sample_sources import module as original

def cases(tag):
    p=ROOT/'inputs'/(tag+'.json');lock=read(ROOT/'inputs/prefix_lock.json')
    assert sha(p)==lock[tag] and sha(ROOT/'首轮探索复核口径.md')==lock['protocol']
    return read(p)

def tail(model,proc,inputs,prefix,feedback,rng):
    pending=[(list(range(8)),prefix)];paths=[[] for _ in range(8)];final={}
    for turn in (1,2):
        next_round=[]
        for slots,ids in pending:
            torch.manual_seed(rng+turn*11+min(slots))
            single=dict(inputs,input_ids=ids,attention_mask=torch.ones_like(ids))
            batch={k:v.repeat(len(slots),*([1]*(v.ndim-1))) for k,v in single.items()}
            ans=model.generate(**batch,max_new_tokens=64,do_sample=True,temperature=1.,output_hidden_states=True,return_dict_in_generate=True)
            continued=defaultdict(list)
            for j,i in enumerate(slots):
                tokens=cut(ans.sequences[j,ids.shape[1]:].tolist(),model.generation_config.eos_token_id)
                paths[i].append(tokens);text=proc.decode(tokens,skip_special_tokens=True)
                if turn==1 and parse_tool_call(text):continued[text].append(i)
                else:final[i]=ans.hidden_states[len(tokens)-1][-1][j,-1].detach().cpu().clone()
            del ans
            for action,indices in continued.items():
                extra=proc(text=[action+f'\n<result>{feedback}</result>\n'],return_tensors='pt').input_ids.to(ids.device)
                next_round.append((indices,torch.cat((ids,extra),dim=1)))
        pending=next_round
    assert len(final)==8
    return [final[i] for i in range(8)],paths

@torch.inference_mode()
def main(seed):
    setup();torch.set_num_threads(1);path,sig=weights(seed+'_deployed');m,proc=policy(path);data=table()
    for tag in ('second','third'):
        rows=cases(tag);dest=ROOT/'records/prefix'/tag/seed/'raw';dest.mkdir(exist_ok=True,parents=True)
        for j,c in enumerate(rows):
            nm=c['image'];p=dest/(nm+'.pt')
            if p.is_file():
                oldest=torch.load(p,weights_only=True);assert oldest['adapter']==sig and oldest['rgb']==c['rgb_sha256'];continue
            im=rgb(data,c);q=c['scene_query'];answer=c['feedback']
            h,greedy_ids,calls,prefix=greedy(m,proc,im,q,answer)
            inp=proc(text=[prompt(proc,q)],images=[im],return_tensors='pt').to('cuda')
            h_locked=[];h_free=[];p_locked=[];p_free=[];rngs=[];checks=0
            cached=None
            if tag=='second':
                previous=ROOT/'records/second'/seed/'raw'/(nm+'.pt')
                raw0=torch.load(previous,weights_only=True);cached=raw0['modes']['scene']
                assert raw0['adapter']==sig and cached['query']==q and cached['greedy']==greedy_ids and torch.equal(cached['hidden'][-1],h)
            for batch in range(2):
                rng=(10000000 if seed=='a' else 20000000)+1000*c['row']+100*batch;rngs.append(rng)
                if prefix is None:
                    locked=[h]*8;ids=[[greedy_ids[0]] for _ in range(8)]
                else:
                    locked,paths=tail(m,proc,inp,prefix.to('cuda'),answer,rng)
                    ids=[[greedy_ids[0]]+v for v in paths]
                if cached is None:
                    free,actions=original.collect(m,proc,inp,list(range(8)),rng,answer);h_free+=free;p_free+=actions
                else:
                    free=list(cached['hidden'][8*batch:8*batch+8]);actions=cached['paths'][8*batch:8*batch+8]
                if prefix is not None and all(v[0]==greedy_ids[0] for v in actions):
                    assert actions==ids and torch.equal(torch.stack(free),torch.stack(locked));checks+=1
                h_locked+=locked;p_locked+=ids
            modes=dict(locked=dict(hidden=torch.stack(h_locked+[h]),paths=p_locked))
            if tag=='third':modes['free']=dict(hidden=torch.stack(h_free+[h]),paths=p_free)
            savept(p,dict(image=nm,rgb=c['rgb_sha256'],adapter=sig,query=q,feedback=answer,greedy=greedy_ids,
                calls=calls,rngs=rngs,exact_shared_context_batches=checks,modes=modes))
            if (j+1)%12==0:print(tag,seed,j+1,len(rows),flush=True)
        dump(dest/'complete.json',dict(adapter=sig,files={c['image']:sha(dest/(c['image']+'.pt')) for c in rows},protocol=sha(ROOT/'首轮探索复核口径.md')))
if __name__=='__main__':main(sys.argv[1])

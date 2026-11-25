"""Sample actions from every live round and retain the exact support used for log-probability replay."""
from collections import defaultdict
import torch
from remote_common import BASE
from policy import prompt,cut
from rollout import parse_tool_call

@torch.no_grad()
def trace_batch(model,proc,image,query,feedback,rng):
    single=proc(text=[prompt(proc,query)],images=[image],return_tensors='pt').to('cuda')
    pending=[(list(range(8)),single.input_ids)]
    history=[[] for _ in range(8)];terminal={};inputs={};cfg=model.generation_config
    assert cfg.top_k==50 and cfg.top_p==1 and cfg.repetition_penalty==1.05
    for turn in range(3):
        later=[]
        for slots,prefix in pending:
            torch.manual_seed(rng+turn*11+min(slots))
            context=type(single)(dict(single,input_ids=prefix,attention_mask=torch.ones_like(prefix)))
            ctxid=len(inputs);inputs[ctxid]=context
            batch={k:v.repeat(len(slots),*([1]*(v.ndim-1))) for k,v in context.items()}
            ans=model.generate(**batch,max_new_tokens=64,do_sample=True,temperature=1.,
                output_hidden_states=True,output_scores=True,return_dict_in_generate=True)
            continued=defaultdict(list)
            for j,slot in enumerate(slots):
                output=ans.sequences[j,prefix.shape[1]:].tolist();ids=cut(output,cfg.eos_token_id)
                assert ids and all(tok==cfg.pad_token_id for tok in output[len(ids):])
                support=[];logps=[]
                for i,tok in enumerate(ids):
                    logits=ans.scores[i][j].float()
                    valid=torch.where(torch.isfinite(logits))[0]
                    assert len(valid)>=cfg.top_k and (valid==tok).any()
                    support.append(valid);logps.append(float(torch.log_softmax(logits,-1)[tok]))
                h=ans.hidden_states[len(ids)-1][-1][j,-1].detach().cpu().clone()
                text=proc.decode(ids,skip_special_tokens=True)
                history[slot].append((ctxid,(ids,None,logps,support,text)))
                if turn<2 and parse_tool_call(text):
                    continued[text].append(slot)
                else:terminal[slot]=h
            del ans
            for action,slots1 in continued.items():
                tokens=proc(text=[action+f'\n<result>{feedback}</result>\n'],return_tensors='pt').input_ids.to(prefix.device)
                later.append((slots1,torch.cat((prefix,tokens),dim=1)))
        pending=later
    assert not pending and len(terminal)==8
    return inputs,history,[terminal[i] for i in range(8)]

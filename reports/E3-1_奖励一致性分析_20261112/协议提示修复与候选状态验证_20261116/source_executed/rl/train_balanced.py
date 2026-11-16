import json
import math
from pathlib import Path
import random
import time
import numpy as np
from milestone_common import ROOT,PLAN,setup,seed,case_images,load_policy,real_tools,lora_hash
from checkpoint_io import restore_adapter_exact
from quantity_model import QuantityScorer
from training_contract import recorded_rollout,grad_logprob
from live_rollout_eos import tensor_sha
from stage_io import read,dump,sha

torch=setup();seed(0);start=time.monotonic();assert read(ROOT/'prepare_result.json')['status']=='complete'
from agent.rollout import SYSTEM_PROMPT
out=ROOT/'training';out.mkdir();cases=read(ROOT/'inputs/cases.json');by={x['image']:x for x in cases};selected=[by[n] for n in PLAN['training_images']]
images=case_images(selected);tools=real_tools();model,processor=load_policy(Path(PLAN['policy']['checkpoint']),True)
restore_adapter_exact(model,Path(PLAN['policy']['checkpoint']));initial=lora_hash(model);assert initial==PLAN['policy']['lora_tensor_hash']
scorer=QuantityScorer(cases);parameters=[p for p in model.parameters() if p.requires_grad]
assert len(parameters)==584 and all(p.dtype==torch.float32 for p in parameters)
cfg=PLAN['training'];optimizer=torch.optim.AdamW(parameters,lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])
dump(out/'settings.json',{'config':cfg,'initial_parameter_hash':initial,'trainable_tensors':len(parameters),'trainable_parameters':sum(p.numel() for p in parameters),'updates':32,'training_images':PLAN['training_images']})
dump(out/'order.json',PLAN['training_order']);updates=[];records=[];reward_features=[];reward_q=[];previous=initial
def checkpoint(label,step):
    folder=out/label/'checkpoint';folder.mkdir(parents=True)
    model.save_pretrained(folder,safe_serialization=True);processor.save_pretrained(folder);torch.save(optimizer.state_dict(),folder/'optimizer.pt')
    torch.save({'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()},folder/'torch_rng.pt');ns=np.random.get_state()
    dump(folder/'python_numpy_rng.json',{'python_random_state':random.getstate(),'numpy_random_state':{'kind':ns[0],'keys':ns[1].tolist(),'position':ns[2],'has_gauss':ns[3],'cached_gaussian':ns[4]}})
    dump(folder/'training_state.json',{'optimizer_updates':step,'parameter_hash':lora_hash(model),'initial_parameter_hash':initial,'plan_sha256':sha(ROOT/'plan.json'),'training_order_sha256':sha(out/'order.json')})
    dump(out/(label+'_checkpoint_manifest.json'),{'ok':True,'optimizer_updates':step,'parameter_hash':lora_hash(model),'files':{str(p.relative_to(folder)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in folder.rglob('*') if p.is_file()}})
try:
    for item in PLAN['training_order']:
        step=item['step'];name=item['image'];assert by[name]['actual_count']==item['actual_count'];group=[]
        for g in range(4):
            ss=1700000+(step-1)*4+g;seed(ss)
            h,tr,_=recorded_rollout(model,processor,images[name]['image'],PLAN['prompt'],tools,SYSTEM_PROMPT,PLAN['rollout'])
            stem=f'{step:02d}_{name}_{g}';sf=out/'priors'/(stem+'.pt');sf.parent.mkdir(exist_ok=True);torch.save(h,sf)
            assert torch.equal(h,torch.load(sf,map_location='cpu',weights_only=True)) and tensor_sha(h)==tr['state_tensor_sha256']
            tf=out/'traces'/(stem+'.json');dump(tf,tr);value=scorer.score(h,by[name]);fmt=.2 if tr['quality_usable'] else 0.
            row={'id':stem,'step':step,'sample':g,'image':name,'split':'train','row':by[name]['row'],'actual_count':item['actual_count'],'seed':ss,
              'quality_usable':tr['quality_usable'],'strict_protocol_usable':tr['strict_protocol_usable'],'tools_used':tr['tools_used'],'stop_reason':tr['stop_reason'],
              **value,'r_format':fmt,'r_total':fmt+value['margin_A'],'state_file':str(sf.relative_to(ROOT)),'state_file_sha256':sha(sf),'generation_state_sha256':tensor_sha(h),
              'trace_file':str(tf.relative_to(ROOT)),'trace_sha256':sha(tf),'policy_parameter_hash_before':previous}
            records.append(row);group.append((row,tr));reward_features.append(scorer.last_feature);reward_q.append(scorer.last_qsim)
            with (out/'samples.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        rewards=torch.tensor([x['r_total'] for x,_ in group],device='cuda',dtype=torch.float32);std=rewards.std(unbiased=True)
        assert float(std)>1e-8,'Degenerate reward group retained; optimizer not advanced'
        advantages=((rewards-rewards.mean())/(std+1e-8)).detach();optimizer.zero_grad(set_to_none=True);model.train();loss_sum=0.;loss_tokens=0;prob=[]
        for g,(row,tr) in enumerate(group):
            for ri,z in enumerate(tr['rounds']):
                lp,details,_=grad_logprob(model,processor,images[name]['image'],z)
                delta=torch.tensor(details['token_logprobs'])-torch.tensor(z['actual_generation_token_logprobs'])
                assert float(delta.abs().max())<=PLAN['probability_gate']['teacher_vs_generation_max_token_logprob_gap'] and float(delta.abs().mean())<=PLAN['probability_gate']['teacher_vs_generation_mean_absolute_token_logprob_gap']
                loss=-advantages[g]*lp/4;assert torch.isfinite(loss);loss.backward();loss_sum+=float(loss.detach());loss_tokens+=details['effective_tokens']
                prob.append({'sample':g,'round':ri,'logprob':float(lp.detach()),'advantage':float(advantages[g]),'token_logprobs':details['token_logprobs'],'effective_tokens':details['effective_tokens'],'max_logprob_gap':float(delta.abs().max()),'mean_abs_logprob_gap':float(delta.abs().mean())})
                del lp,loss
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip_norm']));assert math.isfinite(norm) and norm>0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters) and all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
        optimizer.step();current=lora_hash(model);assert current!=previous and all(torch.isfinite(p).all() for p in parameters)
        assert all(not p.requires_grad and p.grad is None for p in scorer.pl.parameters())
        update={'step':step,'image':name,'actual_count':item['actual_count'],'optimizer_update':True,'reward_mean':float(rewards.mean()),'reward_std':float(std),
          'advantages':advantages.cpu().tolist(),'loss':loss_sum,'effective_loss_tokens':loss_tokens,'gradient_norm_before_clip':norm,'parameter_hash_before':previous,'parameter_hash_after':current,
          'quality_count':sum(x['quality_usable'] for x,_ in group),'strict_quality_count':sum(x['strict_protocol_usable'] for x,_ in group),'actual_tools':sum(x['tools_used'] for x,_ in group),'probability_replays':prob}
        updates.append(update);previous=current
        with (out/'updates.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(update)+'\n')
        dump(ROOT/'progress.json',{'phase':'balanced_training','optimizer_updates':step,'total_updates':32,'image':name,'actual_count':item['actual_count'],'quality_count':update['quality_count']})
        print('BALANCED_UPDATE',step,'/32','image',name,'count',item['actual_count'],'quality',update['quality_count'],'/4',flush=True)
        if step==16:checkpoint('intermediate_16',step)
    assert len(updates)==32 and len(records)==128;checkpoint('final',32)
except Exception:
    checkpoint('interrupted',len(updates));raise
torch.save({'record_ids':[x['id'] for x in records],'image_features':torch.cat(reward_features),'quantity_similarities':torch.stack(reward_q)},out/'reward_tensors.pt')
dump(out/'result.json',{'ok':True,'optimizer_updates':32,'samples':128,'training_images':8,'initial_parameter_hash':initial,'final_parameter_hash':previous,
 'quality_samples':sum(x['quality_usable'] for x in records),'actual_tool_events':sum(x['tools_used'] for x in records),'checkpoint':'training/final/checkpoint','checkpoint_selection':'fixed update32',
 'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
print('BALANCED_32_UPDATES_COMPLETE',previous,flush=True)

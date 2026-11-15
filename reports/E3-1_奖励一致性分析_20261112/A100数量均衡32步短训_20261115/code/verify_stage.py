import argparse
import collections
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import json
import numpy as np
import torch
from stage_io import read,dump,sha,parents
from verify_metrics import verify
torch.set_num_threads(2)
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root;plan=read(r/'plan.json');analysis=read(r/'analysis.json')
for snap in ['core_code_hashes.json','training_code_hashes.json','post_code_hashes.json']:
    for n,h in read(r/snap).items():assert sha(r/n)==h,n
for n,h in plan['inputs_sha256'].items():assert sha(r/n)==h,n
d,e=parents(r);parentcheck=r/'scratch/parent_diagnostic_recheck.json'
p=subprocess.run([sys.executable,str(d/'code/verify_diagnostic.py'),'--root',str(d),'--out',str(parentcheck)],capture_output=True,text=True);assert p.returncode==0,p.stdout+p.stderr
assert read(parentcheck)['ok']
cases=read(r/'inputs/evaluation_cases.json');by={x['image']:x for x in cases};assert len(by)==85
trainset=set(plan['training_images']);holdout=set(plan['confirmation_train_holdout_images']);val=set(plan['supplementary_existing_val_images'])
assert len(trainset)==8 and len(holdout)==55 and len(val)==22 and not trainset&holdout and not trainset&val and not holdout&val
assert all(by[n]['split']=='train' for n in trainset|holdout) and all(by[n]['split']=='val' for n in val)
scores=read(r/'scoring/all_scores.json');t=torch.load(r/'scoring/tensors.pt',map_location='cpu',weights_only=True);f=torch.load(r/'features/tensors.pt',map_location='cpu',weights_only=True);gallery=read(r/'features/gallery.json')
assert len(scores)==1148 and t['record_ids']==[x['id'] for x in scores];q64=t['image_features'].double()@f['quantity_text_features'].double().T
assert float((q64-t['quantity_similarities'].double()).abs().max())<1e-6
ix={x:i for i,x in enumerate(t['record_ids'])};v64=t['image_features'][[ix[x] for x in t['val_record_ids']]].double()@f['original_val_text_features'].double().T
assert float((v64-t['val_similarities'].double()).abs().max())<1e-6;vi={x:i for i,x in enumerate(t['val_record_ids'])}
actual=read(r/'scoring/actual_tools.json');classes=ranks=events=rounds=0;ties=[]
for i,x in enumerate(scores):
    h=torch.load(r/x['state_file'],map_location='cpu',weights_only=True);assert h.dtype==torch.bfloat16 and h.shape==(3584,) and torch.isfinite(h).all()
    hs=hashlib.sha256(h.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest();assert sha(r/x['state_file'])==x['state_file_sha256'] and hs==x['generation_state_sha256']==x['scorer_state_sha256']
    assert sha(r/x['trace_file'])==x['trace_sha256'];tr=read(r/x['trace_file']);assert hs==tr['state_tensor_sha256'] and not tr['forced_tool_witness']
    if x['role']!='shared':
        assert tr['quality_usable']==x['quality_usable'] and tr['strict_protocol_usable']==x['strict_protocol_usable']
        assert len(tr['tool_events'])==x['tools_used']
        for z in tr['rounds']:
            ids=z['generated_ids'];assert len(ids)==z['generated_token_count'] and ids[0] not in z['eos_token_ids'] and all(k not in z['eos_token_ids'] for k in ids[:-1])
            assert (z['prompt_token_ids']+ids)[-2]==z['state_input_token_id'] and ids[-1]==z['last_emitted_token_id']
            assert z['generation_kwargs']['min_new_tokens']==1 and z['generation_kwargs']['top_k']==0 and z['generation_kwargs']['top_p']==1.;rounds+=1
        if x['quality_usable']:assert tr['rounds'][-1]['last_emitted_token_id'] in tr['rounds'][-1]['eos_token_ids']
        for event in tr['tool_events']:
            assert event['origin']=='model_requested_actual_tool' and event['text']==actual[x['image']]['text'] and abs(event['confidence']-actual[x['image']]['confidence'])<1e-6;events+=1
    else:assert x['reference_image']=='102.tif'
    for wording,cols in gallery['quantity_columns'][x['image']].items():
        cp,cn=[float(t['quantity_similarities'][i,c]) for c in cols];assert cp==x[wording+'_positive'] and cn==x[wording+'_negative'] and cp-cn==x['margin_'+wording] and (cp>cn)==x['correct_'+wording]
        same=bool(q64[i,cols[0]]>q64[i,cols[1]])==x['correct_'+wording];classes+=same
        if not same:
            margin=float(q64[i,cols[0]]-q64[i,cols[1]]);assert abs(margin)<1e-6;ties.append({'id':x['id'],'wording':wording,'float64_margin':margin})
    if x['split']=='val':
        row=v64[vi[x['id']]];pos=gallery['val_positive_columns'][x['image']];rank=int((row[:391]>row[pos].max()).sum());assert rank==x['retrieval_rank0'] and x['retrieval_rr']==1/(rank+1);ranks+=1
training=read(r/'training/result.json');updates=[json.loads(s) for s in (r/'training/updates.jsonl').read_text().splitlines()];samples=[x for x in scores if x['arm']=='training']
assert len(updates)==32 and len(samples)==128 and training['optimizer_updates']==32
last=plan['policy']['lora_tensor_hash'];prob_rows=0
for u,scheduled in zip(updates,plan['training_order']):
    step=u['step'];assert (u['image'],u['actual_count'],step)==(scheduled['image'],scheduled['actual_count'],scheduled['step'])
    xs=sorted([x for x in samples if x['step']==step],key=lambda x:x['sample']);assert len(xs)==4 and all(x['image']==u['image'] for x in xs)
    for g,x in enumerate(xs):
        assert x['seed']==1700000+(step-1)*4+g and x['policy_parameter_hash_before']==last and x['actual_count']==by[x['image']]['actual_count']
        assert x['r_format']==(.2 if x['quality_usable'] else 0.) and x['r_total']==x['r_format']+x['margin_A']
    rw=np.array([x['r_total'] for x in xs],dtype=np.float32);assert abs(float(rw.astype(float).mean())-u['reward_mean'])<1e-7 and abs(float(rw.astype(float).std(ddof=1))-u['reward_std'])<1e-7 and u['reward_std']>1e-8
    adv=(torch.tensor(rw)-np.float32(u['reward_mean']))/(np.float32(u['reward_std'])+1e-8);assert np.allclose(adv.numpy(),u['advantages'],atol=1e-6,rtol=0)
    losses=[];ntokens=0
    for z in u['probability_replays']:
        trace=read(r/xs[z['sample']]['trace_file'])['rounds'][z['round']];lp=z['token_logprobs'];assert len(lp)==trace['generated_token_count']==z['effective_tokens']
        assert all(math.isfinite(v) and v<=0 for v in lp) and abs(math.fsum(lp)-z['logprob'])<2e-4
        delta=np.array(lp)-np.array(trace['actual_generation_token_logprobs']);assert float(abs(delta).max())<=.25 and float(abs(delta).mean())<=.05
        assert abs(z['advantage']-u['advantages'][z['sample']])<1e-7
        losses.append(float(-torch.tensor(z['advantage'],dtype=torch.float32)*torch.tensor(z['logprob'],dtype=torch.float32)/4));ntokens+=len(lp);prob_rows+=len(lp)
    assert abs(math.fsum(losses)-u['loss'])<1e-5 and ntokens==u['effective_loss_tokens'] and u['gradient_norm_before_clip']>0
    assert u['parameter_hash_before']==last and u['parameter_hash_after']!=last;last=u['parameter_hash_after']
assert last==training['final_parameter_hash'] and read(r/'training/final_checkpoint_manifest.json')['parameter_hash']==last
w=torch.load(r/'contract_gate/logit_witness.pt',map_location='cpu',weights_only=True);wrows=0
for z in w.values():
    for key,expected in [('generation_logits','generation_chosen_logprobs'),('teacher_logits','teacher_chosen_logprobs')]:
        logits=z[key].double();target=z['targets'];lp=torch.log_softmax(logits,dim=-1).gather(1,target[:,None]).flatten().numpy();assert np.allclose(lp,z[expected],atol=2e-5,rtol=0);wrows+=len(lp)
gate=read(r/'contract_gate/result.json');assert gate['ok'] and gate['permanent_optimizer_updates']==0 and gate['restored_initial_hash']==plan['policy']['lora_tensor_hash'] and gate['temporary_optimizer_steps']==1
ci,per=verify(plan,analysis,scores,t,gallery);g=analysis['groups']['confirmation55'];m=g['raw']['after']
checks={'optimizer_updates_exactly_32':True,'updates_per_count_exactly_8':all(sum(u['actual_count']==c for u in updates)==8 for c in [1,2,3,4]),'updates_per_image_exactly_4':all(sum(u['image']==n for u in updates)==4 for n in trainset),
 'holdout_sample_strict_quality_at_least_90pct':m['sample_mean']['strict_quality']['mean']>=.9,'holdout_greedy_strict_quality_at_least_95pct':m['greedy']['strict_quality']['mean']>=.95,
 'holdout_sample_actual_tools_at_least_80pct':m['sample_mean']['tool_usage']['mean']>=.8,'holdout_generated_result_failure_at_most_2pct':m['sample_mean']['generated_result_failure']['mean']<=.02,
 'holdout_count_macro_B_gain_interval_above_zero':g['macro']['after_minus_before__sample_mean']['pairacc_B']['interval95'][0]>0,'holdout_own_minus_shared_macro_B_interval_above_zero':g['macro']['after_own_minus_shared__greedy']['pairacc_B']['interval95'][0]>0}
assert checks==analysis['adoption']['checks'] and all(checks.values())==analysis['adoption']['passed']
dump(a.out,{'ok':True,'optimizer_updates_verified':32,'training_records_verified':128,'evaluation_own_records_verified':850,'shared_control_records_verified':170,'all_state_and_trace_records_verified':1148,
 'generation_rounds_checked':rounds,'actual_tool_events_checked':events,'training_probability_tokens_checked':prob_rows,'float64_full_vocabulary_witness_rows':wrows,
 'float64_quantity_classifications_equal':int(classes),'quantity_classification_checks':2296,'numerical_tie_records':ties,'float64_val_ranks_equal':ranks,'val_rank_checks':264,
 'per_image_rows_rebuilt':per,'bootstrap_intervals_rebuilt':ci,'adoption_gates_rebuilt':True,'parent_eos_sft_and_split_diagnostic_recursively_verified':True,'final_policy_hash':last,'torch':torch.__version__,'numpy':np.__version__})
print('ALL_BALANCED_TRAINING_AND_EVALUATION_EVIDENCE_VERIFIED')

"""Independent reconstruction of samplegroups, advantage/loss/hash chain and checkpointoptimizer state."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
from safetensors.torch import load_file
r=Path(sys.argv[1]).resolve();torch.set_num_threads(4)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def near(a,b,tol=1e-6):assert abs(float(a)-float(b))<=tol,(a,b,tol)
p=read(r/'plan.json');initial=p['policy']['lora_hash'];info={};checked=0
for arm in [sys.argv[2]]:
 assert arm in p['reward_arms']
 root=r/'training_v2'/arm;result=read(root/'result.json');settings=read(root/'settings.json');assert settings['initial_policy_hash']==initial and settings['fresh_optimizer'] and settings['config']==p['training'];rows=[json.loads(v) for v in (root/'samples.jsonl').read_text().splitlines()];updates=[json.loads(v) for v in (root/'updates.jsonl').read_text().splitlines()];assert len(rows)==256 and len(updates)==32;previous=initial;actual=0
 for ai,u in enumerate(updates):
  group=rows[ai*8:(ai+1)*8];name=p['training']['image_order'][ai];assert u['attempt']==ai+1 and u['image']==name and u['policy_hash_before']==previous
  for si,row in enumerate(group):
   assert row['arm']==arm and row['attempt']==ai+1 and row['image']==name and row['sample']==si and row['seed']==2026111800+100*ai+si and row['policy_hash_before']==previous;near(row['reward'],row['r_'+arm],1e-12)
  rewards=torch.tensor([x['reward'] for x in group],dtype=torch.float32);std=rewards.std(unbiased=True);informative=float(std)>1e-8;near(u['reward_std'],std,1e-6);near(u['reward_mean'],rewards.mean(),1e-6);assert u['optimizer_update']==informative
  adv=(rewards-rewards.mean())/(std+1e-8) if informative else torch.zeros(8);assert torch.max(torch.abs(adv-torch.tensor(u['advantages'])))<1e-5
  if informative:
   actual+=1;assert u['policy_hash_after']!=previous and u['gradient_norm_before_clip']>0 and np.isfinite(u['gradient_norm_before_clip']);loss=0.;tokens=0
   for si,row in enumerate(group):
    tr=read(r/row['trace_file'])
    for ri,z in enumerate(tr['rounds']):
     d=next(v for v in u['probability_replays'] if v['sample']==si and v['round']==ri);lp=np.array(d['token_logprobs']);gen=np.array(z['actual_generation_token_logprobs']);delta=lp-gen;assert len(lp)==len(gen)==d['effective_tokens'];near(d['max_token_logprob_gap'],np.max(abs(delta)));near(d['mean_absolute_token_logprob_gap'],np.mean(abs(delta)));assert np.max(abs(delta))<=.25 and np.mean(abs(delta))<=.05;near(d['advantage'],adv[si]);loss+=-float(adv[si])*float(torch.tensor(lp,dtype=torch.float32).sum())/8;tokens+=len(lp);checked+=1
   near(u['loss'],loss,2e-3);assert tokens==u['effective_loss_tokens']
  else:assert u['policy_hash_after']==previous and not u['probability_replays'] and u['gradient_norm_before_clip'] is None
  previous=u['policy_hash_after'];assert u['optimizer_updates_so_far']==actual
 assert result['initial_policy_hash']==initial and result['final_policy_hash']==previous and result['optimizer_updates']==actual and result['zero_variance_skips']==32-actual;cpinfo={}
 for label,end in [('intermediate_16',16),('final',32)]:
  mf=read(root/(label+'_checkpoint_manifest.json'));folder=root/label/'checkpoint';expected=sum(bool(v['optimizer_update']) for v in updates[:end]);assert mf['optimizer_updates']==expected and mf['attempts_completed']==end
  for n,v in mf['files'].items():assert (folder/n).stat().st_size==v['bytes'] and sha(folder/n)==v['sha256']
  tensors=load_file(str(folder/'adapter_model.safetensors'));live={n.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight'):v for n,v in tensors.items()};assert len(live)==584 and all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in live.values());h=hashlib.sha256()
  for n,v in sorted(live.items()):h.update(n.encode());h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
  assert h.hexdigest()==mf['policy_hash']==updates[end-1]['policy_hash_after'];opt=torch.load(folder/'optimizer.pt',weights_only=True,map_location='cpu');assert len(opt['param_groups'])==1;g=opt['param_groups'][0]
  for key in ['lr','eps','weight_decay']:near(g[key],p['training'][key],1e-12)
  assert list(g['betas'])==p['training']['betas'];steps=[int(v['step']) for v in opt['state'].values()];assert (len(steps)==584 and set(steps)=={expected}) if expected else len(steps)==0;assert all(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in opt['state'].values());cpinfo[label]={'manifest_files':len(mf['files']),'adapter_tensors':len(live),'optimizer_steps':expected,'policy_hash':h.hexdigest()};del tensors,live,opt
 info[arm]={'samples':256,'attempts':32,'optimizer_updates':actual,'zero_variance_skips':32-actual,'checkpoints':cpinfo}
result={'ok':True,'arms':info,'probability_rounds_rebuilt':checked,'scope':'one completed arm, other arms still in progress','common_start_bound':True,'all_group_rewards_advantages_losses_rebuilt':True,'checkpoint_optimizer_steps_match_actual_updates':True};path=r/'independent_completed_arm_verification.json';assert not path.exists();path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))

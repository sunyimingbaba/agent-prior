"""Fresh greedy train+val anchors authorize optional parent baseline reuse."""
from pathlib import Path
from milestone_common import ROOT,setup,seed,load_policy,lora_hash
from checkpoint_io import restore_adapter_exact
from stochastic_rollout import rollout
from prompt_design import messages
from reward_model_v2 import RewardScorer
from stage_io import read,dump,sha
torch=setup();p=read(ROOT/'plan.json');prompt=read(ROOT/'prompt_selection_lock.json')['selected_prompt'];out=ROOT/'baseline_replay';out.mkdir();assert read(ROOT/'probability_gate/result.json')['ok']
if prompt!='contract_only':
 dump(out/'result.json',{'ok':True,'baseline_reuse':False,'reason':'selected prompt differs from authorized reusable parent contract'});raise SystemExit(0)
parent=Path(p['parent_protocol_root']);records=read(parent/'generation_v2/records.json');cases=read(ROOT/'inputs/quantity_cases.json');by={c['image']:c for c in read(ROOT/'inputs/cohort.json')};names=[p['training_images'][0],next(c['image'] for c in cases if c['split']=='val')]
seed(0);model,processor=load_policy(Path(p['policy']['checkpoint']),False);restore_adapter_exact(model,Path(p['policy']['checkpoint']));assert lora_hash(model)==p['policy']['lora_hash'];scorer=RewardScorer();items=[];baseline=torch.load(parent/'evaluation/calibrated_own/selected_last/correct/tensors.pt',map_location='cpu',weights_only=True)
from agent.rollout import SYSTEM_PROMPT
for name in names:
 c=by[name];group='pilot_initial/contract_only' if c['split']=='train' else 'selected_validation';ref=next(x for x in records if x['image']==name and x['group']==group);old=read(parent/ref['trace_file']);oldstate=torch.load(parent/ref['tensor_file'],map_location='cpu',weights_only=True)
 h,tensors,tr=rollout(model,processor,scorer.case_image(c),messages(prompt,c,SYSTEM_PROMPT),{**c['feedback'],'source_image':name,'source_original_trace_sha256':c['feedback_original_trace_sha256']},p['rollout'],temperature=0.)
 token_exact=len(tr['rounds'])==len(old['rounds']) and all(a['generated_ids']==b['generated_ids'] and a['prompt_token_ids']==b['prompt_token_ids'] for a,b in zip(tr['rounds'],old['rounds']));states_exact=torch.equal(h,oldstate['candidates']['last']) and len(tensors['step_states'])==len(oldstate['step_states']) and all(torch.equal(a,b) for a,b in zip(tensors['step_states'],oldstate['step_states']));protocol_exact=all(tr[k]==old[k] for k in ['stop_reason','tools_used','generated_result_tag','strict_protocol_usable']);value=scorer.score(h,c,tr['strict_completed_consumed_tool']);i=next(i for i,x in enumerate(cases) if x['image']==name);feature_exact=torch.equal(scorer.last_tensors['image_feature'][0],baseline['quantity_image_features'][i]);err=float((scorer.last_tensors['quantity_scores']-baseline['quantity_scores'][i]).abs().max());ok=token_exact and states_exact and protocol_exact and feature_exact and err<1e-6
 sf=out/(c['key']+'.pt');tf=out/(c['key']+'.json');torch.save({**tensors,**scorer.last_tensors},sf);dump(tf,tr);items.append({'image':name,'split':c['split'],'all_round_tokens_exact':token_exact,'all_step_states_and_last_exact':states_exact,'protocol_exact':protocol_exact,'image_feature_exact':feature_exact,'quantity_score_max_abs_error':err,'ok':ok,'tensor_sha256':sha(sf),'trace_sha256':sha(tf),'parent_tensor_sha256':ref['tensor_sha256'],'parent_trace_sha256':ref['trace_sha256']});print('FRESH_GREEDY_ANCHOR',name,ok,flush=True)
reusable=all(x['ok'] for x in items);dump(out/'result.json',{'ok':True,'baseline_reuse':reusable,'anchors':items,'readout_frozen':scorer.verify_frozen(),'policy_hash':lora_hash(model),'fallback':'fresh baseline generation if any anchor differs'});print('BASELINE_REUSE',reusable,flush=True)

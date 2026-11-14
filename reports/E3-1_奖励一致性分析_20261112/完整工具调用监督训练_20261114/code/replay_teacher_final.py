import json
from pathlib import Path
from milestone_common import ROOT,PLAN,setup,seed,dump,case_images,load_policy,lora_hash
from checkpoint_io import restore_adapter_exact
from supervised_format import response_loss

torch=setup();seed(0)
result=json.loads((ROOT/'training/result.json').read_text());checkpoint=ROOT/'training/final/checkpoint'
model,processor=load_policy(checkpoint,False);restored=restore_adapter_exact(model,checkpoint)
assert lora_hash(model)==result['final_parameter_hash']
examples=json.loads((ROOT/'training/teacher_examples.json').read_text());cases=json.loads((ROOT/'inputs/cases.json').read_text())
images=case_images([x for x in cases if x['image'] in PLAN['training_images']])
reference={x['example_id']:x for x in json.loads((ROOT/'training/teacher_evaluation_final.json').read_text())['examples']}
eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos)
maximum=0.;records=[]
with torch.no_grad():
    for example in examples:
        value,stats=response_loss(model,processor,images[example['image']]['image'],example,eos)
        ref=reference[example['id']];error=abs(float(value)-ref['loss_sum']);maximum=max(maximum,error)
        assert error<1e-4 and stats['target_ids_active']==ref['stats']['target_ids_active']
        assert max(abs(a-b) for a,b in zip(stats['nll_per_active_token'],ref['stats']['nll_per_active_token']))<1e-4
        records.append({'example_id':example['id'],'replayed_loss_sum':float(value),'original_loss_sum':ref['loss_sum'],'absolute_error':error})
assert lora_hash(model)==result['final_parameter_hash']
dump(ROOT/'teacher_replay.json',{'ok':True,'examples_replayed':len(records),'maximum_abs_loss_error':maximum,
 'restoration':restored,'parameter_hash':result['final_parameter_hash'],'records':records})
print('ALL_30_TEACHER_LOSSES_REPLAYED_FROM_SAVED_CHECKPOINT',maximum,flush=True)

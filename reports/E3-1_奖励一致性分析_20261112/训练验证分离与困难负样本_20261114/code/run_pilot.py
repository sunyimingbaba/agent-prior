import json
import math
import time

import numpy as np
from milestone_common import ROOT,PLAN,setup,seed,dump,sha,table_images,load_policy,real_tools,lora_hash
from split_reward import SplitReward
from live_rollout import rollout,tensor_sha
torch=setup();start=time.monotonic();seed(0)
from agent.rollout import SYSTEM_PROMPT
out=ROOT/'pilot';out.mkdir(exist_ok=False)
model,processor=load_policy();initial=lora_hash(model);assert initial==PLAN['inherited_initial_hash']
scorer=SplitReward('train',PLAN['training_rows']);tools=real_tools();images=table_images(PLAN['training_rows'],'train')
records=[];features=[];similarities=[];groups=[]
for step,row in enumerate(PLAN['training_rows']):
    group=[]
    for g in range(4):
        sample_seed=700000+step*4+g;seed(sample_seed)
        hidden,trace=rollout(model,processor,images[row]['image'],images[row]['caption'],tools,SYSTEM_PROMPT,PLAN['rollout'])
        stem=f'{step:03d}_{row}_{g}';path=out/'priors'/(stem+'.pt');path.parent.mkdir(exist_ok=True);torch.save(hidden,path)
        values=scorer.score(hidden,row);fmt=.2 if trace['quality_usable'] else 0.
        dump(out/'traces'/(stem+'.json'),trace)
        x={'id':'pilot__'+stem,'row':row,'split':'train','step':step,'sample':g,'seed':sample_seed,
           'image':images[row]['name'],'state_file':str(path.relative_to(ROOT)),'state_file_sha256':sha(path),
           'generation_state_sha256':tensor_sha(hidden),'scorer_state_sha256':tensor_sha(hidden),
           'trace_file':str((out/'traces'/(stem+'.json')).relative_to(ROOT)),
           'quality_usable':trace['quality_usable'],'tools_used':trace['tools_used'],'r_format':fmt,**values}
        records.append(x);group.append(x);features.append(scorer.last_feature);similarities.append(scorer.last_similarities)
        with (out/'records.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
    advantages=[]
    for metric in ['r_global','r_scene']:
        rewards=torch.tensor([x['r_format']+x[metric] for x in group],dtype=torch.float32)
        sd=rewards.std();adv=(rewards-rewards.mean())/(sd+1e-8) if sd>1e-8 else torch.zeros_like(rewards)
        advantages.append(adv)
    maximum=float((advantages[0]-advantages[1]).abs().max())
    groups.append({'row':row,'max_standardized_advantage_difference':maximum,'additional_signal':maximum>.05,
                   'all_global_negatives_same_scene':all(x['global_negative_is_same_scene'] for x in group)})
    dump(ROOT/'progress.json',{'phase':'pilot','completed':step+1,'total':len(PLAN['training_rows'])})
    print('PILOT',step+1,'/',len(PLAN['training_rows']),maximum,flush=True)
assert lora_hash(model)==initial
torch.save({'record_ids':[x['id'] for x in records],'image_features':torch.cat(features),
            'similarities':torch.stack(similarities),'text_features':scorer.text_features},out/'reward_tensors.pt')
required=math.ceil(.25*len(groups));actual=sum(x['additional_signal'] for x in groups);enable=actual>=required
policies=['baseline','format_only','global_margin']+(['scene_margin'] if enable else [])
dump(out/'branch_decision.json',{'ok':True,'scene_arm_enabled':enable,'required_groups':required,'actual_groups':actual,
    'policies':policies,'arms':policies[1:],'initial_lora_tensor_hash':initial,'groups':groups,
    'global_negative_same_scene_count':sum(x['global_negative_is_same_scene'] for x in records),
    'samples':len(records),'quality_pass':sum(x['quality_usable'] for x in records),'elapsed_seconds':time.monotonic()-start})
print('PILOT_BRANCH_DECISION',enable,actual,'/',required,flush=True)

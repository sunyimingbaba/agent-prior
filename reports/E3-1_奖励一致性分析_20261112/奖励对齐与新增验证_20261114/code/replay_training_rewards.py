import json
import time

from milestone_common import ROOT,PLAN,setup,dump,sha,real_tools,table_images
from downstream_reward import FinalRetrievalReward
from live_rollout import tensor_sha
torch=setup();start=time.monotonic()
assert json.loads((ROOT/'run_result.json').read_text())['status']=='complete'
out=ROOT/'training_reward_verification';out.mkdir(exist_ok=False)
reward=FinalRetrievalReward(PLAN['training_rows']+PLAN['smoke_rows'])
rows=[];features=[];sims=[];max_error=0.
for arm in ['smoke','aligned_margin']:
    for line in (ROOT/'training'/arm/'samples.jsonl').read_text().splitlines():
        x=json.loads(line);hidden=torch.load(ROOT/x['state_file'],map_location='cpu',weights_only=True)
        assert sha(ROOT/x['state_file'])==x['state_file_sha256']
        assert tensor_sha(hidden)==x['generation_state_sha256']==x['scorer_state_sha256']
        v=reward.score(hidden,x['row']);error=abs(v['r_aligned']-x['r_aligned']);max_error=max(max_error,error)
        assert error<1e-6 and v['actual_rank0']==x['actual_rank0']
        assert x['r_format']==(.2 if x['quality_usable'] else 0.) and x['r_total']==x['r_format']+x['r_aligned']
        rows.append({'id':arm+'__'+x['id'],'arm':arm,'source_id':x['id'],'row':x['row'],**v})
        features.append(reward.last_feature);sims.append(reward.last_similarities)
torch.save({'record_ids':[x['id'] for x in rows],'image_features':torch.cat(features),
            'actual_similarities':torch.stack(sims),'text_features':reward.text_features},out/'tensors.pt')
dump(out/'records.json',rows)
tools=real_tools();images=table_images(set(PLAN['validation_rows'])|set(PLAN['training_rows']))
actual={row:tools['scene_classify'](item['image']) for row,item in images.items()};events=0
all_records=[]
for arm in ['smoke','aligned_margin']:all_records += [json.loads(x) for x in (ROOT/'training'/arm/'samples.jsonl').read_text().splitlines()]
for policy in PLAN['evaluation']['policy_points']:all_records += [json.loads(x) for x in (ROOT/'evaluation'/policy/'records.jsonl').read_text().splitlines()]
for x in all_records:
    trace=json.loads((ROOT/x['trace_file']).read_text());reference=actual[x['row']]
    for e in trace['tool_events']:
        assert e['text']==reference['text'] and abs(e['confidence']-reference['confidence'])<1e-6;events+=1
dump(out/'actual_tool_replays.json',actual)
dump(out/'result.json',{'ok':True,'training_rewards_replayed':len(rows),'maximum_margin_replay_error':max_error,
    'all_training_ranks_equal':True,'real_tool_rows_recomputed':len(actual),'tool_events_checked':events,
    'elapsed_seconds':time.monotonic()-start})
print('ALIGNED_TRAINING_REWARDS_AND_TOOLS_VERIFIED',len(rows),events,flush=True)

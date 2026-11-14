import json
import time

from milestone_common import ROOT,PLAN,setup,seed,dump,sha,table_images,real_tools
from split_reward import SplitReward
from live_rollout import tensor_sha
torch=setup();seed(0);start=time.monotonic();out=ROOT/'replay_verification';out.mkdir(exist_ok=False)
branch=json.loads((ROOT/'pilot/branch_decision.json').read_text())
sources=[ROOT/'pilot/records.jsonl']+[ROOT/'training'/arm/'samples.jsonl' for arm in ['smoke']+branch['arms']]
scorer=SplitReward('train',PLAN['training_rows']);count=0;maximum=0.;records=[]
for path in sources:
    for line in path.read_text().splitlines():
        x=json.loads(line);assert x['split']=='train' and sha(ROOT/x['state_file'])==x['state_file_sha256']
        hidden=torch.load(ROOT/x['state_file'],map_location='cpu',weights_only=True)
        assert tensor_sha(hidden)==x['generation_state_sha256']==x['scorer_state_sha256']
        value=scorer.score(hidden,x['row'])
        for metric in ['r_global','r_scene']:
            error=abs(x[metric]-value[metric]);maximum=max(error,maximum);assert error<1e-6
        assert x['rank0']==value['rank0']
        records.append({'source':str(path.relative_to(ROOT)),'id':x['id'],'row':x['row'],**value});count+=1
dump(out/'training_reward_replays.json',records)
tools=real_tools();actual={}
for split,rows in [('train',PLAN['training_rows']),('val',PLAN['validation_rows'])]:
    for row,item in table_images(rows,split).items():actual[split+':'+str(row)]=tools['scene_classify'](item['image'])
events=0
sources += [ROOT/'evaluation'/policy/'records.jsonl' for policy in branch['policies']]
for path in sources:
    for line in path.read_text().splitlines():
        x=json.loads(line);reference=actual[x['split']+':'+str(x['row'])]
        for e in json.loads((ROOT/x['trace_file']).read_text())['tool_events']:
            assert e['text']==reference['text'] and abs(e['confidence']-reference['confidence'])<1e-6;events+=1
dump(out/'actual_tool_replays.json',actual)
dump(out/'result.json',{'ok':True,'training_and_pilot_rewards_replayed':count,'max_margin_replay_error':maximum,
     'real_tool_images_recomputed':len(actual),'tool_events_checked':events,'elapsed_seconds':time.monotonic()-start})
print('TRAINING_REWARDS_AND_ACTUAL_TOOLS_VERIFIED',count,events,flush=True)

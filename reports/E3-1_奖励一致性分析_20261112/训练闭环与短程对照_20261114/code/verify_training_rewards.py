"""Frozen-model replay of every training reward and every real tool result."""
import json
import time

from milestone_common import ROOT,PLAN,dump,sha,setup,load_proxy,real_tools,table_images
from live_rollout import tensor_sha
torch=setup()

assert json.loads((ROOT/'run_result.json').read_text())['status']=='complete'
start=time.monotonic();out=ROOT/'training_reward_verification';out.mkdir(exist_ok=False)
scorer=load_proxy();records=[];features=[];sims=[]
all_rows=set(PLAN['validation_rows'])|set(PLAN['training_rows'])|set(PLAN['smoke_rows'])
images=table_images(all_rows);tools=real_tools();tool_results={}
for row,item in images.items():tool_results[row]=tools['scene_classify'](item['image'])
tool_events=0
for arm in ['smoke','retrieval','format_only']:
    samples=[json.loads(x) for x in (ROOT/'training'/arm/'samples.jsonl').read_text().splitlines()]
    lam=1. if arm=='smoke' else PLAN['arms'][arm]['lambda_retrieval']
    for x in samples:
        hidden=torch.load(ROOT/x['state_file'],map_location='cpu',weights_only=True)
        assert sha(ROOT/x['state_file'])==x['state_file_sha256']
        assert tensor_sha(hidden)==x['generation_state_sha256']==x['scorer_state_sha256']
        captured=[];original=scorer.pl.model.encode_image
        def capture(*args,**kwargs):
            value=original(*args,**kwargs);captured.append(value.detach().clone());return value
        scorer.pl.model.encode_image=capture
        try:rank,reward=scorer.score(hidden.float().tolist(),x['row'])
        finally:scorer.pl.model.encode_image=original
        assert rank==x['proxy_rank0'] and reward==x['r_retrieval']
        assert x['r_format']==(.2 if x['quality_usable'] else 0.)
        assert x['r_total']==x['r_format']+lam*reward
        feature=captured[0]/captured[0].norm(dim=1,keepdim=True)
        ps=(feature@scorer.image_features.cuda().T).cpu().flatten()
        records.append({'id':arm+'__'+x['id'],'arm':arm,'source_id':x['id'],'row':x['row'],
                        'proxy_rank0':rank,'r_retrieval':reward,'r_total':x['r_total']})
        features.append(feature.cpu());sims.append(ps)
        trace=json.loads((ROOT/x['trace_file']).read_text())
        expected=tool_results[x['row']]
        for e in trace['tool_events']:
            assert e['text']==expected['text'] and abs(e['confidence']-expected['confidence'])<1e-6
            tool_events+=1
for policy in PLAN['evaluation']['policy_points']:
    for line in (ROOT/'evaluation'/policy/'records.jsonl').read_text().splitlines():
        x=json.loads(line);trace=json.loads((ROOT/x['trace_file']).read_text());expected=tool_results[x['row']]
        for e in trace['tool_events']:
            assert e['text']==expected['text'] and abs(e['confidence']-expected['confidence'])<1e-6
            tool_events+=1
torch.save({'record_ids':[x['id'] for x in records],'proxy_features':torch.cat(features),
            'proxy_similarities':torch.stack(sims)},out/'tensors.pt')
dump(out/'records.json',records)
dump(out/'actual_tool_replays.json',tool_results)
dump(out/'result.json',{'ok':True,'training_rewards_replayed':len(records),'all_training_proxy_ranks_rewards_exact':True,
    'tool_rows_recomputed':len(tool_results),'training_and_evaluation_tool_events_checked':tool_events,
    'elapsed_seconds':time.monotonic()-start})
print('TRAINING_REWARDS_AND_REAL_TOOLS_VERIFIED',len(records),tool_events,flush=True)

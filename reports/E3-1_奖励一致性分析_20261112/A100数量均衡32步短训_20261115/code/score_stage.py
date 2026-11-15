import json
import time
from milestone_common import ROOT,PLAN,setup,seed,case_images,real_tools
from quantity_model import QuantityScorer
from live_rollout_eos import tensor_sha
from stage_io import read,dump,sha
torch=setup();seed(0);start=time.monotonic();assert read(ROOT/'training_pipeline_result.json')['status']=='complete'
cases=read(ROOT/'inputs/evaluation_cases.json');by={x['image']:x for x in cases};scorer=QuantityScorer(read(ROOT/'inputs/cases.json'))
own={arm:[json.loads(s) for s in (ROOT/arm/'records.jsonl').read_text().splitlines()] for arm in ['before','after']};records=[]
for arm in ['before','after']:
    assert len(own[arm])==425;records+=own[arm]
    ref=next(x for x in own[arm] if x['image']=='102.tif' and x['role']=='greedy')
    for x in cases:
        records.append({**ref,'id':arm+'_shared__'+x['image'],'arm':arm+'_shared','image':x['image'],'split':x['split'],'row':x['row'],'role':'shared','reference_image':'102.tif','seed':0})
training=[json.loads(s) for s in (ROOT/'training/samples.jsonl').read_text().splitlines()];assert len(training)==128
records += [{**x,'arm':'training','role':'candidate'} for x in training]
out=ROOT/'scoring';out.mkdir();scores=[];features=[];qs=[];vs=[];vids=[];replay_error=0.
for i,x in enumerate(records,1):
    h=torch.load(ROOT/x['state_file'],map_location='cpu',weights_only=True);assert sha(ROOT/x['state_file'])==x['state_file_sha256'] and tensor_sha(h)==x['generation_state_sha256']
    value=scorer.score(h,by[x['image']])
    if x['arm'] in ['before','training'] and 'A_positive' in x:
        for m in ['A_positive','A_negative','B_positive','B_negative','margin_A','margin_B']:
            error=abs(value[m]-x[m]);replay_error=max(replay_error,error);assert error<1e-6
        assert value['retrieval_rank0']==x['retrieval_rank0']
    scores.append({**x,**value,'scorer_state_sha256':tensor_sha(h)});features.append(scorer.last_feature);qs.append(scorer.last_qsim)
    if scorer.last_vsim is not None:vids.append(x['id']);vs.append(scorer.last_vsim)
    if i%50==0:print('STAGE_SCORED',i,'/1148',flush=True)
assert len(scores)==1148 and len({x['id'] for x in scores})==1148
dump(out/'all_scores.json',scores);torch.save({'record_ids':[x['id'] for x in scores],'image_features':torch.cat(features),'quantity_similarities':torch.stack(qs),'val_record_ids':vids,'val_similarities':torch.stack(vs)},out/'tensors.pt')
tools=real_tools();actual={n:tools['scene_classify'](v['image']) for n,v in case_images(cases).items()};events=0
for x in scores:
    if x['role']=='shared':continue
    tr=read(ROOT/x['trace_file'])
    for e in tr['tool_events']:
        assert e['origin']=='model_requested_actual_tool' and e['text']==actual[x['image']]['text'] and abs(e['confidence']-actual[x['image']]['confidence'])<1e-6;events+=1
dump(out/'actual_tools.json',actual)
dump(out/'result.json',{'ok':True,'records':1148,'before_own':425,'after_own':425,'shared_controls':170,'training_samples_replayed':128,'maximum_replay_abs_error':replay_error,'real_tool_events_checked':events,'actual_tool_images':85,'elapsed_seconds':time.monotonic()-start})
print('ALL_1148_SCORES_AND_TOOLS_VERIFIED',flush=True)

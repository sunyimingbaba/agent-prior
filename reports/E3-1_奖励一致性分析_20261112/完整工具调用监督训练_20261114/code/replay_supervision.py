import json
import time

from milestone_common import ROOT,PLAN,setup,seed,dump,case_images,real_tools,sha
from quantity_model import QuantityScorer
from live_rollout_legacy import tensor_sha
torch=setup();seed(0);start=time.monotonic();out=ROOT/'replay_verification';out.mkdir(exist_ok=False)
cases=json.loads((ROOT/'inputs/cases.json').read_text());by={x['image']:x for x in cases};scorer=QuantityScorer(cases)
scores=json.loads((ROOT/'scoring/all_scores.json').read_text())
max_error=0.;count=0
for x in scores:
    h=torch.load(ROOT/x['state_file'],map_location='cpu',weights_only=True)
    assert sha(ROOT/x['state_file'])==x['state_file_sha256'] and tensor_sha(h)==x['generation_state_sha256']==x['scorer_state_sha256']
    current=scorer.score(h,by[x['image']])
    for metric in ['A_positive','A_negative','B_positive','B_negative','margin_A','margin_B']:
        error=abs(x[metric]-current[metric]);max_error=max(max_error,error);assert error<1e-6
    if x['split']=='val':assert x['retrieval_rank0']==current['retrieval_rank0']
    count+=1
tools=real_tools();actual={name:tools['scene_classify'](item['image']) for name,item in case_images(cases).items()};events=0
for x in scores:
    for e in json.loads((ROOT/x['trace_file']).read_text())['tool_events']:
        ref=actual[x['image']];assert e['text']==ref['text'] and abs(e['confidence']-ref['confidence'])<1e-6;events+=1
dump(out/'actual_tool_replays.json',actual)
dump(out/'result.json',{'ok':True,'quantity_score_records_replayed':count,'maximum_abs_score_error':max_error,
     'all_val_retrieval_ranks_equal':True,'actual_tool_images_recomputed':100,'tool_events_checked':events,'elapsed_seconds':time.monotonic()-start})
print('ALL_SFT_SCORES_AND_ACTUAL_TOOLS_REPLAYED',count,events,flush=True)

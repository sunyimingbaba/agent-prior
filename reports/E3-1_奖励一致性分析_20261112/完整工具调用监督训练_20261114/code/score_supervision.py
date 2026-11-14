import json
import time
from milestone_common import ROOT,PLAN,setup,seed,dump,sha
from quantity_model import QuantityScorer
from live_rollout_legacy import tensor_sha
torch=setup();seed(0);start=time.monotonic();out=ROOT/'scoring';out.mkdir(exist_ok=False)
cases=json.loads((ROOT/'inputs/cases.json').read_text());by={x['image']:x for x in cases};scorer=QuantityScorer(cases)
assert json.loads((ROOT/'training_pipeline_result.json').read_text())['status']=='complete'
records=[]
for arm in PLAN['evaluation']['arms']:
    records.extend(json.loads(s) for s in (ROOT/'evaluation'/arm['name']/'records.jsonl').read_text().splitlines())
assert len(records)==935
scores=[];features=[];qsim=[];vsim=[];vids=[]
for i,x in enumerate(records,1):
    assert sha(ROOT/x['state_file'])==x['state_file_sha256']
    hidden=torch.load(ROOT/x['state_file'],map_location='cpu',weights_only=True)
    assert tensor_sha(hidden)==x['generation_state_sha256']==x['cache_state_sha256']
    value=scorer.score(hidden,by[x['image']]);fmt=.2 if x['quality_usable'] else 0.
    scores.append({**x,**value,'r_format':fmt,'r_total_quantity':fmt+value['margin_A'],
        'r_total_quality_gated':fmt+(value['margin_A'] if x['quality_usable'] else 0.),'scorer_state_sha256':tensor_sha(hidden)})
    features.append(scorer.last_feature);qsim.append(scorer.last_qsim)
    if scorer.last_vsim is not None:vids.append(x['id']);vsim.append(scorer.last_vsim)
    dump(ROOT/'progress.json',{'phase':'scoring','completed':i,'total':935});print('SCORED',i,'/935',flush=True)
index={x['id']:x for x in scores};assert len(index)==935
dump(out/'all_scores.json',scores)
for arm in PLAN['evaluation']['arms']:dump(out/(arm['name']+'.json'),[x for x in scores if x['arm']==arm['name']])
torch.save({'record_ids':[x['id'] for x in scores],'image_features':torch.cat(features),'quantity_similarities':torch.stack(qsim),
    'val_record_ids':vids,'val_similarities':torch.stack(vsim)},out/'tensors.pt')
dump(out/'result.json',{'ok':True,'all_records':935,'primary_policy_records':425,'compatibility_records':85,'val_records':len(vids),'elapsed_seconds':time.monotonic()-start})

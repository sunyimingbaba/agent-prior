import json
import time

from milestone_common import ROOT,PLAN,setup,seed,dump,sha
from split_reward import SplitReward
torch=setup();seed(0);start=time.monotonic()
out=ROOT/'scoring';out.mkdir(exist_ok=False)
branch=json.loads((ROOT/'pilot/branch_decision.json').read_text())
scorer=SplitReward('val',PLAN['validation_rows'])
records=[]
for policy in branch['policies']:
    assert json.loads((ROOT/'evaluation'/policy/'result.json').read_text())['ok']
    records.extend(json.loads(x) for x in (ROOT/'evaluation'/policy/'records.jsonl').read_text().splitlines())
scores=[];features=[];similarities=[]
for i,x in enumerate(records,1):
    assert x['split']=='val' and sha(ROOT/x['state_file'])==x['state_file_sha256']
    hidden=torch.load(ROOT/x['state_file'],map_location='cpu',weights_only=True)
    value=scorer.score(hidden,x['row']);scores.append({**x,'utility_rank0':value['rank0'],'utility_rr':value['rr'],
                                                     'utility_margin':value['r_global']})
    features.append(scorer.last_feature);similarities.append(scorer.last_similarities)
    dump(ROOT/'progress.json',{'phase':'scoring','completed':i,'total':len(records)})
    print('SCORED',i,'/',len(records),flush=True)
dump(out/'scores.json',scores)
torch.save({'record_ids':[x['id'] for x in scores],'image_features':torch.cat(features),
            'similarities':torch.stack(similarities),'text_features':scorer.text_features},out/'tensors.pt')
dump(out/'result.json',{'ok':True,'records':len(records),'split':'val','elapsed_seconds':time.monotonic()-start})

import json
from pathlib import Path
import shutil

from milestone_common import ROOT,PLAN,setup,dump,sha
from quantity_model import QuantityScorer
torch=setup();cases=json.loads((ROOT/'inputs/cases.json').read_text());scorer=QuantityScorer(cases)
out=ROOT/'historical_anchors';out.mkdir(exist_ok=False);(out/'priors').mkdir()
records=[];features=[];qsim=[];maximum=0.
for item in cases:
    path=Path('/private/AGSE_E1_20261110/ucm_prior_archive/agent_cache_domain_tool')/item['historical_prior_relative_path']
    assert sha(path)==item['historical_prior_sha256'];target=out/'priors'/(item['image']+'.pt');shutil.copy2(path,target)
    h=torch.load(path,map_location='cpu',weights_only=True);scores=scorer.score(h,item)
    for w in ['A','B']:
        for sign in ['positive','negative']:
            error=abs(scores[w+'_'+sign]-item['pairs'][w]['historical_'+sign+'_score']);maximum=max(maximum,error);assert error<1e-6
    records.append({'image':item['image'],'state_file':str(target.relative_to(ROOT)),'state_file_sha256':sha(target),**scores})
    features.append(scorer.last_feature);qsim.append(scorer.last_qsim)
torch.save({'images':[x['image'] for x in cases],'image_features':torch.cat(features),'quantity_similarities':torch.stack(qsim)},out/'tensors.pt')
dump(out/'records.json',records);dump(out/'result.json',{'ok':True,'images':100,'historical_scalar_scores_checked':400,'historical_pairs_checked':200,'maximum_abs_score_error':maximum})
print('ALL_100_HISTORICAL_QUANTITY_ANCHORS_PASSED',maximum,flush=True)

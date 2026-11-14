import json

from milestone_common import ROOT,SOURCE,PLAN,setup,seed,dump,sha
from downstream_reward import FinalRetrievalReward
torch=setup();seed(0)
reward=FinalRetrievalReward([22])
source_gallery=json.loads((SOURCE/'scoring/text_gallery.json').read_text())
for row,cols in source_gallery['positive_gallery_columns'].items():assert reward.positives[int(row)]==cols
record=next(x for x in json.loads((SOURCE/'scoring/scores.json').read_text()) if x['id']=='22_reference_0')
hidden=torch.load(SOURCE/record['prior_file'],map_location='cpu',weights_only=True)
actual=reward.score(hidden,22)
assert actual['actual_rank0']==record['utility_rank0']
assert abs(actual['r_aligned']-record['utility_margin'])<1e-6
assert torch.equal(reward.text_features,torch.load(SOURCE/'scoring/scoring_tensors.pt',map_location='cpu',weights_only=True)['text_features'])
dump(ROOT/'reward_bridge_witness.json',{'ok':True,'all_210_positive_sets_from_dataset':True,
    'previous_22_positive_sets_equal':True,'reference_id':record['id'],
    'actual_rank_equal':True,'absolute_margin_error':abs(actual['r_aligned']-record['utility_margin']),
    'all_original_text_features_reused_bitwise':True,'actual':actual,
    'reward_model_sha256':PLAN['utility_model_sha256']})
print('FINAL_RETRIEVAL_REWARD_WITNESS_PASSED',flush=True)

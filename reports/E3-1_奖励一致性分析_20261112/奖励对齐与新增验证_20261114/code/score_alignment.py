import json
from pathlib import Path
import shutil
import time

from milestone_common import ROOT,PLAN,SOURCE,dump,sha,setup,seed,load_proxy
from downstream_reward import FinalRetrievalReward
torch=setup();seed(0)
start=time.monotonic();out=ROOT/'scoring';out.mkdir(exist_ok=False)
verify=ROOT/'verification_inputs';verify.mkdir()
for source,name in [(SOURCE/'scoring/scoring_tensors.pt','original_scoring_tensors.pt'),
    (SOURCE/'scoring/text_gallery.json','text_gallery.json'),
    (SOURCE.parent/'proxy_gallery_20261113/grpo_val_cache_rebuilt_last50.pt','proxy_gallery.pt')]:shutil.copy2(source,verify/name)
scorer=load_proxy();reward=FinalRetrievalReward(PLAN['validation_rows'])
dump(out/'positive_columns.json',reward.positives)
old_scores={x['id']:x for x in json.loads((Path(PLAN['previous_stage_root'])/'scoring/scores.json').read_text())}
records=[]
for policy in PLAN['evaluation']['policy_points']:
    assert json.loads((ROOT/'evaluation'/policy/'result.json').read_text())['ok']
    records += [json.loads(x) for x in (ROOT/'evaluation'/policy/'records.jsonl').read_text().splitlines()]
assert len(records)==864
scores=[];images=[];proxies=[];actual=[];proxy_sims=[];anchors=[]
for n,record in enumerate(records,1):
    assert sha(ROOT/record['state_file'])==record['state_file_sha256']
    h=torch.load(ROOT/record['state_file'],map_location='cpu',weights_only=True)
    captured=[];original=scorer.pl.model.encode_image
    def capture(*args,**kwargs):
        value=original(*args,**kwargs);captured.append(value.detach().clone());return value
    scorer.pl.model.encode_image=capture
    try:rank,ret=scorer.score(h.float().tolist(),record['row'])
    finally:scorer.pl.model.encode_image=original
    proxy=captured[0]/captured[0].norm(dim=1,keepdim=True)
    with torch.no_grad():ps=(proxy@scorer.image_features.cuda().T).cpu().flatten()
    value=reward.score(h,record['row']);fmt=.2 if record['quality_usable'] else 0.
    entry={**record,'proxy_rank0':rank,'r_proxy':ret,'r_format':fmt,'r_aligned':value['r_aligned'],
        'r_total_proxy':fmt+ret,'r_total_aligned':fmt+value['r_aligned'],
        'utility_rank0':value['actual_rank0'],'utility_rr':value['r_actual_rr'],'utility_margin':value['r_aligned']}
    if record['reused_previous']:
        old=old_scores[record['id']];error=abs(old['utility_margin']-entry['utility_margin'])
        assert old['proxy_rank0']==rank and old['utility_rank0']==entry['utility_rank0'] and error<1e-6
        anchors.append({'id':record['id'],'both_ranks_equal':True,'margin_abs_error':error})
    scores.append(entry);images.append(reward.last_feature);proxies.append(proxy.cpu())
    actual.append(reward.last_similarities);proxy_sims.append(ps)
    dump(ROOT/'progress.json',{'phase':'scoring','completed':n,'total':864})
    print('SCORED',n,'/864',record['id'],flush=True)
dump(out/'scores.json',scores)
torch.save({'record_ids':[x['id'] for x in scores],'text_features':reward.text_features,
    'image_features':torch.cat(images),'proxy_features':torch.cat(proxies),
    'actual_similarities':torch.stack(actual),'proxy_similarities':torch.stack(proxy_sims)},out/'scoring_tensors.pt')
dump(out/'verification.json',{'ok':True,'records':864,'all_original_text_features_reused_bitwise':True,
    'all_210_positive_sets_from_original_dataset':True,'legacy_score_anchors':anchors,
    'verification_input_sha256':{p.name:sha(p) for p in verify.iterdir()},'elapsed_seconds':time.monotonic()-start})
print('SCORING_COMPLETE',flush=True)

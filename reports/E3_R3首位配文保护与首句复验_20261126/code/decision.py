"""Apply the pre-specified comparisons and save a compact machine-readable decision."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results'
def read(name):
    return json.loads((RESULT/name).read_text())
def values(tag, arm, baseline='ordinary'):
    return read('priority_'+tag+'.json')['against'][baseline][arm]


def main():
    audit=read('核对与归因.json')
    out={}
    for arm in ('guard','margin_guard','rank_priority'):
        fresh=values('first98',arm)
        change=values('first98',arm,'calibrated')
        old=audit['first98']['arms']['calibrated']
        new=audit['first98']['arms'][arm]
        observed=dict(
            fresh_detail_both_seeds=min(fresh['detail']['seeds'])>0,
            fresh_mrr=fresh['mrr']['mean']>=0,
            fresh_net=fresh['net']['mean']>=0,
            fresh_mrr_over_calibrated=change['mrr']['mean']>=0,
            fresh_lower_drop_frequency=new['drop_rate_if_original_first']<old['drop_rate_if_original_first'],
            fresh_lower_first_loss=new['first_cost']>old['first_cost'],
            positive_pooled_detail_interval=values('pooled_scene',arm)['detail']['interval'][0]>0,
            historical_scene148_mrr=values('scene148',arm)['mrr']['mean']>=0,
            historical_scene122_mrr=values('scene122',arm)['mrr']['mean']>=0)
        if arm=='margin_guard':
            observed['historical_first78_mrr']=values('first78',arm)['mrr']['mean']>=0
        if arm=='rank_priority':
            observed['cv_all_positive']=all(v['mean']>0 for v in values('dev',arm).values())
            observed['pooled_above_rank_only']=values('pooled_scene',arm,'rank_only')['detail']['mean']>0
            observed['fresh_above_rank_only']=values('first98',arm,'rank_only')['detail']['mean']>0
        out[arm]=dict(checks=observed,all_pass=all(observed.values()))
    (RESULT/'晋级判定.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()

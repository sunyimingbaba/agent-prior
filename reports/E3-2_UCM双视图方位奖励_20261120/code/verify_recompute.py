"""Compare independent host reruns against private-server statistics and keep the numerical receipts."""
import argparse
import csv
import json
from pathlib import Path
from shared import ROOT,sha,dump

def isclose(a,b):
    if type(a)!=type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(isclose(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(isclose(x,y) for x,y in zip(a,b,strict=True))
    if isinstance(a,float):return abs(a-b)<1e-12
    return a==b

def main(folder):
    out=ROOT/'results'
    names=['states.json','decision.json','e1_replay.json','paired_contrasts.json','alternative_scoring.json',
           'training_audit.json','checkpoint_index.json','scene_sensitivity.json','quarter_stats.json','prior_swap.json','credit_summary.json',
           'retrieval_by_source.csv','detail_by_source.csv','e1_by_source.csv','behavior_by_source.csv','same_candidate_credit.csv',
           'e1_paired_contrasts.csv','direction_transitions.csv','quarter_by_source.csv','prior_swap_by_source.csv','检查点汇总.csv','E1分项汇总.csv']
    proof={}
    for name in names:
        local,server=out/name,folder/name
        a,b=sha(local),sha(server)
        r=dict(local_sha=a,server_sha=b,byte_equal=a==b)
        if a!=b:
            assert local.suffix=='.json',name
            left,right=[json.loads(f.read_bytes()) for f in (local,server)]
            assert isclose(left,right),name
            r.update(numerically_equal_at_1e_minus12=True)
            if name=='alternative_scoring.json':
                r['roundoff_local']=left['maximum_gram_rounding_difference']
                r['roundoff_server']=right['maximum_gram_rounding_difference']
        proof[name]=r
    dump(out/'local_recompute.json',dict(outputs=proof,files=len(proof),byte_equal=sum(r['byte_equal'] for r in proof.values()),
         tolerance_for_auxiliary_floats=1e-12,canonical_files='server'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('server_results',type=Path);main(p.parse_args().server_results)

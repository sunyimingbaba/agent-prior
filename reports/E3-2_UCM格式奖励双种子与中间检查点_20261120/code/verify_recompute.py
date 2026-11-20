"""Compare locally recomputed statistics to the saved server results, retaining both numerical receipts."""
import argparse
from shared import ROOT,sha,read,dump
from pathlib import Path

def main(server):
    folder=ROOT/'results'
    outputs=['retrieval_by_source.csv','detail_by_source.csv','e1_by_source.csv','behavior_by_source.csv','states.json','decision.json',
             'e1_replay.json','paired_contrasts.json','e1_paired_contrasts.csv','alternative_scoring.json','deployment_checks.json',
             'format_counterfactual_groups.csv','format_credit.json','检查点汇总.csv','E1分项汇总.csv','屏蔽首步结束符.csv']
    proof={}
    for file in outputs:
        a,b=folder/file,server/file
        proof[file]=dict(local_sha256=sha(a),server_sha256=sha(b),byte_equal=a.read_bytes()==b.read_bytes())
        if not proof[file]['byte_equal']:
            assert file=='alternative_scoring.json'
            aa,bb=read(a),read(b);proof[file].update(local=aa,server=bb)
            diagnostic='maximum_gram_rounding_difference'
            assert all(abs(x[diagnostic])<1e-12 for x in (aa,bb))
            assert {k:v for k,v in aa.items() if k!=diagnostic}=={k:v for k,v in bb.items() if k!=diagnostic}
    dump(folder/'local_recompute.json',dict(outputs=proof,checks=len(outputs),byte_equal=sum(v['byte_equal'] for v in proof.values()),
                                            stored_alternative_scoring='Server copy; local auxiliary rounding diagnostic retained in this receipt.'))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('server_results',type=Path);main(ap.parse_args().server_results)

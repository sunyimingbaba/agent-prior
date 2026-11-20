"""Diagnose between-run drift before the original format signal was active."""
import argparse
from pathlib import Path
import numpy as np
import torch
from shared import ROOT,read,dump
from metrics import fixed,recalls

def compare(left,right):
    detail=[]
    for a,b in zip(left,right,strict=True):
        changed=sum([q['tokens'] for q in t['rounds']]!=[q['tokens'] for q in u['rounds']] for t,u in zip(a['traces'],b['traces'],strict=True))
        detail.append(dict(step=a['step'],identical_log_and_tokens=a['traces']==b['traces'],differing_token_trajectories=changed,
                      same_rewards=a['rewards']==b['rewards'],same_margins=a['caption_and_hard_margins']==b['caption_and_hard_margins'],
                      left_format=a.get('format_score'),right_format=b.get('format_score')))
    return detail

def main(previous):
    p=ROOT/'numerical_control'
    aa=read(p/'policy_a_off/steps.json');ab=read(p/'policy_a_on/steps.json');prior=read(previous/'policy_hard/steps.json')
    matched=compare(aa,ab); replay=compare(aa,prior)
    f=torch.load(p/'eval/a_off_24/native.pt',weights_only=True,map_location='cpu')['visual']
    text=torch.load(ROOT/'inputs/retrieval_text.pt',weights_only=True,map_location='cpu')
    values=fixed(f,text['native'])[:,text['caption_order']]
    owner=np.arange(210)[:,None]==np.repeat(np.arange(210),5)[None,:]
    score=np.mean([v.mean() for d in (recalls(values,owner),recalls(values.T,owner.T)) for v in d.values()])
    ref=read(previous/'results/summary.json')['native']['hard']['mR']['mean']
    out=dict(default_paired_groups=matched,previous_vs_default_off=replay,default_off_native_mR=score,previous_off_native_mR=ref,
             first_paired_different_candidates=next(x['step'] for x in matched if x['differing_token_trajectories']),
             first_paired_group_with_any_differential_format=next(x['step'] for x in matched if any(len(set(x[k]))>1 for k in ('left_format','right_format'))))
    dump(p/'numerical_comparison.json',out)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('previous',type=Path);main(ap.parse_args().previous)

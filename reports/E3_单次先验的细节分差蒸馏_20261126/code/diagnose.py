from collections import Counter
import numpy as np
import torch
from local import ROOT,OLD,RFPO,load,read,dump
from greedy_behavior import records

def main():
    cases=read(RFPO/'inputs/panel.json');bank=load(RFPO/'records/text.pt')
    lost=read(OLD/'results/mined_detail_changes.json')['records'];out=[];groups=[]
    for s in 'ab':
        old=records(RFPO/'archives'/(s+'_deployed.tar.gz'),True)
        after=records(OLD/'archives'/('mined_policy_'+s+'_selected_joint.tar.gz'),False)
        vs=[load(p)['features'].double() for p in (RFPO/'vectors'/(s+'_deployed_primary.pt'),OLD/'mined_policy'/(s+'_selected_joint')/'evaluate/primary.pt')]
        for r in (r for r in lost if r['seed']==s):
            n=r['image'];i=next(j for j,c in enumerate(cases) if c['image']==n)
            ix=bank['maps']['primary'][n][r['wording']];v=bank['hard'][ix[0]].double()-bank['hard'][ix[1]].double()
            margin=[float(f[i]@v) for f in vs];assert margin[0]>0>=margin[1]
            a=old['primary',n];b=after['primary',n];assert a[3:]==b[3:]
            h,h2=a[2].float(),b[2].float()
            out.append(dict(**r,margin=margin,actions_equal=a[0]==b[0],tool_feedback_equal=a[1]==b[1],actions=a[0],
                         relative_hidden_change=float(torch.linalg.vector_norm(h-h2)/torch.linalg.vector_norm(h))))
        for z in read(OLD/'mined_policy'/(s+'_selected_joint')/'steps.json'):
            rew=np.array(z['rewards']['joint']);ids=np.flatnonzero(rew==rew.max());paths=[z['paths'][int(i)] for i in ids];anchor=z['anchor']
            groups.append(dict(seed=s,image=z['image'],changed_winner=not np.array_equal(rew==rew.max(),np.array(z['rewards']['ordinary'])==max(z['rewards']['ordinary'])),
                has_joint=bool(max(z['joint_fraction'])),top_count=len(paths),all_match_greedy=all(p==anchor for p in paths),
                share_full=sum(p==anchor for p in paths)/len(paths),share_first=sum(p[0]==anchor[0] for p in paths)/len(paths)))
    valid=[z for z in groups if z['has_joint']]
    dump(ROOT/'results/diagnosis.json',dict(five=out,training_groups=groups,groups_with_good=len(valid),
        good_groups_winner_same_first=float(np.mean([g['share_first'] for g in valid])),
        good_groups_winner_same_full=float(np.mean([g['share_full'] for g in valid]))))
if __name__=='__main__':main()

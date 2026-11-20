"""Change only the format term on the very same eight sampled trajectories; summarize how often credit actually changes."""
import re
from collections import Counter
import numpy as np
from shared import ROOT,read,dump
from metrics import csvwrite
from analyze import study_states

def sign(x):return 1 if x>0 else -1 if x<0 else 0
def normalize(x):
    sd=np.std(x,ddof=1)
    return (x-x.mean())/(sd+1e-8)

def main():
    groups=[];result={}
    for arm in read(ROOT/'inputs/protocol.json')['arms']:
        steps=read(ROOT/('policy_'+arm)/'steps.json');rows=[]
        for z in steps:
            rewards=np.array(z['semantic_midrank'],dtype=float)
            fmt=np.array(z['format_score']);aug=rewards+7*fmt
            top=np.flatnonzero(rewards==rewards.max()).tolist()
            new=np.flatnonzero(aug==aug.max()).tolist()
            fmt_yes=fmt>0
            pairs=[(i,j) for i in range(8) for j in range(8) if fmt_yes[i] and not fmt_yes[j]]
            changes=[(sign(rewards[i]-rewards[j]),sign(aug[i]-aug[j])) for i,j in pairs]
            change=normalize(aug)-normalize(rewards)
            row=dict(arm=arm,step=z['step'],image=z['image'],rewarded=int(fmt_yes.sum()),mixed=int(bool(pairs)),
                     changed_top=int(top!=new),mixed_pairs=len(pairs),
                     overtook=sum(a<0 and b>0 for a,b in changes),orders_changed=sum(a!=b for a,b in changes),
                     max_advantage_change=float(np.max(abs(change))),mean_advantage_gain_rewarded=float(change[fmt_yes].mean()) if fmt_yes.any() else None)
            groups.append(row);rows.append(row)
        result[arm]={k:sum(r[k] for r in rows) for k in ('rewarded','mixed','changed_top','mixed_pairs','overtook','orders_changed')}
        result[arm]['max_advantage_change']=max(r['max_advantage_change'] for r in rows)
    out=ROOT/'results';csvwrite(out/'format_counterfactual_groups.csv',groups)
    summaries={}
    for state in study_states():
        folder=ROOT/'eval'/state
        d={}
        for part in read(folder/'complete.json')['parts']:
            traces=read(folder/(part+'.json'))['records'];ctrl=Counter();full=Counter()
            for r in traces:
                joined='\n'.join(q['text'] for q in r['rounds'])
                matches=re.findall(r'<answer>(.*?)</answer>',joined,flags=re.DOTALL)
                assert bool(matches)==bool(r['format'])
                ctrl['complete_answer']+=bool(matches)
                ctrl['answer_contains_letter_or_digit']+=any(re.search(r'[^\W_]',x) is not None for x in matches)
                ctrl['first_eos']+=r['immediate_eos']
                ctrl['real_tool']+=bool(r['events'])
                ctrl['count']+=1
                full[joined]+=1
            d[part]={**ctrl,'distinct_output_text':len(full),'most_common_text_count':full.most_common(1)[0][1]}
        summaries[state]=d
    dump(out/'format_credit.json',dict(training_same_candidate_scores=result,output_text=summaries))
if __name__=='__main__': main()

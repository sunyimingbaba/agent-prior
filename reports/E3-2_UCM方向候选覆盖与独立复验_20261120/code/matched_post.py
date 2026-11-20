from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import numpy as np
import torch
from cases import cases
from shared import ROOT,read,dump,sha
from analyze import endpoint,sign_test
from metrics import fixed,csvwrite,ci
def main():
    lock=read(ROOT/'matched_choice/complete.json');rows=[];summary={};scores=[];total=0
    for group in ('train','fresh'):
        tally=Counter();by=defaultdict(list)
        for row in cases(group):
            name=row['image'];folder=ROOT/'matched_choice'/group
            meta=read(folder/(name+'.json'));assert sha(folder/(name+'.json'))==lock[group][name] and sha(folder/(name+'.pt'))==meta['bundle']
            assert meta['input']==sha(ROOT/'choice'/group/(name+'.json')) and meta['source']==sha(ROOT/'code/matched_choice.py')
            assert meta['design']==sha(ROOT/'inputs/matched_choice_protocol.json') and meta['all_vocab_logits_exact']
            d=torch.load(folder/(name+'.pt'),weights_only=True,map_location='cpu')
            texts=torch.load(ROOT/'outputs'/group/(Path(name).stem+'.pt'),weights_only=True,map_location='cpu')['text']
            sc=fixed(d['visual'],texts);assert len(meta['rows'])==len(sc)==8
            vectors=defaultdict(list)
            for z,h,s in zip(meta['rows'],d['hidden'],sc,strict=True):
                assert hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()==z['hidden']
                w='AB'.index(z['word']);a,b=z['logits']
                cup=s[2*w]-s[2*w+1];agent=(a-b)*(1-2*z['order'])
                vectors[z['word'],z['view']].append((cup,agent))
                scores.append(dict(group=group,image=name,word=z['word'],view=z['view'],order=z['order'],cup=cup,agent=agent))
                total+=1
            for w in 'AB':
                u=np.mean(vectors[w,0],axis=0);r=np.mean(vectors[w,1],axis=0)
                cz=endpoint(group,row,w,'cup_same_state',u[0],r[0]);az=endpoint(group,row,w,'agent_same_state',u[1],r[1]);rows.extend([cz,az])
                tally['both_heads_correct']+=cz['both'] and az['both'];tally['only_cup']+=cz['both'] and not az['both']
                tally['only_agent']+=az['both'] and not cz['both'];tally['neither']+=not (az['both'] or cz['both'])
                by[name].append(az['both']-cz['both'])
        summary[group]=dict(paired_transitions=dict(tally),difference_ci=ci([np.mean(v) for v in by.values()]))
        for method in ('cup_same_state','agent_same_state'):
            sel=[r for r in rows if r['group']==group and r['method']==method];bins=defaultdict(list)
            for x in sel:bins[x['image']].append(x['centered'])
            summary[group][method]=dict(n=len(sel),both=sum(r['both'] for r in sel),pairacc=np.mean([r['pairacc'] for r in sel]),centered=np.mean([r['centered'] for r in sel]),
                                        p=sign_test([np.mean(z) for z in bins.values()]))
    csvwrite(ROOT/'results/matched_choice_margins.csv',scores);csvwrite(ROOT/'results/matched_choice_by_source.csv',rows)
    dump(ROOT/'results/matched_choice.json',dict(groups=summary,shared_states=total))
if __name__=='__main__':main()

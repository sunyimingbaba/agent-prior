from collections import defaultdict
import numpy as np
from shared import ROOT,read,sha,dump
from cases import cases
from metrics import csvwrite,ci
from analyze import endpoint,sign_test

def main():
    receipt=read(ROOT/'choice_backbone/replay.json')
    assert receipt['known_on_prompts_exact']==12 and receipt['source']==sha(ROOT/'code/backbone_choice.py')
    rows=[];summaries=[]
    for group in ('train','fresh'):
        for row in cases(group):
            a=ROOT/'choice'/group/(row['image']+'.json')
            b=read(ROOT/'choice_backbone'/group/a.name);assert b['source']==sha(a)
            for name,records in [('enabled',read(a)['records']),('disabled',b['records'])]:
                for w in 'AB':
                    raw=[];mass=[]
                    for view in range(3):
                        obs=sorted((z for z in records if z['word']==w and z['view']==view),key=lambda v:v['order'])
                        assert len(obs)==2
                        raw.append(((obs[0]['logit_a']-obs[0]['logit_b'])-(obs[1]['logit_a']-obs[1]['logit_b']))/2)
                        if view!=2:mass.extend(z['valid_mass'] for z in obs)
                    out=endpoint(group,row,w,name,*raw[:2])
                    rows.append(dict(**out,gray=raw[2],mass=np.mean(mass)))
        selected=[r for r in rows if r['group']==group]
        for name in ('enabled','disabled'):
            x=[r for r in selected if r['method']==name];by=defaultdict(list)
            for r in x:by[r['image']].append(r['centered'])
            summaries.append(dict(group=group,method=name,both=sum(r['both'] for r in x),n=len(x),pairacc=np.mean([r['pairacc'] for r in x]),
                  centered=np.mean([r['centered'] for r in x]),p_sign=sign_test([np.mean(v) for v in by.values()]),mass=np.mean([r['mass'] for r in x])))
    csvwrite(ROOT/'results/backbone_by_source.csv',rows);csvwrite(ROOT/'results/backbone_table.csv',summaries)
if __name__=='__main__':main()

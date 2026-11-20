"""Evaluate same-source +90/-90 and all four angles using the independently generated raw priors."""
import csv
import hashlib
from collections import defaultdict
import numpy as np
import torch
from shared import ROOT,read,sha,dump
from metrics import fixed,ci,csvwrite

def main():
    stage=ROOT/'results';inputs=ROOT/'inputs';questions=read(inputs/'quarter_questions.json')
    ext=read(inputs/'quarter_protocol.json')
    assert sha(inputs/'quarter_questions.json')==ext['questions'] and sha(inputs/'detail.json')==ext['original']
    with (stage/'detail_by_source.csv').open(encoding='utf-8-sig') as f:
        half={(r['state'],r['image']):r for r in csv.DictReader(f) if r['part']=='detail'}
    states=['base_00']+[f'{s}_{arm}_16' for s in 'ab' for arm in ('single','separate','joint')]
    records=[];sm={};by={}
    for state in states:
        folder=ROOT/'quarter_eval'/state;proof=read(folder/'complete.json');primary=read(ROOT/'eval'/state/'complete.json')
        assert proof['active']==primary['active'] and proof['readout']==primary['readout']
        receipt=read(folder/'quarter.json')
        assert sha(folder/'quarter.json')==proof['parts']['quarter'] and sha(folder/'quarter.pt')==receipt['bundle'] and receipt['active']==proof['active']
        assert sha(folder/'quarter_text.pt')==proof['train_text']
        data=torch.load(folder/'quarter.pt',weights_only=True,map_location='cpu')
        text=torch.load(folder/'quarter_text.pt',weights_only=True,map_location='cpu')
        assert [r['image'] for r in receipt['records']]==[r['image'] for r in questions for _ in range(2)]
        scores=fixed(data['visual'],text['features'])
        for r,h in zip(receipt['records'],data['hidden'],strict=True):
            assert r['hidden_sha']==hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()
        rows=[]
        for i,r in enumerate(questions):
            row=dict(state=state,image=r['image'],block=r['block']);qcount=0;fcount=0;weak=[];fourweak=[]
            other=half[state,r['image']]
            for w in 'CD':
                pos,neg=[text['phrases'].index(q) for q in r[w]]
                vals=[]
                for k,(view,sign) in enumerate([('cw90',1),('ccw90',-1)]):
                    tr=receipt['records'][2*i+k];assert tr['view']==view and tr['query']==r['scene_query']
                    val=float((scores[2*i+k,pos]-scores[2*i+k,neg])*sign)
                    row[w+'_'+view]=val;vals.append(val)
                absolute_min=min(vals);all_min=min(*vals,float(other[w+'_upright']),float(other[w+'_rotated']))
                qcount+=int(absolute_min>0);fcount+=int(all_min>0)
                weak.append(absolute_min);fourweak.append(all_min)
                row[w+'_quarter_both']=int(absolute_min>0);row[w+'_all_four']=int(all_min>0)
            row.update(quarter_both=qcount/2,all_four=fcount/2,quarter_weak=np.mean(weak),all_four_weak=np.mean(fourweak))
            rows.append(row);records.append(row)
        sm[state]=dict(quarter_both=int(2*sum(r['quarter_both'] for r in rows)),all_four=int(2*sum(r['all_four'] for r in rows)),
                       out_of=32,quarter_weak=float(np.mean([r['quarter_weak'] for r in rows])),all_four_weak=float(np.mean([r['all_four_weak'] for r in rows])))
        by[state]={r['image']:r for r in rows}
    contrasts={}
    for a,b in [('joint','separate'),('joint','single'),('joint','base')]:
        for seed in ('a','b','mean'):
            ss='ab' if seed=='mean' else seed
            p={}
            for metric in ('quarter_both','all_four','quarter_weak','all_four_weak'):
                diffs=[]
                for s in ss:
                    x=by[s+'_'+a+'_16'];y=by['base_00' if b=='base' else s+'_'+b+'_16']
                    names=sorted(x);assert names==sorted(y)
                    diffs.append([x[n][metric]-y[n][metric] for n in names])
                p[metric]=ci(np.mean(diffs,axis=0))
            contrasts[f'{seed}_{a}-{b}']=p
    csvwrite(stage/'quarter_by_source.csv',records);dump(stage/'quarter_stats.json',dict(states=sm,contrasts=contrasts))

if __name__=='__main__':main()

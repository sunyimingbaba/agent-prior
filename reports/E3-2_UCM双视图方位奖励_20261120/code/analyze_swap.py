"""Summarize the crossed frozen readouts from saved vectors for identical source images."""
import csv
import numpy as np
import torch
from shared import ROOT,read,dump,sha
from metrics import fixed,csvwrite
def main():
    plan=read(ROOT/'inputs/prior_swap_protocol.json');rows=[];summary={}
    texts=torch.load(ROOT/'inputs/retrieval_text.pt',weights_only=True,map_location='cpu')
    sources=read(ROOT/'inputs/detail.json')
    with (ROOT/'results/detail_by_source.csv').open(encoding='utf-8-sig') as fp:
        ref={(r['state'],r['image']):r for r in csv.DictReader(fp) if r['part']=='detail'}
    for state in plan['states']:
        path=ROOT/'readout'/(state+'.pt')
        binding=read(path.with_suffix('.json'))
        assert binding['bundle']==sha(path) and binding['images']==[r['image'] for r in sources]
        assert binding['protocol']==sha(ROOT/'inputs/prior_swap_protocol.json')
        z=torch.load(path,weights_only=True,map_location='cpu')
        sims=fixed(z.flatten(0,2),texts['relation']).reshape(16,2,2,-1)
        group=[]
        for m,r,sc in zip(binding['images'],sources,sims,strict=True):
            for w in 'CD':
                a,b=map(texts['texts'].index,r[w])
                s00,s01,s10,s11=(sc[:,:,a]-sc[:,:,b]).ravel()
                source=ref[state,m]
                assert s00==float(source[w+'_upright']) and s11==-float(source[w+'_rotated'])
                pix=(s00+s01-s10-s11)/4;prior=(s00+s10-s01-s11)/4
                assert abs(pix+prior-(s00-s11)/2)<1e-14
                row=dict(state=state,image=m,word=w,s00=s00,s01=s01,s10=s10,s11=s11,
                         natural_both=int(s00>0 and s11<0),swapped_both=int(s01>0 and s10<0),
                         fixed_original=int(s00>0 and s10<0),fixed_rotated=int(s01>0 and s11<0),
                         natural_signed=(s00-s11)/2,pixel_component=pix,hidden_component=prior)
                group.append(row)
        rows.extend(group)
        stats={name:sum(r[name] for r in group) for name in ('natural_both','swapped_both','fixed_original','fixed_rotated')}
        stats.update({name:float(np.mean([r[name] for r in group])) for name in ('natural_signed','pixel_component','hidden_component')})
        summary[state]=stats
    csvwrite(ROOT/'results/prior_swap_by_source.csv',rows)
    dump(ROOT/'results/prior_swap.json',summary)
if __name__=='__main__':main()

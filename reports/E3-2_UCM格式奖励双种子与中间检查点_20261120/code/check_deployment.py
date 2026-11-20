"""Verify baseline and reference-policy replay and the unchanged cases of the one-token intervention."""
import argparse
import torch
from shared import ROOT, dump, read, sha
from pathlib import Path

def main(prior):
    same={}
    for new,old in [('base_00','base'),('a_off_24','hard')]:
        p=ROOT/'eval'/new;q=prior/'eval'/old
        row={}
        for part in ['native','generic','detail']:
            ref=torch.load(q/'vectors.pt',weights_only=True,map_location='cpu')[part]
            mapping={tuple(key):i for i,key in enumerate(ref['order'])}
            now=torch.load(p/(part+'.pt'),weights_only=True,map_location='cpu')
            counts={'n':0,'hidden':0,'tokens':0,'readout_exact':0,'max_readout_difference':0.}
            for i,r in enumerate(read(p/(part+'.json'))['records']):
                k=mapping[r['image'],r['view'],-1]; source=q/part/(Path(r['image']).stem+'_'+r['view']+'.json')
                original=read(source);old_trace=original['traces'][0]
                assert r['query']==original['query'] and r['rendered_sha']==original['rendered_sha256'] and old_trace['k']==-1
                counts['n']+=1; counts['hidden']+=r['hidden_sha']==old_trace['hidden']
                counts['tokens']+=r['rounds']==old_trace['rounds']
                v=ref['visual'][k];x=now['visual'][i]
                counts['readout_exact']+=torch.equal(v,x)
                counts['max_readout_difference']=max(counts['max_readout_difference'],float((v-x).abs().max()))
            row[part]=counts
        same[new]=row
    forced={}
    for seed in ('a','b'):
        for arm in ('off','on'):
            f=ROOT/'eval'/f'{seed}_{arm}_24_force';n=ROOT/'eval'/f'{seed}_{arm}_24'
            by={}
            for part in ('native','detail'):
                normal=read(n/(part+'.json'));control=read(f/(part+'.json'));assert normal['active']==control['active']
                nv=torch.load(n/(part+'.pt'),weights_only=True,map_location='cpu')['visual'];fv=torch.load(f/(part+'.pt'),weights_only=True,map_location='cpu')['visual']
                triggered=0;unchanged=0;equal=0
                for idx,(a,b) in enumerate(zip(normal['records'],control['records'],strict=True)):
                    for k in ('image','row','view','query','seed','rendered_sha'):assert a[k]==b[k]
                    assert not b['immediate_eos']
                    if a['immediate_eos']: triggered+=1
                    else:
                        unchanged+=1
                        equal+=int(a['events']==b['events'] and a['rounds']==b['rounds'] and a['hidden_sha']==b['hidden_sha'] and torch.equal(nv[idx],fv[idx]))
                assert unchanged==equal
                by[part]=dict(triggered=triggered,unaffected=unchanged,unchanged_exact=equal)
            forced[seed+'_'+arm]=by
    assert all(z['n']==z['tokens']==z['hidden'] for z in same['base_00'].values())
    dump(ROOT/'results/deployment_checks.json',dict(prior_stage=same,force=forced))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('previous',type=Path);main(parser.parse_args().previous)

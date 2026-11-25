"""Paired ablation of only the ordinary-caption priority gate, retaining the exact reward ordering and calibration."""
import sys
import numpy as np
from reward_resolution import ROOT,OLD,PRIOR,read,sha,dump,load,bank,rr,K,calibrate,boot
def main(tag):
    roster=read(ROOT/'inputs'/(tag+'.json'));b=bank(OLD if tag=='first' else PRIOR/('guard' if tag=='third' else 'reward_confirm'))
    fname='controlled_timing_third' if tag=='third' else tag+'_images'
    ref={(r['image'],r['seed']):r['free' if tag=='third' else 'scene'] for r in read(ROOT/'records'/(fname+'.json'))}
    rows=[]
    for s in 'ab':
        base=ROOT/'records'/tag/s if tag!='third' else ROOT/'records/prefix/third'/s
        for c in roster:
            nm=c['image']
            if (nm,s) not in ref:continue
            f=load(base/'features'/(nm+'.pt'))['modes'];z=f['free' if tag=='third' else 'scene'];v=z['visual'].double().numpy()
            if tag=='third':v[-1]=f['locked']['visual'][-1].double().numpy()
            ix=b['maps'][nm];p0=ix['native'][0];pos=[p0]+sorted(set(ix['native'])-{p0},key=b['hashes'].__getitem__)
            mask=np.ones(len(b['even']),bool);mask[pos]=False
            ns=v@b['native'].T;hs=v@b['hard'].T;left=ns[:,mask&b['even']];right=ns[:,mask&~b['even']]
            margin=ns[:16,p0]-left[:16].max(1);hard=hs[:16,ix['A'][0]]-hs[:16,ix['A'][1]]
            rank=np.asarray(z['ranks'][:16]);rew=calibrate(margin+K*hard,rank);win=np.flatnonzero(rew==rew.max())
            detail=np.mean([hs[:,ix[k][0]]>hs[:,ix[k][1]] for k in 'CDE'],axis=0)
            vals=[]
            for held in pos[1:]:
                ranks=np.array([rr(score[held],negative) for score,negative in zip(ns,right)],object)
                good=(ranks[:16]>ranks[16])&(detail[:16]>detail[16])
                vals.append([good[win].mean(),ranks[win].astype(float).mean(),detail[win].mean()])
            new=dict(zip(('joint','mrr','detail'),np.mean(vals,0)))
            rows.append(dict(image=nm,seed=s,pair=c['pair'],original=ref[nm,s],without=new))
    summary={}
    for m in ('joint','mrr','detail'):
        old=np.array([r['original'][m] for r in rows]);new=np.array([r['without'][m] for r in rows])
        summary[m]={k:boot(y,np.ones(len(y)),[r['pair'] for r in rows]) for k,y in [('original',old),('without',new),('change',new-old)]}
    dump(ROOT/'results'/('without_gate_'+tag+'.json'),summary);dump(ROOT/'records'/('without_gate_'+tag+'_images.json'),rows)
if __name__=='__main__':main(sys.argv[1])

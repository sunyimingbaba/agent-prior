"""Recheck the alternative-caption UCM gallery using the independently replayed deployment evidence."""
import numpy as np
from device_readout import ROOT,FORM,read,dump,load
from transfer_scores import counts,summary,scalar_values
def main():
    gal=load(FORM/'results/ucm_text.pt');expected=read(FORM/'results/paired_results.json')['ucm'];out={}
    for mode in ('native','alternatives'):
        before=[];after=[]
        for s in 'ab':
            v=load(ROOT/'records/device'/(s+'_official.pt'))['vectors'];ref=load(FORM/'results'/('ucm_'+s+'_R2.pt'))
            ids,base,ds=counts(ref['features'],gal,mode)
            ids2,b,ds2=counts(v[:,0],gal,mode);ids3,a,ds3=counts(v[:,1],gal,mode)
            assert np.array_equal(ids,ids2) and np.array_equal(ids,ids3) and np.array_equal(ds,ds2) and np.array_equal(ds,ds3)
            native=scalar_values(base.sum(0),np.asarray(len(ds)),np.asarray(ds.sum()))[-1]
            assert abs(100*native-expected['all_'+mode]['levels'][s+'_R2']['mR'])<1e-10
            assert np.allclose(b,base,rtol=0,atol=1e-10)
            before.append(b);after.append(a)
        B=np.mean(before,0);A=np.mean(after,0)
        out[mode]=dict(images=len(ids),captions=int(ds.sum()),before=summary(B,ds),after=summary(A,ds),change=summary(A-B,ds))
    dump(ROOT/'results/device_ucm_gallery.json',out)
if __name__=='__main__':main()

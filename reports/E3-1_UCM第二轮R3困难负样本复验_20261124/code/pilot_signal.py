"""Count gradients that would change on the same on-policy group when the hard term is added."""
import numpy as np
from analyze import ROOT,read,dump
from learning_signal import reward,advantage
def main():
    k=read(ROOT/'inputs/continuous_lock.json')['kappa'];out={}
    for s in ('a','b'):
        for method in ('R2','R3'):
            steps=read(ROOT/'pilot'/(s+'_'+method)/'steps.json');changed=[];flipped=[]
            for z in steps:
                c=z['scored'];rank,native,A=[np.array([r[key] for r in c]) for key in ('rank','native','A')]
                a,b=[advantage(z['format']+reward(order,rank)) for order in (native,native+k*A)]
                if abs(a-b).max()>1e-5:changed.append(z['image'])
                if a@b< -1e-8:flipped.append(z['image'])
                assert np.allclose(a if method=='R2' else b,z['advantages'],atol=1e-7)
            out[s+'_'+method]=dict(groups=len(steps),changed=len(changed),changed_images=changed,opposite=len(flipped))
    dump(ROOT/'pilot/signal.json',out)
if __name__=='__main__':main()

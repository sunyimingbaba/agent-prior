"""Counterfactual caption and negative-gallery reads, with identical held-out comparisons."""
import numpy as np
from panels import ROOT,read,dump,question_panels,transfer_panel
from gate import I,J,BASE,rr_matrix,tiescore
from run_stage import boot
from collections import defaultdict

def main():
    nominee=read(ROOT/'inputs/nomination.json')['rule']
    panels={}
    for tag in ('development','repeat','transfer'):
        primary='scene' if tag=='transfer' else 'question'
        raw=transfer_panel() if tag=='transfer' else question_panels(tag)
        images=defaultdict(list);pair={}
        for c,seed,mode,b,v,_,_,_,_ in raw:
            if mode!=primary:continue
            nm=c['image'];ix=b['maps'][nm];p=ix['native'];pos=[p[0]]+sorted(set(p)-{p[0]},key=b['hashes'].__getitem__)
            x=v[:16]@b['native'].T;keep=np.ones(len(b['even']),bool);keep[pos]=False
            tr=rr_matrix(x[:,pos],x[:,keep&b['even']]);te=rr_matrix(x[:,pos],x[:,keep&~b['even']])
            cos=b['native'][pos]@b['native'][pos].T;folds=[]
            for j in range(1,len(pos)):
                actual=te[:,j].astype(float);mask=actual[I]!=actual[J]
                if not mask.any():continue
                refs=[i for i in range(1,len(pos)) if i!=j];allow=[0]+refs
                if nominee=='diverse_two':use=[0,min(refs,key=lambda i:cos[0,i])]
                elif nominee=='central_two':use=[0,max(refs,key=lambda i:np.mean([cos[i,q] for q in allow if q!=i]))]
                elif nominee=='first':use=[0]
                else:use=allow
                def signal(bank,refs,minimum=False):
                    a=bank[:,refs];return (a.min(1) if minimum else a.sum(1)/len(refs)).astype(float)
                base=signal(tr,[0,refs[0]])
                pred=signal(tr,use,nominee=='min_all')
                other=signal(te,use,nominee=='min_all')
                def ties(y):return float(np.mean(y[I][mask]==y[J][mask]))
                folds.append([tiescore(pred,actual),tiescore(other,actual),tiescore(tr[:,j].astype(float),actual),
                    tiescore(actual,actual),ties(pred),ties(base)])
            if folds:images[nm].append(np.mean(folds,axis=0));pair[nm]=c['pair']
        vals=np.stack([np.mean(rows,axis=0) for rows in images.values()]);labels=[pair[x] for x in images]
        names=['ordinary','same_negatives','same_caption','identical','nominated_ties','baseline_ties']
        out={k:boot(vals[:,i],np.ones(len(vals)),labels) for i,k in enumerate(names)}
        out['change_only_negatives']=boot(vals[:,1]-vals[:,0],np.ones(len(vals)),labels)
        out['change_only_caption']=boot(vals[:,2]-vals[:,0],np.ones(len(vals)),labels)
        assert np.allclose(vals[:,3],1)
        expected=read(ROOT/'results'/(tag+'.json'))['levels'][nominee]['concordance']['mean']
        assert abs(expected-out['ordinary']['mean'])<1e-10
        panels[tag]=out
    dump(ROOT/'results/error_sources.json',panels)
if __name__=='__main__':main()

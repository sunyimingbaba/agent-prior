"""Verified frozen reference and separate actual-gallery scores for each independent single draw."""
import numpy as np
from rosters import ROOT,read,sha,dump
from report_scores import load
from single_scores import calculate,RFPO,FORM,summary_by_seed
from run_stage import boot

def main():
    results={};records=[]
    for sub,rows in [('primary',read(RFPO/'inputs/panel.json')),('official',read(FORM/'inputs/ucm.json'))]:
        labels=[r.get('pair',r['image']) for r in rows]
        groups={m:{} for m in ('greedy','draw1','draw2','average','change')};actions={};checks={}
        avail=[s for s in 'ab' if (ROOT/'records/stochastic'/s/(sub+'_read.json')).is_file()]
        if not avail:continue
        for seed in avail:
            base=ROOT/'records/stochastic'/seed;f=base/(sub+'_features.pt');proof=read(base/(sub+'_read.json'))
            assert proof['features']==sha(f) and proof['archive']==sha(base/(sub+'.tar.gz'));z=load(f)
            ref=load(RFPO/'vectors'/(seed+'_deployed_'+sub+'.pt'));v=z['visual']
            assert ref['adapter']==z['adapter']==proof['adapter'] and ref['images']==[r['image'] for r in rows]==[r['image'] for r in z['meta']]
            assert v.shape==(len(rows),3,768)
            checks[seed]=float((v[:,0]-ref['features']).abs().max());assert checks[seed]<1e-4
            ref_scores=calculate(sub,ref['features'],rows)
            for i,m in enumerate(('greedy','draw1','draw2')):groups[m][seed]=calculate(sub,v[:,i],rows)
            for k,val in ref_scores.items():assert np.allclose(val,groups['greedy'][seed][k],atol=1e-12,rtol=0)
            groups['average'][seed]={k:(groups['draw1'][seed][k]+groups['draw2'][seed][k])/2 for k in ref_scores}
            groups['change'][seed]={k:groups['average'][seed][k]-ref_scores[k] for k in ref_scores}
            actions[seed]={k:np.mean([z1[k] for z1 in z['meta']],axis=0).tolist() for k in ('same','rounds','tokens')}
            for j,c in enumerate(rows):
                records.append(dict(image=c['image'],seed=seed,sub=sub,outcomes={m:{k:float(vv[j]) for k,vv in g[seed].items()} for m,g in groups.items()}))
        def summarize(g):
            if len(g)==2:return summary_by_seed(g,labels)
            key=next(iter(g));return {k:dict(mean=float(v.mean()),interval=boot(v,np.ones(len(labels)),labels)['interval'],
                                             seeds=[float(v.mean())]) for k,v in g[key].items()}
        results[sub]=dict(seeds=avail,metrics={k:summarize(g) for k,g in groups.items()},actions=actions,checks=checks)
    if not results:return
    complete=len(results)==2 and all(len(v['seeds'])==2 for v in results.values())
    name='single' if complete else 'partial'
    dump(ROOT/'results'/('stochastic_'+name+'.json'),results);dump(ROOT/'records'/('stochastic_'+name+'_images.json'),records)
if __name__=='__main__':main()

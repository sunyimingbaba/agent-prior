"""Check both-view accuracy under matched and independent tie/random draws for each selector."""
import csv,collections
import numpy as np
from scipy.stats import rankdata
from shared import ROOT,dump,read
from analyze_split import estimate,summary,csvout,selected
def joint(a,b,same):
    if not same:return float(a.mean()*b.mean())
    # Exact common-quantile coupling on the sorted candidate IDs.
    n=int(np.lcm(len(a),len(b)));ix=np.arange(n)
    return float((a[ix*len(a)//n]*b[ix*len(b)//n]).mean())
def main():
    for folder in ('val','object'):
        out=ROOT/folder
        with (out/'candidates.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        results=[]
        for name in sorted({z['image'] for z in rows if z['mode']=='rotated'}):
            v={}
            for mode in ('scene','rotated'):
                g=sorted([z for z in rows if z['image']==name and z['mode']==mode and int(z['k'])>=0],key=lambda z:int(z['k']))
                assert len(g)==8
                x={w:np.array([float(z[w]) for z in g]) for w in 'ABCD'}
                picks={'AB':selected(rankdata(x['A'])+rankdata(x['B'])),'random':np.arange(8)}
                v[mode]={(r,w):(x[w][idx]>0) for r,idx in picks.items() for w in 'CD'}
            e=dict(image=name,pair=g[0].get('pair',name),shadow=int(name in ('1487.tif','1988.tif')))
            for coupled in (False,True):
                tag='same_draw' if coupled else 'independent'
                for rule in ('AB','random'):
                    e[tag+'_'+rule]=np.mean([joint(v['scene'][rule,w],v['rotated'][rule,w],coupled) for w in 'CD'])
            results.append(e)
        csvout(out/'paired_choice.csv',results);stats={}
        for group in ['all']+(['entities','shadow'] if folder=='object' else []):
            subset=[r for r in results if group=='all' or r['shadow']==(group=='shadow')]
            for mode in ('same_draw','independent'):
                for rule in ('AB','random','AB_minus_random'):
                    def val(r):
                        if rule=='AB_minus_random':return r[mode+'_AB']-r[mode+'_random']
                        return r[mode+'_'+rule]
                    stats[group+'/'+mode+'/'+rule]=summary(subset,val) if folder=='val' else estimate([val(r) for r in subset])
        dump(out/'paired_choice.json',stats)
if __name__=='__main__':main()

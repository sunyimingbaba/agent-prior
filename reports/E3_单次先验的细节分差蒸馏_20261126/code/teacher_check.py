"""Look at what the locked teaching target itself delivers on disjoint pictures and other wording."""
import numpy as np
from local import ROOT,read,dump,load,banks,feature_path,bank_for,digest,summary_by_seed,paired
from prepare import build_target
from margin_scores import detail_native

def main():
    rows=read(ROOT/'inputs/holdout.json');packs=banks();labels=[r['pair'] for r in rows]
    odd=np.array([int(digest(','.join(map(str,x)))[-1],16)%2==1 for x in packs[0]['native_tokens'].tolist()])
    out={};log=[]
    for view in ('scene','first'):
        vals={a:{} for a in ('teacher','anchor')}
        for s in 'ab':
            by={k:{'CDE':[],'mrr':[]} for k in vals}
            for row in rows:
                b=bank_for(row,packs);v=load(feature_path(row,s,view))['visual']
                _,proof=build_target(row,s,view,b,odd);w=proof['winner'];z={}
                for arm,j in [('teacher',w),('anchor',16)]:
                    ds,ps=detail_native(row,v[j],b,~odd)
                    by[arm]['CDE'].append(np.mean(ds));by[arm]['mrr'].append(float(sum(ps)/len(ps)))
                    z[arm]={'correct':ds,'mrr':by[arm]['mrr'][-1]}
                log.append(dict(seed=s,view=view,image=row['image'],source=proof['file'],winner=w,**z))
            for a,v in by.items():vals[a][s]={k:np.array(z) for k,z in v.items()}
        out[view]=dict(levels={a:summary_by_seed(v,labels) for a,v in vals.items()},
            delta=paired(vals['teacher'],vals['anchor'],labels),used=sum(r['winner']!=16 for r in log if r['view']==view))
    dump(ROOT/'results/held_teachers.json',out);dump(ROOT/'results/held_teacher_images.json',log)
if __name__=='__main__':main()

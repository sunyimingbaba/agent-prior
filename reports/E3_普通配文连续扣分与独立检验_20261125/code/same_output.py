from head_scores import OUT,HEAD,LOCK,RFPO,FORM,read,load,dump,detailed,compare,sha
from compare import rr,bank
import torch,numpy as np

def main():
    cases=read(RFPO/'inputs/panel.json');official=read(FORM/'inputs/ucm.json')
    mapped={r['rgb_sha256']:i for i,r in enumerate(official)}
    shared=[c for c in cases if c['rgb_sha256'] in mapped];assert len(shared)==106
    ix=[mapped[c['rgb_sha256']] for c in shared];bd=load(RFPO/'records/text.pt');result={};records=[]
    arms=('baseline','ordinary','hard')
    for tag in ('official','fresh','extension'):
        rows=shared if tag=='official' else read(OUT/'inputs'/(tag+'.json'))
        if tag!='official':b=bank(OUT);bi=load(OUT/'records/text.pt')
        vals={k:{a:{} for a in arms} for k in (('detail',) if tag=='official' else ('detail','mrr'))}
        for a in arms:
            for s in 'ab':
                z=load(HEAD/'test'/tag/s/(a+'.pt'));assert z['nomination']==sha(LOCK)
                if tag=='official':
                    vals['detail'][a][s]=detailed(z['ordinary'][ix],bd,rows,bd['maps']['primary']).mean(1)
                else:
                    v=z['scene'];vals['detail'][a][s]=detailed(v,bi,rows,bi['maps']).mean(1)
                    scores=v.double().numpy()@b['native'].T;values=[]
                    for r,sc in zip(rows,scores):
                        ps=b['maps'][r['image']]['native'];alt=sorted(set(ps)-{ps[0]});mask=~b['even'].copy();mask[ps]=False
                        values.append(float(np.mean([float(rr(sc[p],sc[mask])) for p in alt])))
                    vals['mrr'][a][s]=np.array(values)
        result[tag]={'photos':len(rows),'metrics':{}}
        for k,vs in vals.items():
            result[tag]['metrics'][k]=dict(levels={a:float(np.mean(list(vs[a].values()))) for a in arms},
                    changes={a+'-'+ref:compare(vs[a],vs[ref],[r['pair'] for r in rows]) for a,ref in (('hard','baseline'),('ordinary','baseline'),('hard','ordinary'))})
        for i,c in enumerate(rows):
            records.append(dict(tag=tag,image=c['image'],pair=c['pair'],values={k:{a:[float(vals[k][a][s][i]) for s in 'ab'] for a in arms} for k in vals}))
    dump(OUT/'results/same_output.json',result);dump(OUT/'records/same_output_rows.json',records)
if __name__=='__main__':main()

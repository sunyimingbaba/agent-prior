"""Post-hoc question-pool ablation. Keep the difficulty score and proxy ranks, test the ordinary priority tier and a fixed same-image anchor."""
import numpy as np
from analyze import ROOT,read,dump,detail,exact
from pooled import FIRST
def main():
    k=read(ROOT.parent/'E3-1_UCM第二轮R3困难负样本复验_20261124/inputs/continuous_lock.json')['kappa']
    table=[]
    for root in [FIRST,ROOT]:
        for seed in 'ab':
            for c in read(root/'inputs/lock.json')['rows']:
                groups=read(root/'records'/seed/(c['image']+'.json'))['groups'];base=groups['scene']['scores'][-1]
                g=groups['question'];sc=g['scores'][:-1]
                v=np.array([detail(s) for s in sc]);rr=np.array([s['rr'] for s in sc])
                good=(v>detail(base))&(rr>base['rr']+1e-12)
                unconstrained=exact([s['native']+k*s['A'] for s in sc],[s['rank'] for s in sc])
                original=np.array(g['retrieval']['R3']);a=original==original.max();b=unconstrained==unconstrained.max()
                order=np.array([s['native']+k*s['A'] for s in sc])
                dense=np.unique(order,return_inverse=True)[1]
                keep=np.array([s['keep'] for s in sc])
                anchored=exact(dense+17*(keep+1e-14>=base['keep']),[s['rank'] for s in sc])
                row=dict(attribute=c['attribute'],available=int(good.any()),changed=int(np.any(a!=b)))
                for name,z in [('tier',a),('plain',b),('anchor',anchored==anchored.max())]:
                    row[name]=[float(x[z].mean()) for x in [good,v,rr]]
                table.append(row)
    attrs=sorted({r['attribute'] for r in table});draw=np.random.default_rng(20261125).integers(27,size=(20000,27))
    groups=[[r for r in table if r['attribute']==a] for a in attrs]
    assert len(groups)==27 and all(len(rs)==8 for rs in groups)
    areas={}
    for i,(label,conditional) in enumerate([('joint_when_available',True),('detail',False),('rr',False)]):
        denominators=np.array([sum(r['available'] if conditional else 1 for r in rs) for rs in groups])
        numerators=np.array([[sum(r[m][i] for r in rs) for rs in groups] for m in ['tier','plain','anchor']])
        den=denominators[draw].sum(1);valid=den>0
        delta=numerators[1]-numerators[0]
        boot=delta[draw].sum(1)[valid]/den[valid]
        areas[label]=dict(tier=float(numerators[0].sum()/denominators.sum()),plain=float(numerators[1].sum()/denominators.sum()),
            delta=float(delta.sum()/denominators.sum()),interval=np.quantile(boot,[.025,.975]).tolist())
        diff=numerators[2]-numerators[0]
        bs=diff[draw].sum(1)[valid]/den[valid]
        areas[label].update(anchor=float(numerators[2].sum()/denominators.sum()),anchor_delta=float(diff.sum()/denominators.sum()),
            anchor_interval=np.quantile(bs,[.025,.975]).tolist())
    dump(ROOT/'results/floor_check.json',dict(groups=len(table),changed_winners=sum(r['changed'] for r in table),metrics=areas))
if __name__=='__main__':main()

"""True first-caption contrast for scoring, separate positive captions and disjoint negative tokens for evaluation."""
import csv,hashlib,sys
import numpy as np
import torch
from scipy.stats import rankdata
from shared import ROOT,read,dump,sha
from analyze_split import csvout,summary,selected
def main(split):
    out=ROOT/split;ix=read(out/'index.json');rows=read(out/'rows.json');buf=torch.load(out/'features.pt',map_location='cpu',weights_only=True)
    with (out/'candidates.csv').open(encoding='utf-8-sig') as f:z=list(csv.DictReader(f))
    caps=buf['visual'].double().numpy()@buf['native'].double().numpy().T
    parity=np.array([int(hashlib.sha256(','.join(map(str,tk)).encode()).hexdigest()[-1],16)%2 for tk in buf['native_tokens'].tolist()])
    groups=[];items=[]
    for name in sorted({r['image'] for r in rows}):
        sel=[i for i,r in enumerate(rows) if r['image']==name and r['mode']=='scene' and r['k']>=0];assert len(sel)==8
        qs=caps[sel];mp=ix['native_target'][name];first=ix['native_first'][name]
        rest=[j for j in mp if j!=first]
        if not rest:continue
        negative=np.ones(len(parity),bool);negative[mp]=False
        n0=qs[:,negative&(parity==0)];n1=qs[:,negative&(parity==1)]
        native=qs[:,first]-n0.max(1);held=qs[:,rest].max(1)
        rr=1/(1+(n1>held[:,None]).sum(1));mar=held-n1.max(1)
        v={k:np.array([float(z[i][k]) for i in sel]) for k in ('A','B','C','D','old')}
        ab=(rankdata(v['A'])+rankdata(v['B']))/2
        joint=(rankdata(ab)+rankdata(native))/2
        e=dict(image=name,pair=rows[sel[0]]['pair'],first=first,other_tokens=len(rest));groups.append(e)
        for rule,inds in [('random',np.arange(8)),('old',selected(v['old'])),('AB',selected(ab)),('caption',selected(native)),('joint',selected(joint))]:
            e[rule+'_CD']=float(((v['C'][inds]>0).mean()+(v['D'][inds]>0).mean())/2)
            for metric,arr in [('RR',rr),('margin',mar)]:e[rule+'_'+metric]=float(arr[inds].mean())
        for k,i in enumerate(sel):items.append(dict(image=name,pair=e['pair'],k=rows[i]['k'],reward_margin=native[k],eval_margin=mar[k],eval_RR=rr[k],relation_rank=ab[k],joint_rank=joint[k]))
    keep=[g for g in groups if sum(h['pair']==g['pair'] for h in groups)==2]
    stats={}
    for metric in ['CD','RR','margin']:
        for rule in ['old','AB','caption','joint','random']:
            stats[rule+'_'+metric]=summary(keep,lambda r:r[rule+'_'+metric])
            if rule!='random':stats[rule+'_minus_random_'+metric]=summary(keep,lambda r:r[rule+'_'+metric]-r['random_'+metric])
    csvout(out/'disjoint_candidates.csv',items);csvout(out/'disjoint_images.csv',keep)
    dump(out/'disjoint.json',dict(images=len(keep),otherwise_eligible_images=len(groups),pool_counts=[int((parity==i).sum()) for i in [0,1]],stats=stats,
                                  plan_hash=sha(ROOT/'inputs/native_reward_comparison.json')))
if __name__=='__main__':main(sys.argv[1])

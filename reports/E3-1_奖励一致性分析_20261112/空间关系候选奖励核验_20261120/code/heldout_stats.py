"""Transfer prespecified rewards to held-out phrasing C on identical image-conditioned priors."""
import numpy as np
from scipy.stats import rankdata
import torch
from shared import ROOT,read,dump,sha
from analyze import choose,stats,save_csv,spearman
idx=read(ROOT/'heldout/index.json')
assert sha(ROOT/'heldout/text.pt')==idx['text_sha256'] and sha(ROOT/'review/heldout_C.tsv')==idx['labels_sha256']
assert sha(ROOT/'第三种措辞检验.md')==idx['protocol_sha256']
vf=torch.load(ROOT/'scoring/vectors.pt',map_location='cpu',weights_only=True)
tf=torch.load(ROOT/'heldout/text.pt',map_location='cpu',weights_only=True)
sims=vf['visual'].double().numpy()@tf['features'].double().numpy().T
rows=read(ROOT/'scoring/rows.json');scored=[]
for i,z in enumerate(rows):
    if z['image'] not in idx['columns']:continue
    pos,neg=idx['columns'][z['image']]
    scored.append(dict(mode=z['mode'],pair=z['pair'],image=z['image'],k=z['k'],reward=z['reward'],A=z['A'],B=z['B'],
                       C_pos=sims[i,pos],C_neg=sims[i,neg],C=sims[i,pos]-sims[i,neg]))
save_csv('第三措辞逐候选.csv',scored)
per=[];summary={}
for mode in ['scene','original']:
    for n in idx['columns']:
        group=[r for r in scored if r['mode']==mode and r['image']==n and r['k']>=0]
        assert len(group)==8
        a,b,c,proxy=[np.array([r[key] for r in group]) for key in ['A','B','C','reward']]
        combo=(rankdata(a)+rankdata(b))/2.
        base=np.mean(c>0);mar=c.mean()
        item=dict(image=n,mode=mode,pair=group[0]['pair'],random_acc=base,random_margin=mar,
                  mixed=int(np.min(c)<=0<np.max(c)),all3_any=int(np.any((a>0)&(b>0)&(c>0))),
                  all3_fraction=float(np.mean((a>0)&(b>0)&(c>0))))
        for key,selector in [('proxy',proxy),('A',a),('B',b),('AB',combo),('oracle',c)]:
            best,acc,t=choose(selector,c)
            item.update({key+'_acc':acc,key+'_gain_acc':acc-base,key+'_margin':best,key+'_gain_margin':best-mar,
                         key+'_rho':spearman(selector,c),key+'_ties':t})
        item['AB_vs_proxy_acc']=item['AB_acc']-item['proxy_acc']
        per.append(item)
    g=[r for r in per if r['mode']==mode]
    summary[mode]=dict(n=len(g),mixed=sum(r['mixed'] for r in g),all3_any=sum(r['all3_any'] for r in g),
                       estimates={k:stats(g,k) for k in g[0] if k not in ['image','mode','pair','mixed','all3_any']})
save_csv('第三措辞逐图.csv',per);dump(ROOT/'第三措辞汇总.json',summary)

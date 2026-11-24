import hashlib,numpy as np,torch
from candidate_guard import ROOT,OLD,read,load,dump,unpack,row_scores,summarize
def main():
    torch.set_num_threads(2);path=ROOT/'guard';bank=load(path/'records/text.pt');k=read(OLD/'inputs/continuous_lock.json')['kappa']
    t=np.array([int(hashlib.sha256(','.join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in bank['native_tokens'].tolist()]);rows=[]
    for seed in 'ab':
        rec=[unpack(path/'records',s) for s in (seed,'extra_'+seed)]
        for c in read(path/'inputs/panel.json'):
            n=c['image'];b=[r[n] for r in rec]
            assert b[0][0]['first']==b[1][0]['first'] and b[0][0]['seed']!=b[1][0]['seed'] and all(z[0]['rgb']==c['rgb_sha256'] for z in b)
            vis=torch.cat([z[1]['visual'] for z in b]);pr=sum((z[1]['proxy_rank'] for z in b),[])
            ix={l:x for l,x in bank['maps'][n].items() if l!='B'}
            sc=row_scores(bank['native'].double().numpy(),bank['hard'].double().numpy(),t,ix,vis,pr,k);rows.append(dict(seed=seed,image=n,pair=c['pair'],scores=sc))
    s=summarize(rows);levels={}
    for meth in ('R2','R3','guard'):
        levels[meth]={l:float(np.mean([r['scores'][meth][l] for r in rows])) for l in ['C','D','E','rr']}
        levels[meth]['CDE']=np.mean([levels[meth][l] for l in 'CDE'])
    dump(ROOT/'results/guard_fresh.json',dict(comparisons=s,levels=levels));dump(ROOT/'results/guard_fresh_choices.json',rows)
if __name__=='__main__':main()

import os,sys,numpy as np,torch
from dev_metrics import ROOT,REP,read,load,sha,ranked_hits,boot,dump,write_csv
sys.path.insert(0,str(REP/'E3-2_UCM与RSITMD独立复验_20261124/code'))
from metrics import fixed,recalls
def main():
    torch.set_num_threads(2);cf=ROOT/'confirm';form=REP/'E3-2_UCM与RSITMD独立复验_20261124';audit=read(form/'inputs/audit.json')
    fr=read(cf/'inputs/panel.json');vl=read(form/'inputs/ucm.json');g=load(form/'results/ucm_text.pt');bank=load(cf/'text.pt')
    assert sha(cf/'inputs/panel.json')==bank['panel']==read(cf/'inputs/lock.json')['panel']
    n=len(vl);mask=np.arange(n)[:,None]==np.arange(n).repeat(5)[None,:];cluster=[r['pair'] for r in fr];nm=[r['image'] for r in vl];fn=[r['image'] for r in fr]
    nom=read(ROOT/'inputs'/os.environ.get('E3_NOMINATION','nomination.json'));arm=nom['arm'];scores={};levels={};rows=[];checked=0;diff0={}
    for seed in 'ab':
        for a in ('start',arm):
            tag=seed+'_'+a;v=load(cf/(tag+'_official.pt'));f=load(cf/(tag+'_blind.pt'))
            assert v['images']==nm and f['images']==fn and f['adapter']==v['adapter']
            if a=='start':
                old=load(form/'results'/('ucm_'+seed+'_R2.pt'));assert old['adapter']==v['adapter']
                diff0[seed]=float((old['features']-v['features']).abs().max())
            else:assert v['adapter']==nom['adapters'][seed]
            sim=(v['features'].double()@g['native'].double().T)[:,g['caption_order']].numpy()
            ir=ranked_hits(sim,mask);tr=ranked_hits(sim.T,mask.T).reshape(n,5,3).mean(1);hits=np.column_stack((ir,tr))
            sx=fixed(v['features'],g['native'])[:,g['caption_order']]
            ch=[ar.mean() for ar in recalls(sx,mask).values()]+[ar.mean() for ar in recalls(sx.T,mask.T).values()]
            assert np.max(np.abs(hits.mean(0)-ch))<1e-10;checked+=6
            sf=fixed(f['features'],bank['hard']);out={'mR':hits.mean(1)}
            for k in 'CDE':
                ix=[bank['maps'][i][k] for i in fn];out[k]=np.array([float(r[a]>r[b]) for r,(a,b) in zip(sf,ix)])
            out['CDE']=np.stack([out[k] for k in 'CDE']).mean(0)
            scores[tag]=out;levels[tag]=dict(**{k:x.mean() for k,x in out.items()},recalls=hits.mean(0).tolist())
            for k,x in out.items():rows.extend(dict(tag=tag,metric=k,image=u,score=float(v)) for u,v in zip(nm if k=='mR' else fn,x))
    delta={}
    for k in scores['a_start']:
        z=np.stack([100*(scores[s+'_'+arm][k]-scores[s+'_start'][k]) for s in 'ab']);delta[k]=dict(**boot(z.mean(0),nm if k=='mR' else cluster),seeds=z.mean(1).tolist())
    dump(ROOT/'results/blind.json',dict(levels=levels,difference_pp=delta,baseline_replay_max_difference=diff0,independent_recalls=checked));write_csv(ROOT/'results/blind.csv',rows)
if __name__=='__main__':main()

"""Frozen development readout. The starting vectors are shared with the previous evaluation."""
import sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];REP=ROOT.parent;HIST=REP/'E3-2_UCM有效图组定向短训_20261124'
sys.path.insert(0,str(HIST/'code'))
from score import BASE,read,load,sha,ranked_hits,boot,dump,write_csv
def main(arms,label):
    torch.set_num_threads(2)
    oldpro=read(HIST/'pilot/vectors/index.json');vg=load(BASE/'retrieval_text.pt');vl=read(BASE/'validation.json');vb=load(HIST/'records/fresh_text.pt');fr=read(HIST/'inputs/panel.json');n=len(vl)
    assert sha(HIST/'inputs/panel.json')==vb['panel'];mask=np.arange(n)[:,None]==np.arange(n).repeat(5)[None,:]
    pic=[r['image'] for r in vl];pair=[r['pair'] for r in fr];allv={};levels={};rows=[]
    for seed in 'ab':
        for arm in ['start',*arms]:
            tag=seed+'_'+arm;folder=(HIST if arm=='start' else ROOT)/'pilot/vectors';items=[]
            for split in ('val','fresh'):
                pp=folder/(tag+'_'+split+'.pt');z=load(pp)
                if arm=='start':assert sha(pp)==oldpro[pp.name]
                else:assert z['adapter']==read(ROOT/'pilot'/tag/'complete.json')['final']
                items.append(z)
            v,f=items;assert v['images']==pic and f['images']==[r['image'] for r in fr] and f['adapter']==v['adapter']
            s=(v['features'].double()@vg['native'].double().T)[:,vg['caption_order']].numpy()
            ir=ranked_hits(s,mask);tr=ranked_hits(s.T,mask.T).reshape(n,5,3).mean(1);hits=np.column_stack((ir,tr))
            x=(f['features'].double()@vb['hard'].double().T).numpy();score={}
            for k in 'CDE':
                ab=[vb['maps'][nm][k] for nm in f['images']];score[k]=np.array([float(y[a]>y[b]) for y,(a,b) in zip(x,ab)])
            score['CDE']=np.mean([score[k] for k in 'CDE'],axis=0);score['mR']=hits.mean(1)
            allv[tag]=score;levels[tag]=dict(**{k:y.mean() for k,y in score.items()},recalls=hits.mean(0).tolist())
            for k,ys in score.items():
                rows.extend(dict(tag=tag,metric=k,image=nm,score=v) for nm,v in zip(pic if k=='mR' else f['images'],ys))
    comparisons={}
    for a in arms:
        for b in ['start']+[x for x in arms if x!=a and ('native' in x)]:
            diff={}
            for k in allv['a_start']:
                z=np.array([100*(allv[s+'_'+a][k]-allv[s+'_'+b][k]) for s in 'ab'])
                diff[k]=dict(**boot(z.mean(0),pic if k=='mR' else pair),seeds=z.mean(1).tolist())
            comparisons[a+'-'+b]=diff
    dump(ROOT/'results'/('dev_'+label+'.json'),dict(levels=levels,difference_pp=comparisons));write_csv(ROOT/'results'/('dev_'+label+'.csv'),rows)
if __name__=='__main__':main(sys.argv[2:],sys.argv[1])

"""Official UCM retrieval and paired held-out fine details for one generated prior per image."""
import sys
import numpy as np
from rosters import REP,ROOT,read,sha,dump
from report_scores import load
from run_stage import boot
RFPO=REP/'E3_UCM多描述奖励同预算RFPO_20261124'
FORM=REP/'E3-2_UCM与RSITMD独立复验_20261124'
HIST=REP/'E3-2_UCM有效图组定向短训_20261124'
sys.path.insert(0,str(HIST/'code'))
from score import ranked_hits
sys.path.insert(0,str(FORM/'code'))
from metrics import fixed,recalls

def calculate(sub,visual,rows):
    names=[c['image'] for c in rows]
    if sub=='official':
        g=load(FORM/'results/ucm_text.pt');n=len(rows)
        assert n==210 and len(g['caption_order'])==n*5
        mask=np.arange(n)[:,None]==np.repeat(np.arange(n),5)[None,:]
        s=fixed(visual,g['native'])[:,g['caption_order']]
        matrix=(visual.double()@g['native'].double().T)[:,g['caption_order']].numpy()
        a=np.stack(list(recalls(s,mask).values()),1);b=np.stack(list(recalls(s.T,mask.T).values()),1).reshape(n,5,3).mean(1)
        assert np.allclose(a,ranked_hits(matrix,mask),rtol=0,atol=1e-10)
        assert np.allclose(b,ranked_hits(matrix.T,mask.T).reshape(n,5,3).mean(1),rtol=0,atol=1e-10)
        z={f'{direction}_R{k}':block[:,j] for direction,block in [('I2T',a),('T2I',b)] for j,k in enumerate((1,5,10))}
        return dict(mR=np.hstack((a,b)).mean(1),I2T=a.mean(1),T2I=b.mean(1),**z)
    text=load(RFPO/'records/text.pt');s=fixed(visual,text['hard']);cols=text['maps']['primary']
    vals={}
    for k in 'CDE':
        ix=np.array([cols[n][k] for n in names]);j=np.arange(len(names))
        vals[k]=(s[j,ix[:,0]]>s[j,ix[:,1]]).astype(float)
    vals['CDE']=np.mean(list(vals.values()),axis=0)
    return vals

def summary_by_seed(values,label):
    out={}
    names=list(values['a'])
    for method in names:
        raw=np.stack([values[s][method] for s in 'ab'])
        z=boot(raw.mean(0),np.ones(len(label)),label)
        out[method]=dict(mean=z['mean'],interval=z['interval'],seeds=raw.mean(1).tolist())
    return out

def routed():
    prev=read(RFPO/'archives/index.json');hist=read(RFPO/'results/outcomes.json')['levels']
    assert sha(RFPO/'records/text.pt')==prev['text'] and sha(FORM/'results/ucm_text.pt')==prev['gallery']
    result={};per_image=[]
    for sub,rows in [('primary',read(RFPO/'inputs/panel.json')),('official',read(FORM/'inputs/ucm.json'))]:
        label=[c.get('pair',c['image']) for c in rows];ids=[c['image'] for c in rows]
        before={};after={};change={};tallies={};stimulus=[];replays={}
        for seed in 'ab':
            directory=ROOT/'records/route'/seed;proof=read(directory/(sub+'_complete.json'))
            path=directory/(sub+'_features.pt')
            assert sha(path)==proof['features'] and sha(directory/(sub+'_trajectories.tar.gz'))==proof['archive']
            new=load(path);old_p=RFPO/'vectors'/(seed+'_deployed_'+sub+'.pt')
            assert sha(old_p)==prev['vectors'][old_p.name]
            old=load(old_p);assert new['adapter']==old['adapter'] and ids==old['images']==[r['image'] for r in new['meta']]
            assert new['route']==sha(ROOT/'inputs/route.json')
            replays[seed]=float((new['baseline'][ids[0]]-old['features'][0]).abs().max())
            assert replays[seed]<1e-4
            before[seed]=calculate(sub,old['features'],rows);after[seed]=calculate(sub,new['visual'],rows)
            for k in ('mR','CDE'):
                if k in before[seed]:assert abs(before[seed][k].mean()-hist[seed+'_deployed'][sub][k])<1e-12
            change[seed]={k:after[seed][k]-a for k,a in before[seed].items()}
            route=[r['attribute'] for r in new['meta']]
            if seed=='a':stimulus=route
            else:assert route==stimulus
            tallies[seed]=dict(asked=sum(bool(s) for s in route),aligns_with_evaluation=sum(bool(v) and v==c.get('attribute') for v,c in zip(route,rows)))
            for i,c in enumerate(rows):
                per_image.append(dict(seed=seed,sub=sub,image=c['image'],route=route[i],pair=label[i],
                    before={k:float(v[i]) for k,v in before[seed].items()},after={k:float(v[i]) for k,v in after[seed].items()}))
        result[sub]=dict(baseline=summary_by_seed(before,label),route=summary_by_seed(after,label),change=summary_by_seed(change,label),
            selection=tallies,replay=replays)
        if sub=='primary':
            for tag,mask in [('match',[s==c['attribute'] for s,c in zip(stimulus,rows)]),('other',[s!=c['attribute'] for s,c in zip(stimulus,rows)])]:
                if any(mask):
                    arr={s:{'CDE':change[s]['CDE'][mask]} for s in 'ab'}
                    result[sub][tag]=dict(images=sum(mask),difference=summary_by_seed(arr,[k for k,b in zip(label,mask) if b]))
    dump(ROOT/'results/route.json',result);dump(ROOT/'records/route_images.json',per_image)
if __name__=='__main__':routed()

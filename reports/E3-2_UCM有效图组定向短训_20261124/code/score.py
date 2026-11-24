"""Paired new-image single-decode evaluation and reconstruction of every selected reward."""
import os,sys
from pathlib import Path
from collections import Counter
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];REP=ROOT.parent;OLD=REP/'E3-1_UCM第二轮R3困难负样本复验_20261124'
os.environ['E3_PANEL_ROOT']=str(ROOT);sys.path.insert(0,str(OLD/'code'))
from analyze import boot,calibrated,read,dump,sha,write_csv
from pilot_score import ranked_hits
ARMS=('regular_R3','selected_R3','focused_R3','focused_R2')
BASE=REP/'E3-2_UCM双视图方位奖励_20261120/inputs'
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def mixed(sc):return any(v['A']>0 and v['B']>0 for v in sc) and any(v['A']<0 and v['B']<0 for v in sc)
def main():
    torch.set_num_threads(3);pil=ROOT/'pilot';vdir=pil/'vectors';idx=read(vdir/'index.json');assert len(idx)==20
    for f,h in idx.items():assert sha(vdir/f)==h
    oldweights=read(OLD/'records/provenance.json')['adapters'];ks=read(OLD/'inputs/continuous_lock.json')['kappa']
    ro=read(ROOT/'inputs/rosters.json');lock=read(ROOT/'inputs/lock.json');cf=read(ROOT/'inputs/panel.json')
    assert sha(ROOT/'inputs/panel.json')==lock['panel'] and sha(ROOT/'实验口径.md')==lock['protocol']
    ps={r['image']:r for sub in ('','confirm') for r in read(OLD/sub/'inputs/panel.json')};known=set(ps)
    cb=load(ROOT/'records/fresh_text.pt');assert cb['panel']==lock['panel']
    assert not known.intersection(r['image'] for r in cf) and len({r['image'] for r in cf})==len(cf)
    for f in ('validation.json','retrieval_text.pt','e1_extension.json'):
        assert sha(BASE/f)==read(BASE/'lock.json')[f]
    for f,h in read(BASE/'e1_extension.json')['files'].items():assert sha(BASE/'e1'/f)==h
    rows=read(BASE/'validation.json');g=load(BASE/'retrieval_text.pt');e1=load(BASE/'e1/features.pt');ec=read(BASE/'e1/cases.json')
    vm=[r['image'] for r in rows];fm=[r['image'] for r in cf];assert len(rows)==210
    valids={r:i for i,r in enumerate(vm)};wordids={r:i for i,r in enumerate(e1['texts'])}
    n=len(rows);mask=np.arange(n)[:,None]==np.arange(n).repeat(5)[None,:];levels={};values={};cluster={};records=[];checks={}
    for seed in ('a','b'):
        eligible=set(ro[seed]['eligible']);matched=ro[seed]['regular'];selected=ro[seed]['selected']
        assert len(set(selected))==len(set(matched)) and set(selected)<=eligible
        assert sorted(Counter(selected).values())==sorted(Counter(matched).values())
        for a,b in zip(selected,matched):
            assert all(ps[a][l]==ps[b][l] for l in ('attribute','side','scene_id'))
        for arm in ('start',*ARMS):
            tag=seed+'_'+arm
            if arm!='start':
                path=pil/tag;cp=read(path/'complete.json');sig=cp['final'];steps=read(path/'steps.json')
                assert [r['image'] for r in steps]==(matched if arm=='regular_R3' else selected)
                assert cp['initial']==oldweights[seed] and cp['roster']==sha(ROOT/'inputs/rosters.json') and cp['contexts']==sha(ROOT/'records'/(seed+'_contexts.pt'))
                assert cp['groups']==len(steps)==32 and cp['updated']==sum(r['norm']>0 for r in steps)
                check=Counter()
                for j,r in enumerate(steps):
                    bs=r['batches'];assert len(bs)==3
                    for b,z in enumerate(bs):
                        assert z['rng']==(7811000 if seed=='a' else 8811000)+3*j+b
                        assert len(z['ids'])==len(z['scored'])==len(z['third'])==8 and z['mixed']==mixed(z['scored'])
                    anymix=any(b['mixed'] for b in bs);pick=next((i for i,z in enumerate(bs) if z['mixed']),0) if arm.startswith('focused') else 0
                    assert pick==r['picked'];sc=bs[pick]['scored'];q=np.array([x['native'] for x in sc]);a=np.array([x['A'] for x in sc]);proxy=np.array([1/(1+x['rank']) for x in sc])
                    rew=[calibrated(y,proxy)+r['format'] for y in (q,q+ks*a)]
                    adv=[(w-w.mean())/(w.std(ddof=1)+1e-8) for w in rew];ix=int(arm.endswith('R3'))
                    assert np.allclose(r['reward'],rew[ix],atol=1e-12,rtol=0) and np.allclose(r['advantages'],adv[ix],atol=1e-8)
                    changed=bool(np.max(abs(adv[1]-adv[0]))>1e-8);assert changed==r['reward_changes']
                    check.update(first_mixed=int(bs[0]['mixed']),picked_mixed=int(bs[pick]['mixed']),available_mixed=int(anymix),changed_batch=int(pick>0),reward_changes=int(changed),actual_updates=int(r['norm']>0))
                checks[tag]=dict(check,eligible_images=len(eligible),unique_images=len(set(x['image'] for x in steps)),overlap=len(set(matched)&set(selected)),visits_to_eligible=sum(r['image'] in eligible for r in steps),max_mean_logp_err=max(r['mean_abs_logp_err'] for r in steps),adapter=sig)
            else:sig=oldweights[seed]
            v=load(vdir/(tag+'_val.pt'));fresh=load(vdir/(tag+'_fresh.pt'))
            assert v['images']==vm and fresh['images']==fm and sig==v['adapter']==fresh['adapter']
            x=v['features'].double();scores=(x@g['native'].double().T)[:,g['caption_order']].numpy();i=ranked_hits(scores,mask);t=ranked_hits(scores.T,mask.T).reshape(n,5,3).mean(1)
            hits=np.column_stack((i,t));metrics={'val_mR':hits.mean(1),'I2T':i.mean(1),'T2I':t.mean(1)}
            ids={k:vm for k in metrics};cluster.update(ids)
            fine=(x@e1['text_features'].double().T).numpy()
            for k in ('attribute','relation'):
                items=[r for r in ec if r['cohort']=='extension' and ('relation' if r['template']=='shadow_direction' else r['dimension'])==k]
                assert len(items)==36
                metrics[k]=np.array([float(fine[valids[r['image']],wordids[r['positive']]]>fine[valids[r['image']],wordids[r['negative']]]) for r in items]);ids[k]=[r['image'] for r in items];cluster[k]=[r['pair'] for r in items]
            scores=(fresh['features'].double()@cb['hard'].double().T).numpy();pairs=[r['pair'] for r in cf]
            for l in 'CDE':
                ab=[cb['maps'][nm][l] for nm in fm];metrics[l]=np.array([float(z[a]>z[b]) for z,(a,b) in zip(scores,ab)])
                ids[l]=fm;cluster[l]=pairs
            metrics['CDE']=np.stack([metrics[l] for l in 'CDE']).mean(0);ids['CDE']=fm;cluster['CDE']=pairs
            pr=list(dict.fromkeys(pairs));assert len(pr)*2==len(cf)
            metrics['both_all']=np.array([float(np.all(metrics['CDE'][np.array(pairs)==p]==1)) for p in pr]);ids['both_all']=cluster['both_all']=pr
            for k,arr in metrics.items():
                records.extend(dict(tag=tag,metric=k,image=nm,cluster=cl,score=float(z)) for nm,cl,z in zip(ids[k],cluster[k],arr,strict=True))
            values[tag]=metrics;levels[tag]=dict(**{k:float(v.mean()) for k,v in metrics.items()},six_recalls=hits.mean(0).tolist(),both_all_count=int(metrics['both_all'].sum()))
        first=[read(pil/(seed+'_'+a)/'steps.json')[0] for a in ('selected_R3','focused_R3','focused_R2')]
        assert all(r['batches']==first[0]['batches'] for r in first)
        assert first[1]['picked']==first[2]['picked']
    comparisons={}
    for a,b in [('focused_R3','regular_R3'),('focused_R3','start'),('selected_R3','regular_R3'),('focused_R3','selected_R3'),('focused_R3','focused_R2'),('selected_R3','start'),('regular_R3','start'),('focused_R2','start')]:
        label=a+'-'+b;comparisons[label]={}
        for k in values['a_start']:
            diff=100*np.array([values[s+'_'+a][k]-values[s+'_'+b][k] for s in ('a','b')])
            comparisons[label][k]=dict(**boot(diff.mean(0),cluster[k]),seeds=diff.mean(1).tolist())
    root=ROOT/'results';write_csv(root/'per_image.csv',records)
    dump(root/'deployment.json',dict(levels=levels,differences_pp=comparisons,training=checks))
if __name__=='__main__':main()

"""Evaluate the predeclared scene-text fusion and matched reward effects on both UCM galleries."""
from collections import Counter,defaultdict
from pathlib import Path
import re
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'audit'))
from image_inputs import IMAGE,QUERY,read,load,sha
from image_metrics import score_inputs,native_metrics,metric

def unit(x):
    assert torch.isfinite(x).all()
    norm=torch.linalg.vector_norm(x,dim=-1,keepdim=True)
    assert (norm>0).all()
    return x/norm

def scene_vector(feedback,bank):
    """Return the normalized mixture of the three public class scores, in FP64."""
    parts=feedback.split(', ');scores=[];vectors=[]
    assert len(parts)==3
    for part in parts:
        tag,score=part.rsplit(' (',1)
        assert re.fullmatch(r'0\.\d{2}\)|1\.00\)',score)
        scores.append(float(score[:-1]));vectors.append(bank['features'][bank['classes'].index(tag)])
    assert scores==sorted(scores,reverse=True) and sum(scores)>0
    w=torch.tensor(scores,dtype=torch.float64)
    return unit((unit(torch.stack(vectors).double())*(w/w.sum())[:,None]).sum(0))

def fuse(visual,scene):
    return unit(unit(visual.double())+unit(scene.double()))

def within_metric(v,rows,bank):
    classes=np.array([(int(Path(r['image']).stem)-1)//100 for r in rows])
    assert all(classes[i]==r['scene_id'] for i,r in enumerate(rows) if 'scene_id' in r)
    ids=np.array([r['panel_positive_ids'] for r in rows]);score=v.double().numpy()@bank['captions'].double().numpy().T
    within=np.full((len(rows),2),np.nan)
    for sid in sorted(set(classes)):
        ix=np.flatnonzero(classes==sid)
        if len(ix)<2:continue
        sim=score[np.ix_(ix,ids[ix].ravel())]
        mask=(sim==sim.max(1,keepdims=True));cols=(sim==sim.max(0,keepdims=True))
        gold=np.repeat(np.eye(len(ix),dtype=bool),5,axis=1)
        within[ix,0]=np.sum(mask&gold,axis=1)/mask.sum(1)
        within[ix,1]=((cols&gold).sum(0)/cols.sum(0)).reshape(-1,5).mean(1)
    return {'R1_within_I2T':within[:,0],'R1_within_T2I':within[:,1],'R1_within':within.mean(axis=1)}

def main():
    checked=read(ROOT/'原生文本编码校验.json');classes=load(ROOT/'场景类别向量.pt')
    assert sha(ROOT/'场景类别向量.pt')==checked['class_features'] and sha(ROOT/'事前口径.md')==checked['protocol']
    assert max(checked['max_absolute_errors'])<1e-4 and len(set(classes['classes']))==21
    assert classes['text']==[f'A remote sensing image of {x}.' for x in classes['classes']]
    first=ROOT.parent/'first_round/evaluation';manifest=read(first/'inputs/manifest.json')
    assert manifest['query']==QUERY
    rows=read(IMAGE/'inputs/roster.json'); full=read(IMAGE/'feedback_verified.json')['checks']; donor=read(ROOT.parent/'scene_evidence/inputs/assignments.json')
    sources=read(ROOT.parent/'scene_evidence/inputs/manifest.json')
    assert sha(ROOT.parent/'scene_evidence/inputs/assignments.json')==sources['assignment']
    assert sha(IMAGE/'feedback_verified.json')==sources['verified_tool']
    official,fine,subix,banks,galleries=score_inputs()
    units={'official':[r['image'] for r in official],'first106':[r['pair'] for r in fine],'extra164':[r['pair'] for r in galleries['extra']]}
    grouped=defaultdict(dict); source_files={};raw={}
    for panel,gallery in galleries.items():
        rr=[r for r in rows if r['panel']==panel];assert [r['rgb'] for r in rr]==[r['rgb_sha256'] for r in gallery]
        real=torch.stack([scene_vector(full[r['rgb']]['feedback'],classes) for r in rr])
        wrong=torch.stack([scene_vector(full[donor[r['rgb']]]['feedback'],classes) for r in rr])
        levels=defaultdict(dict)
        for arm in ('strict','r2'):
            for seed in 'ab':
                tag=seed+'_'+arm;d=first/'records'/tag/panel;cp=read(d/'complete.json');z=load(d/'features.pt')
                assert z['adapter']==cp['adapter']==manifest['policies'][tag]['adapter']
                assert cp['features']==sha(d/'features.pt') and z['images']==[r['image'] for r in rr]
                assert (z['features'].norm(dim=1)-1).abs().max()<1e-5
                source_files[tag+'_'+panel]=cp['features']
                for cond,vec in [('none',z['features']),('real',fuse(z['features'],real)),('shuffled',fuse(z['features'],wrong)),('scene',real)]:
                    v=native_metrics(vec,gallery,banks[panel]);v.update(within_metric(vec,gallery,banks[panel]))
                    for name,ix in ([('official',np.arange(210)),('first106',subix)] if panel=='official' else [('extra164',np.arange(164))]):
                        keys=(['DE','mrr'] if name=='first106' else ['mR','I2T','T2I','R1_within','R1_within_I2T','R1_within_T2I',
                            'I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10'])
                        if name=='extra164':keys=keys+['DE','mrr']
                        for k in keys:
                            vals=v[k][ix];grouped[name,arm+'_'+cond,k][seed]=vals
                            raw[f'{name}_{tag}_{cond}_{k}']=vals
    prior=read(first/'results/单次检索.json')
    levels={};contrasts={}
    for panel,unit_ids in units.items():
        group_names=sorted({a for p,a,_ in grouped if p==panel})
        keys=sorted({k for p,_,k in grouped if p==panel})
        comparisons=[(a,None) for a in group_names]
        comparisons+=[(a+'_real',a+'_'+b) for a in ['strict','r2'] for b in ('none','shuffled')]
        comparisons+=[('strict_'+c,'r2_'+c) for c in ('none','real','shuffled')]
        for a,b in comparisons:
            name=a+'-'+b if b else a;obj=contrasts if b else levels;stats={}
            for k in keys:
                vals=np.stack([grouped[panel,a,k][s]-(grouped[panel,b,k][s] if b else 0) for s in 'ab'])
                got=metric(vals,unit_ids,panel=='extra164');stats[k]=dict(**got,images=int(np.isfinite(vals[0]).sum()))
                if a.endswith('_none') and b is None and k in prior[panel][a.split('_')[0]]:
                    old=prior[panel][a.split('_')[0]][k]
                    for m in ('mean','interval','seeds'):np.testing.assert_allclose(got[m],old[m],rtol=0,atol=1e-13)
            obj.setdefault(name,{})[panel]=stats
    chosen=[contrasts['strict_real-strict_'+x] for x in ('none','shuffled')]
    system=[d['official']['mR']['interval'][0]>0 and d['extra164']['mR']['mean']>0 for d in chosen]
    reward=contrasts['strict_real-r2_real']
    policy=dict(positive_106_DE_lower=reward['first106']['DE']['interval'][0]>0,
        positive_or_zero_106_RR=reward['first106']['mrr']['mean']>=0,positive_or_zero_210_mR=reward['official']['mR']['mean']>=0,
        **{f'positive_or_zero_164_{k}':reward['extra164'][k]['mean']>=0 for k in ('DE','mrr','mR')})
    obj=dict(protocol=checked['protocol'],class_bank=checked['class_features'],source_vectors=source_files,
        permutation=sources['assignment'],levels=levels,differences=contrasts,system_gain=all(system),
        system_checks=dict(real_over_unfused=system[0],real_over_permuted=system[1]),reward_conditions=policy,reward_pass=all(policy.values()))
    np.savez_compressed(ROOT/'固定样本指标.npz',**raw)
    import json
    (ROOT/'结果.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':
    torch.set_num_threads(2);main()

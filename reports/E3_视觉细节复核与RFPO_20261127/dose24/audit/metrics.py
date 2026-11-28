"""Recompute both fixed-gallery tests and their predeclared promotion rule."""
import sys
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import torch
from inputs import DOSE, STAGE, REPORTS, PANEL, FORMAL, FINE, read, load, digest, output

spec = spec_from_file_location('paired_retrieval', STAGE/'extra164/audit/check.py')
retrieval = module_from_spec(spec)
spec.loader.exec_module(retrieval)

def compare(got, reported, ci='ci'):
    for a,b in [('mean','mean'),('ci',ci),('seeds','seeds')]:
        np.testing.assert_allclose(got[a],reported[b],rtol=0,atol=1e-13)

def summarize(delta,units,seed,method):
    labels=sorted(set(units)) if method=='multinomial' else list(dict.fromkeys(units))
    grouped=np.stack([delta[:,[i for i,v in enumerate(units) if v==label]].mean(axis=1) for label in labels])
    if method=='multinomial':
        weights=np.random.default_rng(seed).multinomial(len(labels),np.full(len(labels),1/len(labels)),size=20000)
        sampled=(weights@grouped.mean(axis=1))/len(labels)
    else:
        picks=np.random.default_rng(seed).integers(len(labels),size=(20000,len(labels)))
        sampled=grouped.mean(axis=1)[picks].mean(axis=1)
    return dict(mean=float(delta.mean()),seeds=delta.mean(axis=1).tolist(),ci=np.quantile(sampled,[.025,.975]).tolist())

def check98():
    rows=read(PANEL/'inputs/rows.json');texts=load(PANEL/'inputs/texts.pt');parts={};adapters={}
    listed=np.load(PANEL/'per_image.npz');expected=read(PANEL/'summary.json'); signed=read(PANEL/'inputs/manifest.json')
    assert signed['rows']==digest(PANEL/'inputs/rows.json')
    for arm in ('strict','r2'):
        parts[arm]=[]
        for i,s in enumerate('ab'):
            tag=s+'_'+arm;file=PANEL/'records'/(tag+'.pt')
            z=load(file); m=read(file.with_suffix('.json'));adapters[tag]=z['adapter']
            assert m['sha256']==digest(file) and z['adapter']==m['adapter']
            assert m['inputs']==signed['rows'] and m['policy']==digest(PANEL/'code/predict.py') and m['protocol']==signed['protocol']
            assert len(z['traces'])==m['images']==len(rows)==98
            assert [(r['image'],r['rgb']) for r in z['traces']]==[(r['image'],r['rgb_sha256']) for r in rows]
            assert z['visual'].shape==(98,768) and z['prior'].shape==(98,3584)
            raw=retrieval.reference(rows,texts,z['visual']);vs=z['visual'].double().numpy();dt=texts['details'].double().numpy()
            fg=np.array([np.mean([vs[j]@dt[pos] > vs[j]@dt[neg] for pos,neg in zip(r['panel_fg_ids'][::2],r['panel_fg_ids'][1::2])]) for j,r in enumerate(rows)])
            raw=np.column_stack((raw,fg)); np.testing.assert_allclose(listed[arm][i],raw,rtol=0,atol=1e-13)
            parts[arm].append(raw)
        parts[arm]=np.stack(parts[arm])
    output_metrics={}
    for arm,values in [*parts.items(),('difference',parts['strict']-parts['r2'])]:
        stats={}
        for col,k in enumerate(('DE','MRR','mR','I2T','T2I','FG')):
            stats[k]=summarize(values[:,:,col],[r['pair'] for r in rows],271127,'multinomial')
            compare(stats[k],(expected['difference'] if arm=='difference' else expected['levels'][arm])[k])
        output_metrics[arm]=stats
    x=output_metrics['difference']
    rules=dict(DE=x['DE']['ci'][0]>0,MRR=x['MRR']['mean']>=0,mR=x['mR']['mean']>=0)
    assert all(rules.values())==expected['conditions_98']
    output('98独立单次与配对区间.json',dict(metrics=output_metrics,conditions=rules,adapters=adapters))
    return adapters,rules

def check210(adapters,rules):
    ev=DOSE/'evaluation';gallery=read(FORMAL/'inputs/ucm.json');panel=read(FINE/'inputs/panel.json')
    native=load(FORMAL/'results/ucm_text.pt'); hard=load(FINE/'records/text.pt')
    lookup={r['rgb_sha256']:r for r in panel};ids={r['rgb_sha256']:i for i,r in enumerate(gallery)}
    subset=[r for r in panel if r['rgb_sha256'] in ids]
    assert len(subset)==106 and len(gallery)==210
    subix=[ids[r['rgb_sha256']] for r in subset]
    rows=[]
    for i,r in enumerate(gallery):
        ent=lookup.get(r['rgb_sha256'])
        detail=[idx for k in 'DE' for idx in hard['maps']['primary'][r['image']][k]] if ent else [0,0,0,0]
        rows.append(dict(r,panel_positive_ids=native['caption_order'].view(210,5)[i].tolist(),panel_de_ids=detail))
    theirs=read(ev/'results/单次检索对照.json'); parts={}
    for arm in ('r2','strict'):
        raw=[]
        for s in 'ab':
            tag=s+'_'+arm;d=ev/'records'/tag;cp=read(d/'complete.json');z=load(d/'features.pt')
            assert adapters[tag]==cp['adapter']==z['adapter'] and digest(d/'features.pt')==cp['features']
            assert z['images']==[r['image'] for r in rows]
            for r,v in zip(rows,z['features'],strict=True):
                rec=d/(r['image']+'.pt'); assert digest(rec)==cp['images'][r['image']]
                trace=load(rec);assert trace['query']==r['orig_query'] and trace['rgb']==r['rgb_sha256']
                assert trace['adapter']==adapters[tag] and torch.equal(trace['v'],v)
            raw.append(retrieval.reference(rows,dict(details=hard['hard'],captions=native['native']),z['features']))
        parts[arm]=np.stack(raw)
    result={}
    for arm,values in [*parts.items(),('strict-r2',parts['strict']-parts['r2'])]:
        result[arm]={}
        for metric,slice_name,data,groups in [('DE','first106',values[:,subix,0],[r['pair'] for r in subset]),('mrr','first106',values[:,subix,1],[r['pair'] for r in subset]),('mR','official',values[:,:,2],[r['image'] for r in gallery])]:
            got=summarize(data,groups,20261125,'image')
            compare(got,theirs[slice_name]['contrasts' if arm=='strict-r2' else 'levels'][arm][metric],ci='interval')
            result[arm][metric]=got
    d=result['strict-r2']
    assessment=dict(panel98=rules,formal106_DE=d['DE']['mean']>=0,formal210_mR=d['mR']['mean']>=0)
    passed=all(rules.values()) and assessment['formal106_DE'] and assessment['formal210_mR']
    output('210独立单次与晋级验收.json',dict(metrics=result,conditions=assessment,promoted=passed))

if __name__=='__main__':
    torch.set_num_threads(1)
    adapters,rules=check98()
    if '--formal' in sys.argv:
        check210(adapters,rules)

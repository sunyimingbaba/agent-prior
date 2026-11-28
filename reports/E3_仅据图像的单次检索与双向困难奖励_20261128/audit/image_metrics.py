"""Independent per-image six-way recall, held-text accuracy and per-pair statistical tests."""
from collections import defaultdict
from importlib.util import spec_from_file_location, module_from_spec
import gzip
import json
import sys
import numpy as np
import torch
from image_inputs import ROOT, IMAGE, OLD, FORM, FINE, read, load, sha, dump

def module(n,p):
    spec=spec_from_file_location(n,p);m=module_from_spec(spec);spec.loader.exec_module(m);return m
rank=module('paired_bank_audit',OLD/'extra164/audit/check.py')
cross=module('tied_recall',OLD/'cross448/audit/check.py')
def native_metrics(v,rows,b):
    scores=v.double().numpy() @ b['captions'].double().numpy().T
    ids=np.array([r['panel_positive_ids'] for r in rows]);ss=scores[:,ids.ravel()]
    six=[]
    for i in range(len(rows)):
        pos=np.arange(5*i,5*i+5)
        a=cross.hits(ss[i],pos)
        rev=np.mean([cross.hits(ss[:,j],[i]) for j in pos],axis=0)
        six.append(np.concatenate((a,rev)))
    six=np.stack(six)
    old=rank.reference(rows,b,v)
    np.testing.assert_allclose(six.mean(1),old[:,2],atol=1e-13,rtol=0)
    out={f'{direction}_R{k}':six[:,offset+col] for offset,direction in [(0,'I2T'),(3,'T2I')] for col,k in enumerate((1,5,10))}
    return dict(**out,**{k:old[:,j] for j,k in enumerate(('DE','mrr','mR','I2T','T2I'))})

def metric(value,labels,extra):
    units=sorted(set(labels)) if extra else list(dict.fromkeys(labels))
    means=value.mean(axis=0)
    ix=[[i for i,x in enumerate(labels) if x==g] for g in units]
    totals=np.array([np.nansum(means[idx]) for idx in ix]);n=np.array([np.isfinite(means[idx]).sum() for idx in ix])
    if extra:
        w=np.random.default_rng(271127).multinomial(len(units),np.full(len(units),1/len(units)),size=20000)
        boot=(w@totals)/(w@n)
    else:
        w=np.random.default_rng(20261125).integers(len(units),size=(20000,len(units)))
        boot=totals[w].sum(1)/n[w].sum(1)
    return dict(mean=float(np.nanmean(means)),seeds=np.nanmean(value,axis=1).tolist(),interval=np.quantile(boot,[.025,.975]).tolist())

def score_inputs():
    official=read(FORM/'inputs/ucm.json');whole=read(FINE/'inputs/panel.json')
    idx={r['rgb_sha256']:i for i,r in enumerate(official)};fine=[r for r in whole if r['rgb_sha256'] in idx]
    subix=[idx[r['rgb_sha256']] for r in fine]
    native=load(FORM/'results/ucm_text.pt');details=load(FINE/'records/text.pt')
    scorerbank={'official':{'captions':native['native'],'details':details['hard']},'extra':load(OLD/'extra164/inputs/texts.pt')}
    hard=details['maps']['primary'];rows={'official':[],'extra':read(OLD/'extra164/inputs/rows.json')}
    for i,r in enumerate(official):
        ix=[j for k in 'DE' for j in hard[r['image']][k]] if i in subix else [0,0,0,0]
        rows['official'].append(dict(r,panel_de_ids=ix,panel_positive_ids=native['caption_order'].reshape(-1,5)[i].tolist()))
    return official, fine, subix, scorerbank, rows

def main(variant):
    root=IMAGE if variant=='image_only' else ROOT/variant/'evaluation'
    official,fine,subix,scorerbank,rows=score_inputs()
    official_records=read(root/'results/单次检索.json');per={}
    with gzip.open(root/'results/逐图指标.jsonl.gz','rt') as src:
        for line in src:
            obs=json.loads(line); per[obs['seed']+'_'+obs['arm'],obs['panel'],obs['image']]=obs
    assert len(per)==4*(210+106+164)
    series=defaultdict(dict);expected=read(ROOT/'audit/通用问题与原始输入独立核验.json');fixed=expected['expected_tokens_by_image_size']['256x256']['digest']
    localtool=read(root/'feedback_verified.json');sr=read(root/'inputs/server_lock.json')
    assert sr['feedback']==sha(root/'feedback_verified.json') and sr['reader']==expected['reader'] and sr['manifest']==sha(root/'inputs/manifest.json')
    assert localtool['clip']==expected['clip'] and localtool['head']==expected['scene_head']
    plots=read(root/'inputs/roster.json');fb={r['rgb']:r['feedback'] for r in plots}
    rj=read(root/'inputs/manifest.json')
    for seed in 'ab':
        for a in ('r2','strict'):
            tag=seed+'_'+a
            for pp,rlist in rows.items():
                actual_tag=rj['policies'][tag].get('actual_output',tag)
                d=root/'records'/actual_tag/pp;cp=read(d/'complete.json');data=load(d/'features.pt')
                assert sha(d/'features.pt')==cp['features'] and cp['adapter']==data['adapter']==rj['policies'][tag]['adapter']
                assert data['images']==[r['image'] for r in rlist] and data['features'].shape==(len(rlist),768) and cp['lock']==sha(root/'inputs/server_lock.json')
                for r,v in zip(rlist,data['features'],strict=True):
                    n=r['image'];rgb=r['rgb_sha256'];f=d/(n+'.pt');z=load(f)
                    assert sha(f)==cp['images'][n] and z['image']==n and z['rgb']==rgb and z['lock']==cp['lock']
                    assert z['query']==z['actual_query']==expected['query'] and z['first_input_ids_sha256']==fixed and z['template']==expected['expected_chat_template_sha256']
                    assert z['adapter']==cp['adapter'] and torch.equal(z['v'],v)
                    assert z['feedback']==fb[rgb]==localtool['checks'][rgb]['feedback'] and all(c==z['feedback'] for c in z['calls'])
                out=native_metrics(data['features'],rlist,scorerbank[pp])
                for panel,ids in ([('official',np.arange(210)),('first106',subix)] if pp=='official' else [('extra164',np.arange(164))]):
                    keys=official_records[panel]['r2'].keys()
                    for k in keys:
                        if k not in out:continue
                        vals=out[k][ids]
                        ref=[per[tag,panel,rlist[i]['image']]['metrics'][k] for i in ids]
                        gold=np.array([x if x is not None else np.nan for x in ref])
                        np.testing.assert_allclose(vals,gold,rtol=0,atol=1e-13,equal_nan=True)
                        series[panel,a,k][seed]=vals
    labels={'official':[r['image'] for r in official],'first106':[r['pair'] for r in fine],'extra164':[r['pair'] for r in rows['extra']]}
    results=defaultdict(lambda:defaultdict(dict))
    for panel,groups in labels.items():
        for a in ('r2','strict','strict_minus_r2'):
            for k in sorted({key for p,_,key in series if p==panel}):
                vals=np.stack([series[panel,a,k][s] if a!='strict_minus_r2' else series[panel,'strict',k][s]-series[panel,'r2',k][s] for s in 'ab'])
                stats=metric(vals,groups,panel=='extra164'); theirs=official_records[panel][a][k]
                for key in ('mean','interval','seeds'):np.testing.assert_allclose(stats[key],theirs[key],rtol=0,atol=1e-13)
                results[panel][a][k]=stats
    main=results['first106']['strict_minus_r2'];main_mr=results['official']['strict_minus_r2']['mR'];extra=results['extra164']['strict_minus_r2']
    gate=dict(de_lower_positive=main['DE']['interval'][0]>0,official_mR_nonnegative=main_mr['mean']>=0,mrr_nonnegative=main['mrr']['mean']>=0)
    assert all(gate.values())==official_records['formal_pass']
    guard={k:extra[k]['mean']>=0 for k in ('DE','mR','mrr')}
    if 'formal_criteria' in official_records:
        assert gate==official_records['formal_criteria'] and guard==official_records['extra164_preserved']
    else:
        assert variant=='matched' and official_records['actual_R0_predictions']==748
    assert (all(guard.values()) and all(gate.values()))==official_records['joint_pass']
    dump(variant+'_双图库独立回算.json',dict(results=results,formal=gate,extra=guard,formal_pass=all(gate.values()),joint_pass=all(gate.values()) and all(guard.values())))

if __name__=='__main__':
    torch.set_num_threads(1);main(sys.argv[1])

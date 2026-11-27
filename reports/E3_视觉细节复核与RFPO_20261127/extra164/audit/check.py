"""Inspect locked data identity and independently score tie-aware cross-modal retrieval."""
from collections import defaultdict
from importlib.util import spec_from_file_location,module_from_spec
from math import comb
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];REP=ROOT.parents[1]
FORM=REP/'E3-2_UCM与RSITMD独立复验_20261124'
RFPO=REP/'E3_UCM多描述奖励同预算RFPO_20261124'
POOL=REP/'E3_回填后双收益训练图筛选_20261127'
CAL=REP/'E3_困难奖励跨图跨措辞校准_20261127'
DPO=REP/'E3_双收益回答对的偏好训练_20261127'
def read(p):return json.loads(p.read_bytes())
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,z):p.write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n')

def recall(scores,targets):
    best=max(scores[j] for j in targets)
    better=int(np.sum(scores>best));tied=int(np.sum(scores==best)); positives=sum(scores[j]==best for j in targets)
    hit=[]
    for k in (1,5,10):
        rest=k-better
        if rest<=0:hit.append(0.)
        elif rest>tied-positives:hit.append(1.)
        else:hit.append(1.-comb(tied-positives,rest)/comb(tied,rest))
    return np.mean(hit)

def reference(rows,texts,image):
    v=image.double().numpy()
    sim=v@texts['captions'].double().numpy().T
    det=v@texts['details'].double().numpy().T
    ids=np.array([r['panel_positive_ids'] for r in rows])
    s=sim[:,ids.ravel()];values=[]
    for i,r in enumerate(rows):
        pos=ids[i];ex=set(pos);alts=ex-{pos[0]}
        neg=np.array([v for j,v in enumerate(sim[i]) if j not in ex])
        reciprocals=[]
        for j in alts:
            val=sim[i,j];first=1+int(np.sum(neg>val));ties=int(np.sum(neg==val))
            reciprocals.append(sum(1/k for k in range(first,first+ties+1))/(1+ties))
        de=r['panel_de_ids']
        accuracy=np.mean([det[i,de[0]]>det[i,de[1]],det[i,de[2]]>det[i,de[3]]])
        ix=np.arange(i*5,i*5+5)
        i2t=recall(s[i],ix); t2i=np.mean([recall(s[:,j],[i]) for j in ix])
        values.append([accuracy,np.mean(reciprocals) if reciprocals else np.nan,(i2t+t2i)/2,i2t,t2i])
    return np.array(values)

def verify_inputs():
    rows=read(ROOT/'inputs/rows.json');txt=load(ROOT/'inputs/texts.pt');manifest=read(ROOT/'inputs/manifest.json')
    assert sha(ROOT/'inputs/rows.json')==manifest['rows'] and sha(ROOT/'inputs/texts.pt')==manifest['texts']
    source=read(REP/'E3_普通配文连续扣分与独立检验_20261125/inputs/real_roster.json')
    included={r['rgb_sha256'] for r in rows}
    ext=read(CAL/'followup_pool/inputs/rows.json')
    assert {r['rgb_sha256'] for r in source}-{r['rgb_sha256'] for r in ext}==included
    x={r['rgb_sha256']:r for r in source};pairs=defaultdict(list)
    datasets={'external118':ext,'official210':read(FORM/'inputs/ucm.json'),'fine126':read(RFPO/'inputs/panel.json'),
              'RFPO24':read(POOL/'scoring/locks/training.json')['rows'],'DPO':read(DPO/'inputs/train_pairs.json')['rows']}
    overlap={k:len({r['rgb_sha256'] for r in v}&included) for k,v in datasets.items()}
    assert not any(overlap.values())
    bankdir=next(REP.glob('E3-1_奖励一致性*/空间关系候选*/native'))
    b=load(bankdir/'features.pt')['features'];ids=read(bankdir/'index.json')['columns']
    assert sha(bankdir/'features.pt')==manifest['native']
    details=defaultdict(list)
    for f,h in manifest['detail_sources'].items():
        p=REP.parent/f;assert sha(p)==h
        bank=load(p)
        for a,vec in zip(bank.get('text',bank.get('texts')),bank['hard']):details[a].append(vec)
    natids=set();dtids=set()
    for r in rows:
        assert x[r['rgb_sha256']]=={k:v for k,v in r.items() if k in x[r['rgb_sha256']]}
        pairs[r['pair']].append(r)
        for t,j in zip(r['captions'],r['panel_positive_ids'],strict=True):
            assert torch.equal(txt['captions'][j],b[ids[t]]);natids.add(j)
        for t,j in zip(r['D']+r['E'],r['panel_de_ids'],strict=True):
            assert any(torch.equal(vec,txt['details'][j]) for vec in details[t]);dtids.add(j)
    assert set(natids)==set(range(len(txt['captions']))) and set(dtids)==set(range(len(txt['details'])))
    assert len(pairs)==82 and all(sorted(r['side'] for r in z)==[0,1] for z in pairs.values())
    return {'images':len(included),'pairs':len(pairs),'source':len(source),'captions':len(natids),'fine_texts':len(dtids),
            'full_RR_coverage':sum(len(set(r['panel_positive_ids']))>1 for r in rows),'overlap_pixels':overlap}

def prior_gallery():
    gallery=read(FORM/'inputs/ucm.json');hard=load(RFPO/'records/text.pt');nat=load(FORM/'results/ucm_text.pt')
    ps={r['rgb_sha256']:r for r in read(RFPO/'inputs/panel.json')}
    covered=[i for i,r in enumerate(gallery) if r['rgb_sha256'] in ps]
    cols=nat['caption_order'].reshape(-1,5).tolist();txt={'captions':nat['native'],'details':hard['hard']};rows=[]
    for i,r in enumerate(gallery):
        ent=ps.get(r['rgb_sha256'])
        ids=[0,0,0,0] if ent is None else [j for k in 'DE' for j in hard['maps']['primary'][ent['image']][k]]
        rows.append(dict(r,panel_de_ids=ids,panel_positive_ids=cols[i]))
    spec=spec_from_file_location('target_score',ROOT/'code/score.py')
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    old={(r['seed'],r['arm'],r['image']):r for r in read(POOL/'evaluation/results/逐图指标.json')}
    out={}
    for s,a in [('a','r2'),('b','r3')]:
        v=load(POOL/'evaluation/records'/(s+'_'+a)/'features.pt')['features']
        extra=mod.measures(rows,txt,v);check=reference(rows,txt,v)
        np.testing.assert_allclose(extra,check,rtol=0,atol=1e-13,equal_nan=True)
        data=[old[s,a,r['image']] for r in gallery]
        errors={k:float(np.max(np.abs(extra[ix,m]-[data[j][tag][k] for j in ix])))
                for m,k,tag,ix in [(0,'DE','first106',covered),(1,'mrr','first106',covered),(2,'mR','official',list(range(len(rows))))]}
        assert max(errors.values())<1e-13
        out[s+'_'+a]=errors
    return out

def main():
    save(ROOT/'audit/输入与计分核查.json',{'inputs':verify_inputs(),'historical_metric_errors':prior_gallery()})
    if '--outputs' not in sys.argv:return
    ref=np.load(ROOT/'per_image.npz');rows=read(ROOT/'inputs/rows.json');texts=load(ROOT/'inputs/texts.pt')
    groups=sorted({r['pair'] for r in rows})
    draws=np.random.default_rng(271127).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=20000)
    ix=[np.flatnonzero([r['pair']==g for r in rows]) for g in groups]
    values={};reports={};previous=read(ROOT/'summary.json')
    for arm,arr in ref.items():
        vals=[]
        for j,s in enumerate('ab'):
            p=ROOT/'records'/(s+'_'+arm+'.pt')
            z=load(p);meta=read(p.with_suffix('.json'))
            assert sha(p)==meta['sha256'] and z['adapter']==meta['adapter']
            sources={'rfpo_r2':(POOL,'r2'),'rfpo_strict':(ROOT.parent,'strict'),'dpo_r2':(DPO,'r2'),'dpo_r3':(DPO,'r3')}
            store= {'rfpo_r2':'e3-aftertool-joint-20261127','rfpo_strict':'e3-visual-detail-rfpo-20261127',
                    'dpo_r2':'e3-joint-answer-pairs-20261127','dpo_r3':'e3-joint-answer-pairs-20261127'}
            loc,trained=sources[arm];name=s+'_'+trained;pr=read(loc/'models'/name/'complete.json')
            assert z['adapter']==pr['final']==sha(Path('/Users/botlv/cup-deploy-assets')/store[arm]/'models'/name/'adapter/adapter_model.safetensors')
            assert meta['inputs']==sha(ROOT/'inputs/rows.json') and meta['policy']==sha(ROOT/'code/predict.py')
            assert meta['images']==len(rows)==len(z['traces'])
            assert [(r['image'],r['rgb']) for r in z['traces']]==[(r['image'],r['rgb_sha256']) for r in rows]
            assert torch.isfinite(z['visual']).all() and torch.isfinite(z['prior']).all()
            d=reference(rows,texts,z['visual']);np.testing.assert_allclose(d,arr[j],rtol=0,atol=1e-13,equal_nan=True);vals.append(d)
        values[arm]=np.stack(vals)
    for a,b in [('dpo_r3','dpo_r2'),('rfpo_strict','rfpo_r2')]:
        difference=(values[a]-values[b]).mean(0)
        stats={}
        for j,k in enumerate(['DE','MRR','mR','I2T','T2I']):
            diff=difference[:,j];n=np.array([sum(np.isfinite(diff[ids])) for ids in ix])
            summ=np.array([np.nansum(diff[ids]) for ids in ix]);q=np.quantile((draws@summ)/(draws@n),[.025,.975])
            old=previous['contrasts'][a][k]
            np.testing.assert_allclose([old['mean'],*old['ci']],[np.nanmean(diff),*q],rtol=0,atol=1e-13)
            stats[k]={'mean':float(np.nanmean(diff)), 'ci':q.tolist(),'images':int(n.sum())}
        reports[a+'-'+b]=stats
    save(ROOT/'audit/独立差值与区间核查.json',reports)

if __name__=='__main__':main()

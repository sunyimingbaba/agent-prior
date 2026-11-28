"""Verify the shuffled classification hints, regenerated inputs, and fixed-gallery contrasts."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import sys
import numpy as np
import torch

from image_inputs import ROOT,IMAGE,MODEL,QUERY,AutoProcessor,read,load,sha,dump,expected_messages,tables,image
from image_metrics import native_metrics,score_inputs,metric

EVID=ROOT/'scene_evidence'

def verify_mapping():
    roster=read(IMAGE/'inputs/roster.json');lock=read(EVID/'inputs/manifest.json');mapping=read(EVID/'inputs/assignments.json')
    old=read(IMAGE/'inputs/manifest.json');feedback=read(IMAGE/'feedback_verified.json')
    assert lock['policies']==old['policies'] and lock['query']==QUERY
    assert sha(IMAGE/'feedback_verified.json')==lock['verified_tool'] and sha(IMAGE/'inputs/roster.json')==lock['roster']
    assert sha(EVID/'检验口径.md')==lock['protocol'] and sha(EVID/'inputs/assignments.json')==lock['assignment']
    assert lock['source']==sha(IMAGE/'inputs/manifest.json')
    local=read(ROOT/'audit/场景分类器逐像素独立回算.json')
    maxconf=max(abs(conf-feedback['checks'][k]['confidence']) for k,conf in local['confidence'].items())
    assert maxconf<1e-4 and set(mapping)==set(local['confidence'])
    result={}
    for p in ('official','extra'):
        selected=[r for r in roster if r['panel']==p]
        rs={r['rgb']:r for r in selected}
        assert set(mapping[i] for i in rs)==set(rs)
        obs=[feedback['checks'][mapping[r['rgb']]]['feedback'] for r in selected]
        src=[r['feedback'] for r in selected]
        assert Counter(obs)==Counter(src)
        top={r['rgb']:r['feedback'].split(' (')[0] for r in selected}; order=sorted(top,key=lambda x:(top[x],x))
        offset=Counter(top.values()).most_common(1)[0][1]
        assert all(mapping[k]==order[(j+offset)%len(order)] and top[k]!=top[mapping[k]] for j,k in enumerate(order))
        result[p]=dict(images=len(selected),different_top=sum(a.split(' (')[0]!=b.split(' (')[0] for a,b in zip(src,obs)),
                            distinct_feedback=len(set(src)),shift=offset,maximum_text_frequency=max(Counter(src).values()))
    dump('预给场景反馈对照独立核查.json',dict(panels=result,label=lock['label'],source_policies=old['policies'],tool_probability_max_abs_error=maxconf))
    return roster,lock,mapping,feedback['checks']

def main(actual):
    roster,lock,mapping,verified=verify_mapping()
    if not actual:return
    official,fine,subix,scorerbank,gallery=score_inputs();t=tables()
    qwen=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    msgs=expected_messages();reference_image=image(roster[0],t)
    known={};parts=defaultdict(dict)
    armnames=('r3_none','r2_none','r2_real','r3_real','r3_shuffled')
    for s in 'ab':
        for a in armnames:
            tag=s+('_r2' if a.startswith('r2_') else '_strict')
            condition=a.split('_')[1];sig=lock['policies'][tag]['adapter']
            for pp,rows in gallery.items():
                selected=[r for r in roster if r['panel']==pp]
                assert [r['rgb_sha256'] for r in rows]==[r['rgb'] for r in selected]
                if condition=='none':
                    p=IMAGE/'records'/tag/pp/'features.pt';data=load(p);v=data['features'];assert data['adapter']==sig
                else:
                    d=EVID/'records'/(tag+'_'+condition);p=d/(pp+'.pt');data=load(p);meta=read(p.with_suffix('.json'))
                    assert meta['generation']==sha(EVID/'code/generate.py') and meta['common']==sha(IMAGE/'code/predict.py') and meta['model']==data['adapter']==sig
                    assert meta['manifest']==sha(EVID/'inputs/manifest.json') and meta['sha256']==sha(p)
                    assert data['prior'].shape==(len(rows),3584) and torch.isfinite(data['prior']).all();v=data['visual']
                    for i,(r,z) in enumerate(zip(selected,data['traces'],strict=True)):
                        key=r['rgb'] if condition=='real' else mapping[r['rgb']]
                        hint=verified[key]['feedback'];query=QUERY+'\n'+lock['label']+hint
                        assert (z['name'],z['rgb'],z['evidence_rgb'],z['query'])==(r['image'],r['rgb'],key,query)
                        assert all(c==r['feedback'] for c in z['calls']) and z['actions']
                        # Every image is the same true size, hence code words depend on the text and visual grid only.
                        if query not in known:
                            msgs[-1]['content'][1]['text']='查询文本: '+query
                            rendered=qwen.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
                            ids=qwen(text=[rendered],images=[reference_image],return_tensors='pt').input_ids
                            known[query]=(hashlib.sha256(rendered.encode()).hexdigest(),hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest())
                        assert (z['template'],z['tokens'])==known[query]
                assert v.shape==(len(rows),768) and torch.isfinite(v).all()
                allvalues=native_metrics(v,rows,scorerbank[pp])
                for panel,ids in ([('official',np.arange(210)),('first106',subix)] if pp=='official' else [('extra164',np.arange(164))]):
                    for k in ((['DE','mrr'] if panel=='first106' else ['mR','I2T','T2I','I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10'])+(['DE','mrr'] if panel=='extra164' else [])):
                        parts[panel,a,k][s]=allvalues[k][ids]
    units={'official':[r['image'] for r in official],'first106':[r['pair'] for r in fine],'extra164':[r['pair'] for r in gallery['extra']]}
    out={}
    for a,b in (('r3_real','r3_none'),('r3_real','r3_shuffled'),('r3_real','r2_real'),('r3_shuffled','r3_none'),('r2_real','r2_none')):
        result={}
        for p,labels in units.items():
            stats={}
            for k in [k for n,variant,k in parts if n==p and variant==a]:
                d=np.stack([parts[p,a,k][s]-parts[p,b,k][s] for s in 'ab'])
                stats[k]=metric(d,labels,p=='extra164')
            result[p]=stats
        out[a+' minus '+b]=result
    recorded=read(EVID/'结果.json')
    for name,panels in out.items():
        ref=recorded['differences'][name.replace('r3_', 'strict_').replace(' minus ','-')]
        for p,stats in panels.items():
            for key,x in stats.items():
                for k in ('mean','interval','seeds'):np.testing.assert_allclose(x[k],ref[p][key][k],rtol=0,atol=1e-13)
    real_none,real_shuffled,real_r2=list(out.values())[:3]
    hint_gate=all(d['official']['mR']['interval'][0]>0 and d['extra164']['mR']['mean']>0 for d in (real_none,real_shuffled))
    reward_gate=real_r2['first106']['DE']['interval'][0]>0 and real_r2['first106']['mrr']['mean']>=0 and real_r2['official']['mR']['mean']>=0
    extra_gate=all(real_r2['extra164'][k]['mean']>=0 for k in ('DE','mrr','mR'))
    assert hint_gate==recorded['evidence_gain'] and (reward_gate and extra_gate)==recorded['reward_pass']
    dump('预给场景证据单次对照独立核算.json',dict(contrasts=out,evidence_pass=hint_gate,formal_reward_pass=reward_gate,joint_reward_pass=reward_gate and extra_gate))

if __name__=='__main__':
    torch.set_num_threads(2);main('--actual' in sys.argv)

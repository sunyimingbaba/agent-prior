import hashlib
import math
import numpy as np

def verify(plan,analysis,scores,tensors,gallery):
    lookup={x['id']:i for i,x in enumerate(scores)};index={};words={w:i+1 for i,w in enumerate(['one','two','three','four','five','six','seven'])}
    cases={x['image']:x for x in analysis['per_image'] if x['arm']=='before' and x['selector']=='greedy'}
    texts=gallery['quantity_text_order'];byimage={}
    for name in cases:
        bcols=gallery['quantity_columns'][name]['B'];prefix=texts[bcols[0]].rsplit(' ',1)[0]+' '
        byimage[name]={words[s.rsplit(' ',1)[1].rstrip('.')]:i for i,s in enumerate(texts) if s.startswith(prefix)}
    for e in analysis['per_image']:
        if e['selector']=='shared':xs=[x for x in scores if x['arm']==e['arm']+'_shared' and x['image']==e['image']]
        else:xs=[x for x in scores if x['arm']==e['arm'] and x['image']==e['image'] and x['role']==('candidate' if e['selector']=='sample_mean' else 'greedy')]
        assert len(xs)==(4 if e['selector']=='sample_mean' else 1) and e['selected_ids']==[x['id'] for x in xs]
        values={}
        for m in ['quality','strict_quality','tool_usage','generated_result_failure','margin_A','margin_B','pairacc_A','pairacc_B','retrieval_rr','argmax_two']:
            vs=[]
            for x in xs:
                if e['selector']=='shared' and m in ['quality','strict_quality','tool_usage','generated_result_failure']:v=None
                elif m=='quality':v=float(x['quality_usable'])
                elif m=='strict_quality':v=float(x['strict_protocol_usable'])
                elif m=='tool_usage':v=float(x['tools_used']>0)
                elif m=='generated_result_failure':v=float(x['stop_reason']=='model_generated_tool_result')
                elif m.startswith('pairacc_'):v=float(x['correct_'+m[-1]])
                elif m=='argmax_two':
                    q=tensors['quantity_similarities'][lookup[x['id']]];cols=byimage[x['image']];v=float(max(sorted(cols),key=lambda c:float(q[cols[c]]))==2)
                else:v=x[m]
                vs.append(v)
            value=None if vs[0] is None else math.fsum(vs)/len(vs);assert value is None and e[m] is None or value is not None and abs(value-e[m])<1e-12
            values[m]=value
        index[e['arm'],e['image'],e['selector']]=values
    def seed(s):return (plan['analysis']['bootstrap_seed']+int(hashlib.sha256(s.encode()).hexdigest()[:8],16))%2**32
    ci=0
    for cohort,g in analysis['groups'].items():
        entries=[x for x in analysis['per_image'] if x['cohort']==cohort and x['arm']=='before' and x['selector']=='greedy'];names=[x['image'] for x in entries]
        def check(value,vs,label):
            nonlocal ci
            v=np.asarray(vs,dtype=np.float64);rng=np.random.default_rng(seed(label));draw=rng.integers(0,len(v),size=(10000,len(v)))
            assert abs(float(v.mean())-value['mean'])<1e-12 and np.allclose(np.percentile(v[draw].mean(axis=1),[2.5,97.5]),value['interval95'],atol=1e-12,rtol=0);ci+=1
        for arm,selectors in g['raw'].items():
            for sel,metrics in selectors.items():
                for m,st in metrics.items():check(st,[index[arm,n,sel][m] for n in names],cohort+'/'+arm+'/'+sel+'/'+m)
        for sel,metrics in g['paired'].items():
            for m,st in metrics.items():check(st,[index['after',n,sel][m]-index['before',n,sel][m] for n in names],cohort+'/paired/'+sel+'/'+m)
        contrasts={'after_minus_before__sample_mean':('after','sample_mean','before','sample_mean'),'after_minus_before__greedy':('after','greedy','before','greedy'),
          'after_minus_before__shared':('after','shared','before','shared'),'after_own_minus_shared__greedy':('after','greedy','after','shared'),'before_own_minus_shared__greedy':('before','greedy','before','shared')}
        for contrast,(arm,sel,base,bsel) in contrasts.items():
            for m,st in g['macro'][contrast].items():
                rng=np.random.default_rng(seed(cohort+'/macro/'+contrast+'/'+m));point=0.;boot=np.zeros(10000)
                for count in [1,2,3,4]:
                    ns=[x['image'] for x in entries if x['actual_count']==count];v=np.asarray([index[arm,n,sel][m]-index[base,n,bsel][m] for n in ns]);point+=float(v.mean())/4;draw=rng.integers(0,len(v),size=(10000,len(v)));boot+=v[draw].mean(axis=1)/4
                assert abs(point-st['mean'])<1e-12 and np.allclose(np.percentile(boot,[2.5,97.5]),st['interval95'],atol=1e-12,rtol=0);ci+=1
        for arm,selectors in g['macro_absolute'].items():
            for sel,metrics in selectors.items():
                for m,st in metrics.items():
                    v=np.mean([np.mean([index[arm,x['image'],sel][m] for x in entries if x['actual_count']==c]) for c in [1,2,3,4]]);assert abs(float(v)-st)<1e-12
        for count,d in g['count_groups'].items():
            ns=[x['image'] for x in entries if x['actual_count']==int(count)];assert len(ns)==d['images']
            for arm in ['before','after']:
                for sel,metrics in d[arm].items():
                    for m,st in metrics.items():assert abs(float(np.mean([index[arm,n,sel][m] for n in ns]))-st)<1e-12
    return ci,len(index)

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import torch
from diagnostic_io import read,sha,hydrate,paired_records,tensor_sha,dump

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root;torch.set_num_threads(2)
plan=read(r/'plan.json');parent=hydrate(r);j=read(r/'analysis.json');assert j['ok'] and plan['optimizer_updates']==j['optimizer_updates']==0
for snap in ['code_hashes.json','analysis_code_hashes.json']:
    for n,h in read(r/snap).items():assert sha(r/n)==h,n
for n,h in plan['inputs_sha256'].items():assert sha(r/n)==h,n
q=subprocess.run([sys.executable,str(parent/'code/verify_eos.py'),'--root',str(parent),'--out',str(r/'scratch/recursive_parent_verification.json')],capture_output=True,text=True)
assert q.returncode==0,q.stdout+q.stderr
assert read(r/'scratch/recursive_parent_verification.json')['ok']
cases=read(r/'inputs/evaluation_cases.json');by={x['image']:x for x in cases};assert len(by)==85
new,old=paired_records(parent);nt=torch.load(parent/'scoring/tensors.pt',map_location='cpu',weights_only=True);ot=torch.load(parent/'reference_run/scoring/tensors.pt',map_location='cpu',weights_only=True)
ni={x:i for i,x in enumerate(nt['record_ids'])};oi={x:i for i,x in enumerate(ot['record_ids'])};f=torch.load(r/'features/tensors.pt',map_location='cpu',weights_only=True);gallery=read(r/'features/gallery.json');texts=gallery['quantity_text_order']
words={x:i+1 for i,x in enumerate(['one','two','three','four','five','six','seven'])}
def count_columns(item):
    label=re.fullmatch(r'The number of visible (.+) is (\w+)\.',item['pairs']['B']['positive']).group(1)
    result={}
    for col,t in enumerate(texts):
        m=re.fullmatch(r'The number of visible (.+) is (\w+)\.',t)
        if m and m.group(1)==label:result[words[m.group(2)]]=col
    assert 1 in result and 2 in result;return result
def two_values(q,item):
    cs=count_columns(item);return float(q[cs[2]]-q[cs[1]]),float(max(sorted(cs),key=lambda k:float(q[cs[k]]))==2)
rebuilt={}
for x in new:
    o=old[x['image'],x['role'],x['seed']];key=x['image'],('greedy' if x['role']=='greedy' else 'sample_mean');rebuilt.setdefault(key,[])
    bp,ba=two_values(ot['quantity_similarities'][oi[o['id']]],by[x['image']]);npref,na=two_values(nt['quantity_similarities'][ni[x['id']]],by[x['image']])
    nh=torch.load(parent/x['state_file'],map_location='cpu',weights_only=True).float();oh=torch.load(parent/'reference_run'/o['state_file'],map_location='cpu',weights_only=True).float()
    rebuilt[key].append({'before_pairacc_B':float(o['correct_B']),'after_pairacc_B':float(x['correct_B']),'delta_pairacc_B':float(x['correct_B'])-float(o['correct_B']),
     'before_pairacc_A':float(o['correct_A']),'after_pairacc_A':float(x['correct_A']),'before_margin_B':o['margin_B'],'after_margin_B':x['margin_B'],'delta_margin_B':x['margin_B']-o['margin_B'],
     'before_two_minus_one':bp,'after_two_minus_one':npref,'delta_two_minus_one':npref-bp,'before_argmax_two':ba,'after_argmax_two':na,
     'hidden_cosine':float(torch.dot(nh,oh)/(torch.linalg.vector_norm(nh)*torch.linalg.vector_norm(oh)))})
for key,xs in rebuilt.items():rebuilt[key]={k:math.fsum(x[k] for x in xs)/len(xs) for k in xs[0]}
for x in j['per_image']:
    for k,v in rebuilt[x['image'],x['selector']].items():assert abs(x[k]-v)<(2e-6 if k=='hidden_cosine' else 1e-12)
def seed(label):return (plan['bootstrap']['seed']+int(hashlib.sha256(label.encode()).hexdigest()[:8],16))%2**32
ci=0
for entry in j['stratified']:
    xs=[x for x in j['per_image'] if x['selector']==entry['selector'] and x['split']==entry['split'] and (entry['dimension']=='all' or str(x[entry['dimension']])==entry['group'])]
    assert len(xs)==entry['images'];label='/'.join([entry['selector'],entry['split'],entry['dimension'],entry['group']])
    for metric in rebuilt[next(iter(rebuilt))]:
        value=math.fsum(rebuilt[x['image'],x['selector']][metric] for x in xs)/len(xs);assert abs(value-entry[metric])<(2e-6 if metric=='hidden_cosine' else 1e-12)
    for metric,suffix in [('delta_pairacc_B','pairacc'),('delta_margin_B','margin')]:
        d=np.asarray([rebuilt[x['image'],x['selector']][metric] for x in xs]);rng=np.random.default_rng(seed(label+'/'+suffix));ix=rng.integers(0,len(xs),size=(10000,len(xs)))
        assert np.allclose(np.percentile(d[ix].mean(axis=1),[2.5,97.5]),entry[metric+'_interval95'],atol=1e-12,rtol=0);ci+=1
for e in j['count_standardized']:
    rng=np.random.default_rng(seed(e['selector']+'/'+e['split']+'/macro'));boot=np.zeros((10000,2));point=np.zeros(2)
    for c in [1,2,3,4]:
        names=[x['image'] for x in cases if x['split']==e['split'] and x['actual_count']==c];v=np.array([[rebuilt[name,e['selector']][k] for k in ['before_pairacc_B','after_pairacc_B']] for name in names]);point+=v.mean(axis=0)/4
        ix=rng.integers(0,len(v),size=(10000,len(v)));boot+=v[ix].mean(axis=1)/4
    assert abs(point[0]-e['before_pairacc_B'])<1e-12 and abs(point[1]-e['after_pairacc_B'])<1e-12
    assert np.allclose(np.percentile(boot[:,1]-boot[:,0],[2.5,97.5]),e['delta_pairacc_B_interval95'],atol=1e-12,rtol=0);ci+=1
for d in j['decomposition']:
    parts=[];direct=[]
    for split in ['train','val']:
        names=[x['image'] for x in cases if x['split']==split and x['actual_count'] in [1,2,3,4]];direct.append(np.mean([rebuilt[n,d['selector']]['delta_pairacc_B'] for n in names]))
    for x in d['parts']:
        vals=[];weights=[]
        for split in ['train','val']:
            names=[z['image'] for z in cases if z['split']==split and z['actual_count']==x['count']];total=sum(z['split']==split and z['actual_count'] in [1,2,3,4] for z in cases)
            vals.append(np.mean([rebuilt[n,d['selector']]['delta_pairacc_B'] for n in names]));weights.append(len(names)/total)
        comp=(weights[0]-weights[1])*sum(vals)/2;within=sum(weights)*(vals[0]-vals[1])/2
        assert abs(comp-x['composition'])<1e-12 and abs(within-x['within_count'])<1e-12;parts.append((comp,within))
    assert abs(math.fsum(x[0] for x in parts)-d['composition_component'])<1e-12 and abs(math.fsum(x[1] for x in parts)-d['within_count_component'])<1e-12
    assert abs(direct[0]-direct[1]-d['split_delta_gap'])<1e-12 and abs(d['composition_component']+d['within_count_component']-d['split_delta_gap'])<1e-12
scores=read(r/'scoring/scores.json');t=torch.load(r/'scoring/tensors.pt',map_location='cpu',weights_only=True);assert len(scores)==510 and t['record_ids']==[x['id'] for x in scores]
q64=t['image_features'].double()@f['quantity_text_features'].double().T;assert float((q64-t['quantity_similarities'].double()).abs().max())<1e-6
ix={x:i for i,x in enumerate(t['record_ids'])};v64=t['image_features'][[ix[x] for x in t['val_record_ids']]].double()@f['original_val_text_features'].double().T
assert float((v64-t['val_similarities'].double()).abs().max())<1e-6;vi={x:i for i,x in enumerate(t['val_record_ids'])};classes=ranks=0;control_index={}
for i,x in enumerate(scores):
    h=torch.load(r/x['state_file'],map_location='cpu',weights_only=True);assert h.shape==(3584,) and h.dtype==torch.bfloat16 and torch.isfinite(h).all()
    assert sha(r/x['state_file'])==x['state_file_sha256'] and tensor_sha(h)==x['state_tensor_sha256']
    if x['arm']=='zero_input':assert torch.count_nonzero(h)==0
    if x['arm'].startswith('shared_'):assert (r/x['state_file']).read_bytes()==(r/'control_states'/(x['arm']+'.pt')).read_bytes()
    for wording,cols in gallery['quantity_columns'][x['image']].items():
        a1,b1=[float(t['quantity_similarities'][i,z]) for z in cols];assert a1==x[wording+'_positive'] and b1==x[wording+'_negative'] and a1-b1==x['margin_'+wording]
        assert bool(q64[i,cols[0]]>q64[i,cols[1]])==x['correct_'+wording];classes+=1
    if x['split']=='val':
        row=v64[vi[x['id']]];pos=gallery['val_positive_columns'][x['image']];rank=int((row[:391]>row[pos].max()).sum());assert rank==x['retrieval_rank0'] and x['retrieval_rr']==1/(rank+1);ranks+=1
    pref,arg=two_values(t['quantity_similarities'][i],by[x['image']]);control_index[x['arm'],x['image']]={**x,'two_minus_one':pref,'argmax_two':arg}
    if x['arm']=='historical_cache':
        for wording in ['A','B']:
            for key in ['positive','negative']:assert abs(x[wording+'_'+key]-by[x['image']]['pairs'][wording]['historical_'+key+'_score'])<1e-6
for e in j['control_summary']:
    xs=[x for x in control_index.values() if x['arm']==e['arm'] and x['split']==e['split']]
    for metric,field in [('pairacc_B','correct_B'),('margin_B','margin_B'),('two_minus_one','two_minus_one'),('argmax_two','argmax_two')]:assert abs(np.mean([x[field] for x in xs])-e[metric])<1e-12
for e in j['shared_state_contrasts']:
    names=[x['image'] for x in cases if x['split']==e['split']];m=e['metric'];d=np.array([float(control_index['shared_eos',n][m])-float(control_index['shared_answer_close',n][m]) for n in names]);rng=np.random.default_rng(seed('shared/'+e['split']+'/'+m));draw=rng.integers(0,len(d),size=(10000,len(d)))
    assert abs(float(d.mean())-e['mean'])<1e-12 and np.allclose(np.percentile(d[draw].mean(axis=1),[2.5,97.5]),e['interval95'],atol=1e-12,rtol=0);ci+=1
mapping=read(r/'cache_mapping.json');loader=read(r/'dataset_loader_replay.json');assert len(mapping)==len(loader)==85
for x in mapping:
    case=by[x['image']];assert x['row']==case['row'] and x['split']==case['split'] and x['file_sha256']==case['historical_prior_sha256'] and sha(r/x['state_file'])==x['file_sha256']
    checks=next(z for z in loader if z['image']==x['image']);assert checks['row']==x['row'] and checks['cache_file']==x['source_file']
    assert all(z['image_row']==x['row'] for z in checks['actual_dataset_get_image_checks'])
contracts=read(r/'checkpoint_cache_contracts.json');assert contracts['utility']['sha256']==plan['utility_model_sha256'] and contracts['proxy']['sha256']==plan['proxy_sha256']
assert read(r/'scoring_pipeline_result.json')['status']=='complete'
dump(a.out,{'ok':True,'source_pair_records_rebuilt':425,'per_image_rows_rebuilt':len(rebuilt),'control_and_anchor_records_checked':510,
 'float64_quantity_classifications_checked':classes,'float64_val_ranks_checked':ranks,'stratified_rows_rebuilt':len(j['stratified']),'bootstrap_intervals_rebuilt':ci,
 'count_standardizations_rebuilt':4,'composition_decompositions_rebuilt':2,'historical_cache_mappings_checked':85,
 'parent_425_EOS_and_935_SFT_trajectories_recursively_reverified':True,'optimizer_updates':0,'torch':torch.__version__,'numpy':np.__version__})
print('ALL_SPLIT_DIAGNOSTIC_EVIDENCE_VERIFIED')

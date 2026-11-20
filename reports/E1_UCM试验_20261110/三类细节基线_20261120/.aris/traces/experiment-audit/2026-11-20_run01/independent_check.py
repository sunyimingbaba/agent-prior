"""Read-only reconstruction of stored evidence; writes only to this audit directory."""
from pathlib import Path
import collections, csv, hashlib, json, statistics
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_bytes())
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def csvrows(p): return list(csv.DictReader(p.open(encoding='utf-8-sig')))

cases = read(ROOT/'inputs/cases.json')
inventory = read(ROOT/'inputs/scored_images.json')
quantity_source = csvrows(ROOT.parent/'第十步_统一句式与对照方案/100组两种句式_对照清单.csv')
expected_quantity = [dict(image=x['图像编号'], split=x['数据来源'].split()[-1], dimension='quantity', wording=x['句式'][0], positive=x['正确描述'], negative=x['错误描述'], source=x['用户确认状态']) for x in quantity_source]
assert [x for x in cases if x['dimension']=='quantity'] == expected_quantity
pair_labels = list(csv.DictReader((ROOT/'inputs/paired_visual_labels.tsv').open(), delimiter='\t'))
for row in pair_labels:
    for side, other in [('a','b'),('b','a')]:
        c = next(c for c in cases if c['image']==row['image_'+side]+'.tif' and c['dimension']==row['dimension'])
        assert c['positive']==row['text_'+side] and c['negative']==row['text_'+other] and c['split']==row['split']
assert len({(x['image'],x['dimension'],x['wording']) for x in cases})==len(cases)
assert {x['image'] for x in cases}=={x['image'] for x in inventory}
summary={ (x['model'],x['split'],x['task']):x for x in csvrows(ROOT/'baseline_summary.csv') }
evidence = {'cases':len(cases), 'unique_images':len(inventory), 'quantity_source_matches':True, 'pair_labels_match':True,
            'unique_rgb_hashes':len({x['rgb_sha256'] for x in inventory}), 'inventory_splits':dict(collections.Counter(x['split'] for x in inventory)),
            'task_splits':dict(collections.Counter(x['split']+'/'+x['dimension']+'/'+x['wording'] if x['dimension']=='quantity' else x['split']+'/'+x['dimension'] for x in cases)),
            'models':{}, 'pair_both_correct':{}, 'tools':{}, 'input_hashes':{}}
sets={d:{x['image'] for x in cases if x['dimension']==d} for d in ['quantity','attribute','relation']}
evidence['cross_dimension_overlap']={a+'/'+b:sorted(sets[a]&sets[b]) for a,b in [('quantity','attribute'),('quantity','relation'),('attribute','relation')]}
evidence['wording_balance']={}
for d in sets:
    rows=[c for c in cases if c['dimension']==d and (d!='quantity' or c['wording']=='A')]
    p=collections.Counter(x['positive'] for x in rows);n=collections.Counter(x['negative'] for x in rows)
    evidence['wording_balance'][d]={'balanced_per_exact_sentence':p==n,'positive_counts':dict(p),'negative_counts':dict(n)}
allrows={}
for model in ['CLIP','CUP_CCB','ASPF','AGSE','CLIP_legacy','AGSE_archived']:
    folder=ROOT/'outputs'/model
    meta=read(folder/'settings.json');rows=read(folder/'rows.json');allrows[model]=rows
    assert meta['cases_sha256']==sha(ROOT/'inputs/cases.json')
    assert meta['scored_images_sha256']==sha(ROOT/'inputs/scored_images.json')
    assert meta['feature_sha256']==sha(folder/'features.pt')
    f=torch.load(folder/'features.pt',map_location='cpu',weights_only=True)
    assert f['image_names']==[x['image'] for x in inventory]
    I=f['image_features'].numpy().astype(np.float64);T=f['text_features'].numpy().astype(np.float64)
    assert np.isfinite(I).all() and np.isfinite(T).all()
    norms=max(np.abs(np.linalg.norm(I,axis=1)-1).max(),np.abs(np.linalg.norm(T,axis=1)-1).max())
    assert norms<1e-5 and len(rows)==len(cases)
    ni={n:i for i,n in enumerate(f['image_names'])};nt={t:i for i,t in enumerate(f['texts'])}
    groups=collections.defaultdict(list);worst=0
    for row,case in zip(rows,cases):
        assert all(row[k]==v for k,v in case.items()) and row['model']==model
        pos=float(np.dot(I[ni[row['image']]],T[nt[row['positive']]]))
        neg=float(np.dot(I[ni[row['image']]],T[nt[row['negative']]]))
        err=max(abs(pos-row['positive_cos']),abs(neg-row['negative_cos']),abs(pos-neg-row['margin']))
        assert err<1e-10 and row['correct']==(pos>neg) and row['tie']==(pos==neg)
        worst=max(worst,err)
        task=row['dimension']+'_'+row['wording'] if row['dimension']=='quantity' else row['dimension']
        for split in ['all',row['split']]:groups[split,task].append(row)
    stat=[]
    for (split,task),rr in sorted(groups.items()):
        x={'split':split,'task':task,'images':len(rr),'correct':sum(z['correct'] for z in rr),'accuracy':statistics.mean(z['correct'] for z in rr),'mean_cosine_margin':statistics.mean(z['margin'] for z in rr),'ties':sum(z['tie'] for z in rr)}
        stat.append(x)
        if model!='AGSE_archived':
            s=summary[model,split,task]
            for k in ['images','correct','accuracy','mean_cosine_margin','ties']: assert abs(float(s[k])-x[k])<1e-12
    evidence['models'][model]={'rows_verified':len(rows),'max_fp64_error':worst,'max_l2_norm_error':float(norms),'summary':stat}
    lookup={(z['image'],z['dimension']):z for z in rows if z['dimension']!='quantity'}
    pp=collections.defaultdict(list)
    for row in rows:
        if row.get('side')=='a':
            other=lookup[row['matched_image'],row['dimension']]
            assert row['positive']==other['negative'] and row['negative']==other['positive']
            pp[row['dimension']].append(int(row['correct'])+int(other['correct']))
    evidence['pair_both_correct'][model]={k:{'pairs':len(v),'both_correct':v.count(2),'one_correct':v.count(1),'neither_correct':v.count(0)} for k,v in pp.items()}
    if model in ['ASPF','AGSE']:
        tracecounts=collections.Counter();used=0
        for rec in inventory:
            p=ROOT/'priors'/model/(Path(rec['image']).stem+'.pt');j=read(p.with_suffix('.json'))
            assert j['sha256']==sha(p) and j['input']==rec and j['query']=='Describe the visible scene.' and j['method']==model
            h=torch.load(p,map_location='cpu',weights_only=True)
            assert h.dtype==torch.bfloat16 and tuple(h.shape)==(3584,) and torch.isfinite(h).all()
            assert j['used']==len(j['events']);used+=bool(j['used']);tracecounts[j['trace']]+=1
        assert used==meta['images_with_tool_use']
        evidence['tools'][model]={'images_with_tool_use':used,'unique_traces':len(tracecounts),'empty_first_output':tracecounts["  [round 1] 模型输出: ''\n"]}
previous=read(ROOT.parent/'第十三步_100图完整评分/完整评分结果.json')
old={(x['model'],x['image'],x['wording'][0]):x for x in previous}
legacy=[(x,old['CLIP',x['image'],x['wording']]) for x in allrows['CLIP_legacy'] if x['dimension']=='quantity']
evidence['legacy']={'comparisons':len(legacy),'agreement':sum(a['correct']==b['correct'] for a,b in legacy),'max_margin_difference':max(abs(a['margin']-b['margin']) for a,b in legacy)}
proof=read(ROOT/'score_verification.json')
assert proof['items_per_method']==len(cases) and proof['unique_scored_images']==len(inventory)
assert proof['legacy_clip_agreement']==evidence['legacy']['agreement'] and proof['legacy_clip_max_margin_difference']==evidence['legacy']['max_margin_difference']
native=read(ROOT/'native_token_control.json');evidence['native_controls']={}
for mode in ['generic','original']:
    rows=[x for x in native['records'] if x['mode']==mode]
    for x in rows:
        ids=x['rounds'][0]['ids'];assert x['first_output_is_single_EOS']==(len(ids)==1 and ids[0] in native['configured_eos_ids'])
    evidence['native_controls'][mode]={'n':len(rows),'single_eos':sum(x['first_output_is_single_EOS'] for x in rows),'same_hidden':sum(x['same_hidden'] for x in rows),'tool_calls':sum(x['tools'] for x in rows)}
bindings=read(ROOT/'asset_bindings.json')['files'];evidence['code_binding']={}
for fn in ['CUP/agent/rollout.py','CUP/agent/tools.py','CUP/prompt_clip/model.py','CUP/vilt/modules/vilt_module.py']:
    local=ROOT.parents[2]/fn
    expected=next(v['sha256'] for k,v in bindings.items() if k.endswith(fn))
    evidence['code_binding'][fn]=sha(local)==expected;assert evidence['code_binding'][fn]
paths=[ROOT/'inputs/cases.json',ROOT/'inputs/scored_images.json',ROOT/'inputs/paired_visual_labels.tsv',ROOT/'asset_bindings.json',ROOT/'native_token_control.json',ROOT/'baseline_summary.csv',ROOT/'score_verification.json']
paths+=list((ROOT/'code').glob('*.py'))+list((ROOT/'outputs').glob('*/settings.json'))+list((ROOT/'outputs').glob('*/rows.json'))+list((ROOT/'outputs').glob('*/features.pt'))+list((ROOT/'review').glob('pairs_*.jpg'))
for p in paths:evidence['input_hashes'][str(p.relative_to(ROOT))]='sha256:'+sha(p)
(OUT/'independent_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in evidence.items() if k not in ['input_hashes','wording_balance','models']},ensure_ascii=False,indent=2))
print('All saved scores, summaries, hashes, labels, and metadata checks passed; no model inference performed.')

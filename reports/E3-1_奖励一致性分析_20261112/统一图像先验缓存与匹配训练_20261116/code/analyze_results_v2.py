"""Fixed test contrasts, paired image-cluster intervals and quantity diagnostics."""
import csv
import hashlib
from pathlib import Path
import numpy as np
import torch
from stage_io import read, dump

r = Path(__file__).resolve().parent.parent
plan = read(r/'post_plan.json')
assert read(r/'matched_pipeline_result_v2.json')['status'] == 'complete'
g = read(r/'test/gallery.json')
owners = np.array(g['owners'])
groups = np.array(g['groups'])
counts = np.bincount(owners, minlength=210)
assert len(counts) == 210
values = {}
summary = {}
per_image = []
for arm in read(r/'evaluation_plan.json')['test_arms']:
    t = torch.load(r/'test'/arm/'tensors.pt', map_location='cpu', weights_only=True)
    ih = t['i2t_hits'].numpy()
    th = t['t2i_hits'].numpy()
    # Supplemental strict image correspondence is fixed before test evaluation.
    si = np.array([[any(owners[j] == i for j in t['top_i'][i,:k].tolist())
                    for k in [1,5,10]] for i in range(210)], dtype=float)
    st = np.array([[owners[j] in t['top_t'][:k,j].tolist() for k in [1,5,10]]
                   for j in range(len(owners))], dtype=float)
    for protocol, a, b in [('repository_duplicate_token_positive',ih,th),('strict_image_correspondence',si,st)]:
        sums = np.stack([b[owners == i].sum(0) for i in range(210)])
        values[protocol,arm] = (a, sums)
        rs = np.r_[a.mean(0), b.mean(0)]
        summary.setdefault(protocol,{})[arm] = {k:float(v) for k,v in zip(
            ['I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10','mR'],np.r_[rs,rs.mean()])}
        for i, name in enumerate(g['names']):
            per_image.append({'protocol':protocol,'arm':arm,'image':name,
                'captions':int(counts[i]),'I2T_hits':a[i].tolist(),'T2I_hit_sums':sums[i].tolist()})
contrasts = []
for protocol in summary:
    for left,right in plan['test_contrasts']:
        label = protocol+'/'+left+'-minus-'+right
        a,b = values[protocol,left]
        c,d = values[protocol,right]
        rng = np.random.default_rng((plan['bootstrap_seed']+int(hashlib.sha256(label.encode()).hexdigest()[:8],16))%2**32)
        draws = rng.integers(210,size=(10000,210))
        boot = ((a-c)[draws].mean(1).sum(1)+((b-d)[draws].sum(1)/counts[draws].sum(1)[:,None]).sum(1))/6
        contrasts.append({'protocol':protocol,'left':left,'right':right,
            'delta_mR':summary[protocol][left]['mR']-summary[protocol][right]['mR'],
            'interval95':np.percentile(boot,[2.5,97.5]).tolist(),'images':210,'caption_queries':len(owners)})
quantity = read(r/'quantity/records.json')
quantity_summary = {}
for split in ['train','val']:
    quantity_summary[split] = {}
    for arm in read(r/'training_plan_v2.json')['arms']:
        xs = [x for x in quantity if x['split']==split and x['arm']==arm]
        common = [x for x in xs if x['actual_count'] in [1,2,3,4]]
        entries = {c:[x for x in common if x['actual_count']==c] for c in [1,2,3,4]}
        metrics = {'images':len(xs),'common_count_images':len(common),
                   'count_sample_sizes':{str(c):len(v) for c,v in entries.items()}}
        for wording in ['A','B']:
            for metric in ['pair_correct','hard_correct','hard_margin']:
                key = wording+'_'+metric
                metrics[key+'_image_mean'] = float(np.mean([x[key] for x in xs]))
                metrics[key+'_counts1to4_macro'] = (float(np.mean([np.mean([x[key] for x in v])
                   for v in entries.values()])) if all(entries.values()) else None)
        quantity_summary[split][arm] = metrics
history = {}
for arm in read(r/'training_plan_v2.json')['arms']:
    d = r/'training'/arm
    history[arm] = {'initial':read(d/'initial_validation/metrics.json'),
                   'epochs':[{'epoch':e,**read(d/f'validation/epoch_{e:02d}/metrics.json')} for e in range(1,11)],
                   'selected':read(r/'selection_lock.json')['arms'][arm],
                   'training_result':read(d/'training_result_v2.json')}
dump(r/'analysis.json',{'ok':True,'test_summary':summary,'paired_mR_intervals':contrasts,
     'training_history':history,'quantity_diagnostics':quantity_summary,
     'quantity_scope':'train63 are within readout training; val22 are a diagnostic subset of validation; original count labels/candidate texts fixed',
     'optimizer_steps_total':1280,'readout_seed_count':1,'all_declared_arms_reported':True})
with (r/'per_image_test_metrics.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(per_image[0]),lineterminator='\n');w.writeheader()
    w.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()} for row in per_image)
print('MATCHED_TEST_AND_QUANTITY_ANALYSIS_COMPLETE',len(contrasts),flush=True)

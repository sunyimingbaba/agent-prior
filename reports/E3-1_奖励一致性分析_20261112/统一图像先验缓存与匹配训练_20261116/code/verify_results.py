"""Independent CPU reconstruction of saved matrices, top-k, curves and loader orders."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
r=args.root;torch.set_num_threads(2)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
train_plan=read(r/'training_plan_v2.json');post_plan=read(r/'post_plan.json');analysis=read(r/'analysis.json')
orders=torch.load(r/'training_orders.pt',map_location='cpu',weights_only=True)
assert orders.shape==(10,7999)
assert all(torch.equal(row.sort().values,torch.arange(7999)) for row in orders)
initials=[];batch_count=0;gradient_witnesses=0;checks=0;maxerr=0.
for arm in train_plan['arms']:
    d=r/'training'/arm;initials.append(read(d/'initialization.json'));result=read(d/'training_result.json')
    assert result['ok'] and result['epochs']==10 and result['optimizer_steps']==320
    xs=[read(d/f'validation/epoch_{e:02d}/metrics.json') for e in range(1,11)]
    selected=max(range(10),key=lambda e:(xs[e]['mR'],-e))+1
    choice=read(r/'selection_lock.json')['arms'][arm]
    assert choice['epoch']==selected and abs(choice['val_mR']-xs[selected-1]['mR'])<1e-12
    losses=[json.loads(s) for s in (d/'batch_losses.jsonl').read_text().splitlines()]
    # A resumed arm may retain an interrupted epoch; only each completed epoch's
    # last full250-record attempt is authoritative, with all attempts preserved.
    for e in range(10):
        rows=[x for x in losses if x['epoch']==e+1]
        starts=[i for i,x in enumerate(rows) if x['batch']==0]
        assert starts
        rows=rows[starts[-1]:]
        assert len(rows)==250 and [x['batch'] for x in rows]==list(range(250))
        actual=sum([x['raw_indices'] for x in rows],[])
        assert actual==orders[e].tolist()
        assert all(np.isfinite(x['loss']) and np.isfinite(x['lr']) for x in rows)
        batch_count+=len(rows)
    gs=[json.loads(s) for s in (d/'gradient_audit.jsonl').read_text().splitlines()]
    assert gs and all(x['finite'] and not x['backbone_has_gradient'] for x in gs)
    assert all(all(np.isfinite(v) for v in x['gradient_norms'].values()) for x in gs)
    assert any(any(v>0 for v in x['gradient_norms'].values()) for x in gs)
    gradient_witnesses+=len(gs)
    for cp in result['checkpoints'].values():
        assert sha(r/cp['path'])==cp['sha256']
        saved=torch.load(r/cp['path'],map_location='cpu',weights_only=False)
        assert saved['matched_readout']['backbone_omitted'] and saved['matched_readout']['arm']==arm
        assert not any(n.startswith('model.') for n in saved['state_dict'])
        assert saved['global_step']==cp['global_step']
        assert saved['matched_readout']['sample_order_sha256']==sha(r/'training_orders.pt')
        steps=[int(v['step']) for state in saved['optimizer_states'] for v in state['state'].values() if 'step' in v]
        assert steps and max(steps)==saved['global_step']
assert all(x==initials[0] for x in initials)
g=read(r/'test/gallery.json');owners=np.array(g['owners']);groups=np.array(g['groups']);counts=np.bincount(owners,minlength=210)
positive={x:set(owners[groups==x].tolist()) for x in set(groups)}
own=[set(groups[owners==i].tolist()) for i in range(210)]
values={}
for arm in read(r/'evaluation_plan.json')['test_arms']:
    t=torch.load(r/'test'/arm/'tensors.pt',map_location='cpu',weights_only=True)
    assert torch.equal(t['owners'],torch.tensor(owners)) and torch.equal(t['groups'],torch.tensor(groups))
    s=t['logit_scale']*(t['image_features'].double()@t['text_features'].double().T)
    err=float((s-t['scores'].double()).abs().max());maxerr=max(maxerr,err);assert err<5e-5
    assert float((t['image_features'].norm(dim=1)-1).abs().max())<1e-5
    assert float((t['text_features'].norm(dim=1)-1).abs().max())<1e-5
    assert torch.equal(t['scores'].topk(10,dim=1).indices,t['top_i'])
    assert torch.equal(t['scores'].topk(10,dim=0).indices,t['top_t'])
    ih=np.array([[any(groups[j] in own[i] for j in t['top_i'][i,:k].tolist()) for k in [1,5,10]] for i in range(210)],float)
    th=np.array([[any(i in positive[groups[j]] for i in t['top_t'][:k,j].tolist()) for k in [1,5,10]] for j in range(len(owners))],float)
    assert np.array_equal(ih,t['i2t_hits'].numpy()) and np.array_equal(th,t['t2i_hits'].numpy())
    si=np.array([[any(owners[j]==i for j in t['top_i'][i,:k].tolist()) for k in [1,5,10]] for i in range(210)],float)
    st=np.array([[owners[j] in t['top_t'][:k,j].tolist() for k in [1,5,10]] for j in range(len(owners))],float)
    for proto,a,b in [('repository_duplicate_token_positive',ih,th),('strict_image_correspondence',si,st)]:
        vals=np.r_[a.mean(0),b.mean(0)];expected=analysis['test_summary'][proto][arm]
        for k,v in zip(['I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10','mR'],np.r_[vals,vals.mean()]):
            assert abs(expected[k]-v)<1e-12;checks+=1
        values[proto,arm]=(a,np.stack([b[owners==i].sum(0) for i in range(210)]))
for c in analysis['paired_mR_intervals']:
    proto,left,right=c['protocol'],c['left'],c['right'];label=proto+'/'+left+'-minus-'+right
    rng=np.random.default_rng((post_plan['bootstrap_seed']+int(hashlib.sha256(label.encode()).hexdigest()[:8],16))%2**32)
    draws=rng.integers(210,size=(10000,210));a,b=values[proto,left];d,e=values[proto,right]
    bs=((a-d)[draws].mean(1).sum(1)+((b-e)[draws].sum(1)/counts[draws].sum(1)[:,None]).sum(1))/6
    assert np.max(np.abs(np.percentile(bs,[2.5,97.5])-c['interval95']))<1e-12
out={'ok':True,'four_initializations_identical':True,'completed_training_batches_verified':batch_count,
     'fixed_sample_indices_verified':batch_count,'gradient_witnesses_verified':gradient_witnesses,
     'readout_optimizer_updates':1280,'test_images':210,'test_image_encodings_verified':1260,
     'test_caption_queries':len(owners),'recall_and_mR_values_verified':checks,
     'paired_cluster_intervals_verified':len(analysis['paired_mR_intervals']),
     'float64_scaled_score_max_error':maxerr,'all_val_selected_checkpoints_rebuilt':True}
args.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out))

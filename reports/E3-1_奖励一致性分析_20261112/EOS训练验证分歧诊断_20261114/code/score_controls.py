import json
from pathlib import Path
import time
import pyarrow as pa
from milestone_common import ROOT,PLAN,setup,seed,ASSETS
from diagnostic_io import read,dump,sha,tensor_sha,hydrate,paired_records
from quantity_model import QuantityScorer

torch=setup();seed(0);start=time.monotonic();parent=hydrate();cases=read(ROOT/'inputs/evaluation_cases.json');scorer=QuantityScorer(read(ROOT/'inputs/cases.json'))
assert not any(p.requires_grad for p in scorer.pl.parameters()) and not scorer.pl.training
new,old=paired_records(parent);ng={x['image']:x for x in new if x['role']=='greedy'};og={x['image']:x for x in old.values() if x['role']=='greedy'}
history={x['image']:x for x in read(ROOT/'cache_mapping.json')};controls=ROOT/'control_states';controls.mkdir()
zero=torch.zeros(3584,dtype=torch.bfloat16);torch.save(zero,controls/'zero.pt')
import shutil
ref=PLAN['reference_image'];assert ref==next(x['image'] for x in cases if x['split']=='train')
shutil.copy2(parent/'reference_run'/og[ref]['state_file'],controls/'shared_answer_close.pt');shutil.copy2(parent/ng[ref]['state_file'],controls/'shared_eos.pt')
from vilt.datasets.base_dataset import BaseDataset
datasets={};loader_audit=[]
for split in ['train','val']:
    ds=BaseDataset.__new__(BaseDataset);ds.table=pa.ipc.open_file(str(ASSETS/'ucm'/f'ucm_caption_karpathy_label_{split}.arrow')).read_all()
    ds.table_names=[f'ucm_caption_karpathy_label_{split}']*len(ds.table);ds.preprocess=scorer.pl.preprocess;ds.use_agent_prior=True;ds.agent_cache_dir=PLAN['historical_cache'];ds.index_mapper={}
    for row,captions in enumerate(ds.table['caption'].to_pylist()):
        for cap in range(len(set(captions))):ds.index_mapper[len(ds.index_mapper)]=(row,cap)
    datasets[split]=ds
for x in cases:
    ds=datasets[x['split']];ids=[i for i,(row,cap) in ds.index_mapper.items() if row==x['row']];checks=[]
    for idx in sorted(set([ids[0],ids[-1]])):
        item=ds.get_image(idx);assert item['img_index']==x['row'] and item['path'].as_py()==x['image'] and tensor_sha(item['agent_hidden'])==history[x['image']]['tensor_sha256']
        checks.append({'sample_index':idx,'image_row':item['img_index'],'caption_index':item['cap_index']})
    loader_audit.append({'image':x['image'],'split':x['split'],'row':x['row'],'cache_file':history[x['image']]['source_file'],'actual_dataset_get_image_checks':checks})
dump(ROOT/'dataset_loader_replay.json',loader_audit)
out=ROOT/'scoring';out.mkdir();records=[];features=[];qs=[];vs=[];vids=[];anchor_max=0.
for x in cases:
    source={'historical_cache':ROOT/history[x['image']]['state_file'],'zero_input':controls/'zero.pt','shared_answer_close':controls/'shared_answer_close.pt','shared_eos':controls/'shared_eos.pt',
      'own_answer_close':parent/'reference_run'/og[x['image']]['state_file'],'own_eos':parent/ng[x['image']]['state_file']}
    for arm,path in source.items():
        h=torch.load(path,map_location='cpu',weights_only=True);value=scorer.score(h,x)
        reference=og[x['image']] if arm=='own_answer_close' else ng[x['image']] if arm=='own_eos' else None
        if reference:
            for metric in ['A_positive','A_negative','B_positive','B_negative','margin_A','margin_B']:
                e=abs(value[metric]-reference[metric]);anchor_max=max(anchor_max,e);assert e<1e-6
            assert value['retrieval_rank0']==reference['retrieval_rank0']
        if arm=='historical_cache':
            for wording in ['A','B']:
                for key in ['positive','negative']:assert abs(value[wording+'_'+key]-x['pairs'][wording]['historical_'+key+'_score'])<1e-6
        rid=arm+'__'+x['split']+'_'+str(x['row'])
        records.append({'id':rid,'arm':arm,'image':x['image'],'split':x['split'],'row':x['row'],'state_file':str(path.relative_to(ROOT)),'state_file_sha256':sha(path),'state_tensor_sha256':tensor_sha(h),**value})
        features.append(scorer.last_feature);qs.append(scorer.last_qsim)
        if scorer.last_vsim is not None:vids.append(rid);vs.append(scorer.last_vsim)
    dump(ROOT/'progress.json',{'phase':'scoring_controls','completed':len(records),'total':510});print('CONTROL_SCORED',len(records),'/510',flush=True)
dump(out/'scores.json',records)
torch.save({'record_ids':[x['id'] for x in records],'image_features':torch.cat(features),'quantity_similarities':torch.stack(qs),'val_record_ids':vids,'val_similarities':torch.stack(vs)},out/'tensors.pt')
dump(out/'result.json',{'ok':True,'records':510,'new_controls':340,'replay_anchors':170,'anchor_maximum_abs_error':anchor_max,'historical_E1_scores_match':True,
 'dataset_image_checks':85,'dataset_caption_samples_checked':sum(len(x['actual_dataset_get_image_checks']) for x in loader_audit),'optimizer_updates':0,'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
print('ALL_510_CONTROLS_AND_ANCHORS_SCORED',flush=True)

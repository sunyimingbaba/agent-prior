import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
import torch
from tokenizers import Tokenizer
torch.set_num_threads(2)

def verify_training(root):
    r=Path(root)
    def read(n):return json.loads((r/n).read_text(encoding='utf-8'))
    plan=read('plan.json');cases=read('inputs/cases.json');sources=read('inputs/teacher_sources.json')
    examples=read('training/teacher_examples.json');result=read('training/result.json');settings=read('training/settings.json')
    assert len(sources)==15 and len(examples)==30 and result['optimizer_updates']==30
    assert {x['image'] for x in sources}==set(plan['training_images'])
    assert all(x['split']=='train' for x in cases if x['image'] in plan['training_images'])
    assert not set(plan['training_images'])&set(plan['evaluation_images'])
    tokenizer=Tokenizer.from_file(str(r/'tokenizer_snapshot/training_tokenizer.json'))
    assert hashlib.sha256((r/'tokenizer_snapshot/training_tokenizer.json').read_bytes()).hexdigest()==settings['tokenizer_snapshot_sha256']
    assert (r/'tokenization_gate/training_tokenizer.json').read_bytes()==(r/'tokenizer_snapshot/training_tokenizer.json').read_bytes()
    eos=tokenizer.token_to_id('<|im_end|>');image_pad=tokenizer.token_to_id('<|image_pad|>')
    source_by={x['image']:x for x in sources};ex_by={x['id']:x for x in examples};masked_body=active_count=0
    actual=read('training/actual_teacher_tools.json');assert len(actual)==15
    for x in examples:
        body=source_by[x['image']]['source_body'];text='<tool_call>scene_classify</tool_call>' if x['kind']=='call' else '<answer>'+body+'</answer>'
        encoded=tokenizer.encode(text,add_special_tokens=False);ids=encoded.ids+[eos]
        if x['kind']=='call':expected=[True]*len(ids)
        else:
            lo,hi=8,8+len(body);expected=[bool(end>start and all(c<lo or c>=hi for c in range(start,end))) for start,end in encoded.offsets]+[True]
            assert x['body_char_span']==[lo,hi];masked_body+=sum(not z for z in expected)
        assert ids==x['target_ids'] and expected==x['loss_mask'] and x['text']==text
        assert [list(z) for z in encoded.offsets]+[[-1,-1]]==x['offsets']
        assert sum(expected)==x['supervised_tokens'] and ids[-1]==eos and ids[0] not in x['eos_token_ids']
        active_count+=sum(expected)
        # Independently rebuild processor-expanded image padding from recorded image grid.
        plain=tokenizer.encode(x['prompt_text'],add_special_tokens=False).ids;assert plain.count(image_pad)==1
        grid=x['image_grid_thw'];assert len(grid)==1
        pads=math.prod(grid[0])//(x['image_merge_size']**2);prompt=[]
        for token in plain:prompt.extend([image_pad]*pads if token==image_pad else [token])
        assert prompt==x['prompt_token_ids'] and len(prompt)==x['prompt_tokens']
        assert hashlib.sha256(x['prompt_text'].encode()).hexdigest()==x['prompt_text_sha256']
        users=re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>',x['prompt_text'],re.S)
        if x['kind']=='call':assert users[-1].endswith('查询文本: '+plan['prompt'])
        else:assert users[-1]=='<result>'+actual[x['image']]['text']+'</result>'
    gate=read('tokenization_gate/result.json');assert gate['ok'] and len(gate['targets'])==30
    assert active_count==sum(x['supervised_tokens'] for x in gate['targets'])
    for target in gate['targets']:
        actual=next(x for x in examples if x['image']==target['image'] and x['kind']==target['kind'])
        assert actual['target_ids']==target['target_ids'] and actual['loss_mask']==target['loss_mask']
    order=[];rng=np.random.default_rng(plan['training']['seed']);names=[x['image'] for x in sources]
    for epoch in range(2):order.extend({'epoch':epoch+1,'image':names[int(i)]} for i in rng.permutation(len(names)))
    assert order==read('training/order.json') and len(order)==30
    updates=[json.loads(s) for s in (r/'training/updates.jsonl').read_text(encoding='utf-8').splitlines()];assert len(updates)==30
    previous=plan['policy']['lora_tensor_hash'];loss_terms=0
    def check_stats(row,example):
        nonlocal loss_terms
        st=row['stats'];mask=example['loss_mask'];targets=[a for a,b in zip(example['target_ids'],mask) if b]
        assert st['target_ids_active']==targets and st['active_tokens']==sum(mask)
        assert st['prompt_tokens']==len(example['prompt_token_ids']) and st['response_tokens']==len(example['target_ids'])
        assert st['forward_tokens']==st['prompt_tokens']+st['response_tokens'] and st['prompt_loss_tokens']==st['body_loss_tokens']==0
        assert st['temperature']==1. and st['first_token_eos_suppressed']
        assert len(st['nll_per_active_token'])==len(targets) and all(math.isfinite(v) and v>=0 for v in st['nll_per_active_token'])
        assert abs(math.fsum(st['nll_per_active_token'])-row['loss_sum'])<1e-4
        assert st['active_argmax_correct']==sum(a==b for a,b in zip(targets,st['argmax_ids_active']))
        loss_terms+=len(targets)
    for step,(update,item) in enumerate(zip(updates,order),1):
        assert update['step']==step and update['epoch']==item['epoch'] and update['image']==item['image'] and update['optimizer_update']
        assert update['parameter_hash_before']==previous and update['parameter_hash_after']!=previous
        assert math.isfinite(update['gradient_norm_before_clip']) and update['gradient_norm_before_clip']>0
        total=0
        for row in update['examples']:
            example=ex_by[row['example_id']];assert example['image']==item['image'];check_stats(row,example);total+=example['supervised_tokens']
        assert len(update['examples'])==2 and update['supervised_tokens']==total
        assert abs(math.fsum(x['loss_sum'] for x in update['examples'])/total-update['loss'])<1e-10
        previous=update['parameter_hash_after']
    assert previous==result['final_parameter_hash'] and previous!=plan['policy']['lora_tensor_hash']
    warm=read('training/gradient_witness.json');assert warm['ok'] and warm['optimizer_updates']==0 and warm['parameter_hash_unchanged']==plan['policy']['lora_tensor_hash']
    assert math.isfinite(warm['gradient_norm']) and warm['gradient_norm']>0
    probability_rows=0;max_probability_error=0.
    for label in ['initial','final']:
        evaluation=read('training/teacher_evaluation_'+label+'.json');rows={x['example_id']:x for x in evaluation['examples']};assert len(rows)==30
        for row in rows.values():check_stats(row,ex_by[row['example_id']])
        assert sum(x['stats']['active_tokens'] for x in rows.values())==evaluation['active_tokens']==active_count
        assert abs(math.fsum(x['loss_sum'] for x in rows.values())-evaluation['loss_sum'])<1e-8
        assert abs(evaluation['loss_sum']/active_count-evaluation['global_mean_nll'])<1e-12
        witness=torch.load(r/('training/logit_witness_'+label+'.pt'),map_location='cpu',weights_only=True)
        for key,logits in witness.items():
            example=ex_by[key];assert example['image']==evaluation['witness_image']
            assert logits.ndim==2 and torch.isfinite(logits).all()
            values=logits.double();values[0,example['eos_token_ids']]=-torch.inf
            targets=torch.tensor(rows[key]['stats']['target_ids_active'],dtype=torch.long)
            nll=-torch.log_softmax(values,dim=-1).gather(1,targets[:,None]).flatten()
            error=float((nll-torch.tensor(rows[key]['stats']['nll_per_active_token'],dtype=torch.float64)).abs().max())
            assert error<1e-4;max_probability_error=max(max_probability_error,error);probability_rows+=len(targets)
    backup=read('模型本机核验.json');assert backup['ok'] and backup['optimizer_updates']==30 and backup['live_adapter_parameter_hash']==previous
    assert backup['optimizer']['parameter_states']==584 and backup['optimizer']['min_parameter_step']==backup['optimizer']['max_parameter_step']==30
    assert settings['trainable_tensors']==584 and settings['base_parameters_frozen'] and settings['all_adapter_parameters_float32']
    return {'ok':True,'training_images':15,'assistant_targets_rebuilt':30,'optimizer_updates_rebuilt':30,'parameter_hash_chain_verified':True,
      'format_positions_per_teacher_pass':active_count,'body_positions_masked':masked_body,'loss_terms_checked':loss_terms,
      'float64_full_vocabulary_probability_rows_checked':probability_rows,'maximum_abs_probability_nll_error':max_probability_error,
      'final_parameter_hash':previous,'trained_checkpoint_and_optimizer_backup_verified':True,'torch':torch.__version__,'numpy':np.__version__}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=verify_training(a.root);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('ALL_FORMAT_TRAINING_TARGETS_LOSSES_AND_UPDATES_VERIFIED')

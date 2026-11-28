"""Verify the raw prompt, processed pixels and naturally generated first action for all matched training slots."""
from collections import Counter
import hashlib
import re
import sys
import torch
from image_inputs import ROOT,FORM,POOL,IMAGE,QUERY,MODEL,AutoProcessor,expected_messages,tables,image,read,load,sha,dump

def main():
    p=ROOT/'matched';rows=read(p/'inputs/roster.json')['rows'];lock=read(p/'inputs/train_lock.json')
    assert [r['rgb_sha256'] for r in rows]==[r['rgb_sha256'] for r in read(POOL/'scoring/locks/training.json')['rows']]
    t=tables();proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    template=proc.apply_chat_template(expected_messages(),tokenize=False,add_generation_prompt=True)
    expected=read(ROOT/'audit/通用问题与原始输入独立核验.json');template_hash=hashlib.sha256(template.encode()).hexdigest()
    assert template_hash==expected['expected_chat_template_sha256']
    fb=read(p/'inputs/feedback_verified.json')
    summary={}
    for s in 'ab':
        base=p/'contexts'/s; proof=read(base/'complete.json');initial=read(IMAGE/'inputs/manifest.json')['policies'][s+'_r2']['initial']
        assert initial==lock['starts'][s]==proof['adapter']
        assert proof['lock']==sha(p/'inputs/train_lock.json') and proof['template']==template_hash and len(proof['files'])==24
        calls=0;answers=0;counts=Counter();chosen=[]
        for r in rows:
            im=image(dict(r,rgb=r['rgb_sha256']),t);n=r['image']
            packed=proc(text=[template],images=[im],return_tensors='pt');f=base/(n+'.pt')
            assert sha(f)==proof['files'][n]
            ctx=load(f)
            assert ctx['query']==QUERY and ctx['image']==n and ctx['rgb']==r['rgb_sha256'] and ctx['template']==template
            assert torch.equal(ctx['first_input_ids'],packed.input_ids) and ctx['token_hash']==expected['expected_tokens_by_image_size']['256x256']['digest']
            assert ctx['pixel_values_sha']==hashlib.sha256(packed['pixel_values'].contiguous().numpy().tobytes()).hexdigest()
            assert torch.equal(ctx['grid'],packed['image_grid_thw'])
            first=proc.decode(ctx['first'],skip_special_tokens=True)
            assert first==ctx['first_text'] and len(ctx['first'])>0
            call=re.search(r'<tool_call>\s*([\w_]+)\s*(?:</tool_call>)?',first)
            assert bool(call)==ctx['eligible']==proof['eligible'][n]
            if call:
                assert call[1]=='scene_classify' and ctx['calls']==[fb['images'][n]['feedback']]
                augmented=proc(text=[first+'\n<result>'+ctx['calls'][0]+'</result>\n'],return_tensors='pt').input_ids
                assert torch.equal(torch.cat((packed.input_ids,augmented),1),ctx['prefix']);calls+=1;chosen.append(n)
            else:
                assert ctx['calls']==[] and ctx['answer'] is None and ctx['prefix'] is None
            answers+=int('<answer>' in first);counts[len(ctx['first'])]+=1
        assert proof['available']==calls
        runs={}
        for a in ('r2','strict'):
            d=p/'models'/(s+'_'+a)
            if calls==0:
                z=read(d/'no_eligible.json');assert z['initial']==initial and z['scheduled_slots']==24 and z['steps']==0 and z['contexts']==sha(base/'complete.json')
                assert not (d/'steps.json').exists() and not (d/'adapter/adapter_model.safetensors').exists()
                runs[a]=dict(steps=0,optimizer_updates=0,saved_new_policy=False)
            else:
                z=read(d/'steps.json'); assert [k['image'] for k in z]==chosen
                runs[a]=dict(steps=len(z))
        summary[s]=dict(scheduled=len(rows),explicit_answers=answers,tool_calls=calls,sampling_groups=calls,
                        first_token_lengths=dict(counts),runs=runs)
    dump('同问题训练首轮与实际更新独立核查.json',summary)
if __name__=='__main__':
    torch.set_num_threads(2);main()

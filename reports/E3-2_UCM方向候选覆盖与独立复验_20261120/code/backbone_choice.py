"""Replay recorded forced-choice inputs with only the historical LoRA enabled/disabled."""
import hashlib
from PIL import Image
import numpy as np
import torch
from shared import ROOT,ASSETS,REBUILD,read,dump,sha,table,rgb,setup
from cases import cases
from rollout import load_mllm

@torch.inference_mode()
def main():
    setup();base=REBUILD/'paired_views_20261120'
    model,processor=load_mllm(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'),'cuda',str(ASSETS/'ucm_main_lora'))
    active=hashlib.sha256()
    for n,v in model.named_parameters():
        if '.lora_' in n:active.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    assert active.hexdigest()==read(base/'eval/base_00/complete.json')['active']
    def values(question,image,ids):
        messages=[{'role':'system','content':'Choose the correct description of the image. Reply with one letter.'},
                  {'role':'user','content':[{'type':'image'},{'type':'text','text':question}]}]
        tpl=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        x=processor(text=[tpl],images=[image],return_tensors='pt').to('cuda')
        l=model(**x,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        lps=l.log_softmax(-1)
        return dict(logit_a=float(l[ids[0]]),logit_b=float(l[ids[1]]),valid_mass=float(lps[ids].exp().sum()),top=int(l.argmax()))
    controls=0
    for group in ('fresh','train'):
        directory=ROOT/'choice_backbone'/group;directory.mkdir(exist_ok=True,parents=True)
        original=ROOT/'choice'/group;receipt=read(original/'complete.json')
        t=table('val' if group=='fresh' else 'train')
        for i,row in enumerate(cases(group)):
            native=original/(row['image']+'.json');p=read(native);ids=p['choice_ids'];assert sha(native)==receipt[row['image']]
            pic=rgb(t,row);views=(pic,pic.transpose(Image.Transpose.ROTATE_180),Image.new('RGB',pic.size,(127,127,127)))
            if (group,i)==('fresh',0):
                for r in p['records']:
                    check=values(r['query'],views[r['view']],ids)
                    assert all(check[key]==r[key] for key in check)
                    controls+=1
            records=[]
            with model.disable_adapter():
                for r in p['records']:
                    vals=values(r['query'],views[r['view']],ids)
                    records.append({**{key:r[key] for key in ('word','order','view')},**vals})
            dump(directory/(row['image']+'.json'),dict(image=row['image'],source=sha(native),records=records))
        print(group,flush=True)
    dump(ROOT/'choice_backbone/replay.json',dict(known_on_prompts_exact=controls,policy_sha=active.hexdigest(),
             experiment=sha(ROOT/'inputs/backbone_choice_protocol.json'),source=sha(ROOT/'code/backbone_choice.py')))
if __name__=='__main__':main()

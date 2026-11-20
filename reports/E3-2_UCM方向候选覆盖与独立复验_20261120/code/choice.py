"""Counterbalance A/B position while reading the same historical Agent's next-token distribution."""
import hashlib
import torch
from PIL import Image
from shared import ROOT,REBUILD,ASSETS,read,dump,sha,setup,table,rgb
from cases import cases
from rollout import load_mllm

@torch.inference_mode()
def main():
    setup();base=REBUILD/'paired_views_20261120'
    adapter=ASSETS/'ucm_main_lora'
    assert sha(adapter/'adapter_model.safetensors')==read(base/'environment.json')['files'][str(adapter/'adapter_model.safetensors')]['sha256']
    model,processor=load_mllm(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'),'cuda',str(adapter))
    active=hashlib.sha256()
    for n,p in model.named_parameters():
        if '.lora_' in n:active.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    assert active.hexdigest()==read(base/'eval/base_00/complete.json')['active']
    tokenizer=processor.tokenizer
    ids=[tokenizer.encode(k,add_special_tokens=False) for k in ('A','B')]
    assert [len(s) for s in ids]==[1,1] and ids[0]!=ids[1]
    a,b=ids[0][0],ids[1][0]
    prompts={}
    for group in ('train','fresh'):
        out=ROOT/'choice'/group;out.mkdir(exist_ok=True,parents=True)
        t=table('train' if group=='train' else 'val')
        for row in cases(group):
            f=out/(row['image']+'.json')
            if f.exists():
                assert read(f)['source']==sha(ROOT/'code/choice.py')
                continue
            im=rgb(t,row);views=[im,im.transpose(Image.Transpose.ROTATE_180),Image.new('RGB',im.size,(127,127,127))]
            vals=[]
            for w in ('A','B'):
                for order in (0,1):
                    options=row[w][::1 if order==0 else -1]
                    question='Which statement correctly describes the picture? Answer only A or B.\nA. '+options[0]+'\nB. '+options[1]
                    msgs=[{'role':'system','content':'Choose the correct description of the image. Reply with one letter.'},
                          {'role':'user','content':[{'type':'image'},{'type':'text','text':question}]}]
                    text=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
                    for k,image in enumerate(views):
                        inp=processor(text=[text],images=[image],return_tensors='pt').to('cuda')
                        logits=model(**inp,use_cache=False,logits_to_keep=1).logits[0,-1].float()
                        logp=logits.log_softmax(-1)
                        vals.append(dict(word=w,order=order,view=k,options=options,query=question,logit_a=float(logits[a]),logit_b=float(logits[b]),
                                         logprob_a=float(logp[a]),logprob_b=float(logp[b]),top=int(logits.argmax()),valid_mass=float(logp[[a,b]].exp().sum())))
            dump(f,dict(image=row['image'],records=vals,active=active.hexdigest(),choice_ids=[a,b],rgb=row['rgb_sha256'],source=sha(ROOT/'code/choice.py'),
                        design=sha(ROOT/'inputs/agent_choice_protocol.json'),lock=sha(ROOT/'inputs'/(group+'_lock.json'))))
        dump(out/'complete.json',{r['image']:sha(out/(r['image']+'.json')) for r in cases(group)})
        print(group,'finished',flush=True)
if __name__=='__main__':main()

"""Pass each verified pre-answer Agent state to its letter head and to the frozen CUP, on the same pixels."""
import hashlib
from PIL import Image
import torch
from cases import cases
from shared import ROOT,REBUILD,ASSETS,read,dump,sha,table,setup,rgb
from readouts import load_model
from rollout import load_mllm
@torch.inference_mode()
def main():
    setup();base=REBUILD/'paired_views_20261120'
    agent,proc=load_mllm(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'),'cuda',str(ASSETS/'ucm_main_lora'))
    active=hashlib.sha256()
    for n,p in agent.named_parameters():
        if '.lora_' in n:active.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    assert active.hexdigest()==read(base/'eval/base_00/complete.json')['active']
    cup=load_model();output_head=agent.get_output_embeddings()
    for group in ('train','fresh'):
        out=ROOT/'matched_choice'/group;out.mkdir(exist_ok=True,parents=True);t=table('val' if group=='fresh' else 'train')
        for row in cases(group):
            name=row['image'];native=ROOT/'choice'/group/(name+'.json')
            metadata=read(native);ids=metadata['choice_ids'];pic=rgb(t,row)
            views=(pic,pic.transpose(Image.Transpose.ROTATE_180))
            pixels=[cup.preprocess(im)[None].cuda() for im in views]
            hs=[];embs=[];log=[]
            for z in metadata['records']:
                view=z['view']
                if view==2:continue
                msgs=[{'role':'system','content':'Choose the correct description of the image. Reply with one letter.'},
                      {'role':'user','content':[{'type':'image'},{'type':'text','text':z['query']}]}]
                text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
                x=proc(text=[text],images=[views[view]],return_tensors='pt').to('cuda')
                y=agent(**x,use_cache=False,output_hidden_states=True,logits_to_keep=1)
                last=y.hidden_states[-1][:,-1,:].contiguous()
                logits=y.logits[0,-1].float()
                rehead=output_head(last).flatten().float()
                assert last.shape==(1,3584) and last.dtype==torch.bfloat16
                assert torch.equal(logits,rehead) and [float(logits[k]) for k in ids]==[z['logit_a'],z['logit_b']]
                feat=cup.img_embeds(pixels[view],agent_hidden=last)['image_feats'][0].cpu()
                assert torch.isfinite(feat).all() and abs(float(feat.norm())-1)<1e-5
                h=last[0].cpu()
                hs.append(h);embs.append(feat)
                log.append(dict(word=z['word'],view=view,order=z['order'],logits=[float(logits[k]) for k in ids],raw_top=int(logits.argmax()),
                                hidden=hashlib.sha256(h.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()))
            dest=out/(name+'.pt');torch.save(dict(hidden=torch.stack(hs),visual=torch.stack(embs)),dest)
            dump(out/(name+'.json'),dict(image=name,bundle=sha(dest),input=sha(native),source=sha(ROOT/'code/matched_choice.py'),
                    design=sha(ROOT/'inputs/matched_choice_protocol.json'),rows=log,active=active.hexdigest(),all_vocab_logits_exact=True))
        print(group,flush=True)
    dump(ROOT/'matched_choice/complete.json',{g:{r['image']:sha(ROOT/'matched_choice'/g/(r['image']+'.json')) for r in cases(g)} for g in ('train','fresh')})
if __name__=='__main__':main()

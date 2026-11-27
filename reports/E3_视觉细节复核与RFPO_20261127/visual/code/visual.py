"""Order-corrected C decisions from image and two attribute alternatives alone."""
from pathlib import Path
import json,hashlib,sys,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parents[1]
SRC=BASE/'second_answer_5090_20261124';sys.path.insert(0,str(SRC/'code'))
from shared import setup,rgb,table,sha,read
from policy import policy

def dump(path, data):
    temp=path.with_suffix('.part')
    temp.write_text(json.dumps(data,ensure_ascii=False)+'\n')
    temp.replace(path)

@torch.inference_mode()
def main():
    setup();model,proc=policy('R0');dataset=table('train')
    ids=[proc.tokenizer.encode(t,add_special_tokens=False) for t in ('A','B')]
    assert all(len(x)==1 for x in ids)
    ids=[x[0] for x in ids]
    folder=ROOT/'results/visual';folder.mkdir(parents=True,exist_ok=True)
    lock=read(ROOT/'inputs/lock.json')
    assert sha(ROOT/'inputs/rows.json')==lock['rows']
    hashes={}; prompts={}; aggregate={}
    for r in read(ROOT/'inputs/rows.json'):
        p=folder/(r['image']+'.json')
        if not p.exists():
            im=rgb(dataset,r); obs=[]
            for opts in (r['options_side0'],r['options_side0'][::-1]):
                text=f'Which statement describes the image? A. {opts[0]} B. {opts[1]} Answer with A or B.'
                message=[{'role':'user','content':[{'type':'image'},{'type':'text','text':text}]}]
                prompt=proc.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
                dat=proc(text=[prompt],images=[im],return_tensors='pt').to('cuda')
                logits=model(**dat,use_cache=False,return_dict=True).logits[0,-1].float()
                obs.append(dict(prompt=prompt,input_ids=hashlib.sha256(dat.input_ids.cpu().numpy().tobytes()).hexdigest(),
                                ab=logits[ids].tolist()))
            first,second=[a['ab'] for a in obs]
            margin=((first[0]-first[1])-(second[0]-second[1]))/2
            dump(p,dict(image=r['image'],rgb=r['rgb_sha256'],side=r['side'],options_side0=r['options_side0'],
                        margin_side0=margin,margin_true=(1-2*r['side'])*margin,observations=obs))
        v=read(p);assert v['rgb']==r['rgb_sha256'] and v['options_side0']==r['options_side0']
        key=r['cohort'],r['pair'];sig=tuple(x['input_ids'] for x in v['observations'])
        assert prompts.setdefault(key,sig)==sig
        hashes[r['image']]=sha(p);aggregate[r['image']]=v['margin_side0']
        if len(hashes)==24: dump(ROOT/'results/train_visual.json',aggregate)
        print(len(hashes),r['image'],flush=True)
    dump(ROOT/'results/visual_margins.json',aggregate)
    dump(ROOT/'results/visual_complete.json',dict(images=len(hashes),files=hashes,lock=sha(ROOT/'inputs/lock.json'),
         adapter=sha(SRC/'R0/adapter/adapter_model.safetensors'),code=sha(Path(__file__)),ids=ids,torch=str(torch.__version__)))
if __name__=='__main__':main()

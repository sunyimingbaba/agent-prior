"""Score each fixed candidate with the original RFPO formula and real AGSE embeddings."""
from pathlib import Path
import torch
from shared import ROOT,ARROW,ASSETS,BASE,SCORER,read,dump,sha,table,rgb,setup,locked
from readouts import clip,load_model,scorer_for
from agent.grpo import reward

def main():
    setup()
    cases=locked(); t=table()
    gallery=ROOT/'gallery/train.pt'
    index=read(ROOT/'gallery/index.json')
    assert sha(gallery)==index['gallery_sha256'] and sha(ARROW)==index['arrow_sha256'] and t['path'].to_pylist()==index['names']
    proxy=load_model(True); sc=scorer_for(proxy,gallery)
    actual=load_model(False)
    texts=sorted({c for r in cases for w in ('A','B') for c in r[w]})
    tokens=clip.tokenize(texts,context_length=77-actual.prompt_length,truncate=False)
    assert len(set(map(tuple,tokens.tolist())))==len(texts)
    out=ROOT/'scoring';out.mkdir(exist_ok=True)
    rows=[];visual=[];proxy_features=[]
    native=proxy.model.encode_image
    seen=[]
    def capture(*args,**kwargs):
        f=native(*args,**kwargs)
        seen.append((f/f.norm(dim=1,keepdim=True)).detach().cpu())
        return f
    proxy.model.encode_image=capture
    with torch.inference_mode():
        text_features=torch.cat([actual.txt_embeds(z.cuda())['text_feats'].cpu() for z in tokens.split(24)])
        for r in cases:
            img=actual.preprocess(rgb(t,r)).unsqueeze(0).cuda()
            assert index['names'][r['row']]==r['image']
            for mode in ('scene','original'):
                folder=ROOT/'samples'/mode/Path(r['image']).stem
                for k in range(-1,8):
                    name='greedy' if k<0 else f'sample{k}'
                    p=folder/(name+'.pt');proof=read(p.with_suffix('.json'))
                    assert proof['image']==r['image'] and proof['k']==k and proof['mode']==mode and proof['sha256']==sha(p)
                    h=torch.load(p,map_location='cpu',weights_only=True)
                    seen.clear()
                    rank,soft=sc.score(h.float().tolist(),r['row'])
                    assert len(seen)==1
                    proxy_features.append(seen[0])
                    v=actual.img_embeds(img,agent_hidden=h.unsqueeze(0).cuda())['image_feats'].cpu()
                    visual.append(v)
                    sims=(v.double()@text_features.double().T).flatten()
                    trajectory='\n'.join(z['decoded'] for z in proof['rounds'])
                    fmt=reward.r_format(trajectory)
                    total=reward.total_reward(trajectory,rank,lam=1.,mode='soft')
                    assert total==fmt+soft
                    x=dict(image=r['image'],row=r['row'],pair=r['pair'],block=r['block'],mode=mode,k=k,seed=proof['seed'],
                           native_tensor=proof['vector_sha256'],format=fmt,rank0=rank,soft=soft,reward=total,tools=proof['used'],
                           text=trajectory,query=proof['query'],final=proof['rounds'][-1]['decoded'])
                    for w in ('A','B'):
                        pos,neg=(float(sims[texts.index(s)]) for s in r[w])
                        x.update({w+'_pos':pos,w+'_neg':neg,w:pos-neg})
                    rows.append(x)
            print('scored',r['image'],flush=True)
    proxy.model.encode_image=native
    assert len(rows)==len(cases)*18
    bank=out/'vectors.pt'
    torch.save(dict(texts=texts,tokens=tokens,text=text_features,visual=torch.cat(visual),proxy=torch.cat(proxy_features),
                    order=[(r['image'],r['mode'],r['k']) for r in rows]),bank)
    dump(out/'rows.json',rows)
    sources=[SCORER,ASSETS/'ucm_main_domain_tool.ckpt',ASSETS/'models/ViT-L-14.pt',ASSETS/'ucm_main_lora/adapter_model.safetensors',
             ASSETS/'models/Qwen2.5-VL-7B-Instruct/generation_config.json',ASSETS/'ucm_scene_cls.pt',
             BASE/'CUP/agent/rollout.py',BASE/'CUP/agent/grpo/scorer.py',BASE/'CUP/agent/grpo/reward.py',BASE/'CUP/agent/tools.py',
             BASE/'CUP/vilt/modules/vilt_module.py',ARROW]
    dump(out/'binding.json',dict(assets={str(p):sha(p) for p in sources},lock=sha(ROOT/'inputs/lock.json'),bank=sha(bank),
                              rows=sha(out/'rows.json'),gallery=sha(gallery),device=torch.cuda.get_device_name(),torch=torch.__version__))
    print('DONE',len(rows),flush=True)
if __name__=='__main__':main()

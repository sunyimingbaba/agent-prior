"""Compute all gallery vectors once and read each stored actual policy prior in FP32."""
import sys,time
import torch
from shared import ROOT,setup,sha,rgb,dump,read
from data import DOMAINS,STATES,dataset
from readouts import load_model,clip

@torch.inference_mode()
def encode(reader,texts):
    tokens=clip.tokenize(texts,context_length=77-reader.prompt_length,truncate=True)
    unique,order=torch.unique(tokens,dim=0,return_inverse=True)
    feats=torch.cat([reader.txt_embeds(block)['text_feats'].cpu() for block in unique.split(16)])
    assert feats.isfinite().all() and (feats.norm(dim=1)-1).abs().max()<1e-4
    return dict(native=feats,caption_order=order,native_tokens=unique)

@torch.inference_mode()
def main():
    setup()
    reader=load_model(False)
    if len(sys.argv)>1 and sys.argv[1]=='attributes':
        cases=read(ROOT/'inputs/attributes.json')
        texts=sorted({r[k] for r in cases for k in ('positive','negative')})
        data=encode(reader,texts)
        torch.save(dict(texts=texts,features=data['native'][data['caption_order']],manifest=sha(ROOT/'inputs/attributes.json')),ROOT/'results/attribute_text.pt')
        return
    sets={}
    for dn in DOMAINS:
        t,cases=dataset(dn)
        sets[dn]=(t,cases)
        gallery=encode(reader,[q for c in cases for q in c['captions']])
        torch.save(gallery,ROOT/'results'/f'{dn}_text.pt')
    for state in STATES:
        for dn in DOMAINS:
            t,cases=sets[dn]
            folder=ROOT/'outputs'/state/dn
            result=ROOT/'results'/f'{dn}_{state}.pt'
            if result.exists():continue
            features=[];hashes=[];behavior=[]
            for i,c in enumerate(cases):
                f=folder/f"{c['row']:04d}.pt"
                while not f.is_file():time.sleep(1)
                obj=torch.load(f,map_location='cpu',weights_only=True)
                assert (obj['image'],obj['rgb'],obj['query'])==(c['image'],c['rgb_sha256'],c['orig_query'])
                pix=reader.preprocess(rgb(t,c))[None]
                v=reader.img_embeds(pix,agent_hidden=obj['prior'].float()[None])['image_feats'][0].cpu()
                assert v.isfinite().all() and abs(v.norm().item()-1)<1e-4
                hashes.append(sha(f)); features.append(v)
                behavior.append(dict(rounds=len(obj['rounds']),calls=len(obj['tools'])))
            while not (folder/'complete.json').is_file():time.sleep(1)
            proof=read(folder/'complete.json')
            assert hashes==proof['rows'] and obj['adapter']==proof['adapter']
            tmp=result.with_suffix('.part')
            torch.save(dict(features=torch.stack(features),rows=[c['row'] for c in cases],adapter=proof['adapter'],
                            manifest=sha(ROOT/'inputs'/f'{dn}.json'),behavior=behavior),tmp)
            tmp.replace(result)
            print(state,dn,len(features),flush=True)
if __name__=='__main__':main()

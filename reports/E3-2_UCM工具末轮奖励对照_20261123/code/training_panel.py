"""Check the terminal margin under fresh draws for the previous and current frozen policies."""
import csv
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch
from policy import policy
from train import sample,SAMPLES,NATIVE
from terminal_prior import finish_group,scene_tool
from readouts import load_model
from shared import ROOT,REBUILD,read,dump,sha,setup,table,rgb
from metrics import csvwrite

@torch.inference_mode()
def main(arm):
    assert arm in ('R0','R1','R2','old_R1','old_R2')
    setup()
    init=read(ROOT/'initial.json')
    first_root=REBUILD/'cold_start_ucm_20261123'
    previous=first_root if init['seed']==20261123 else REBUILD/'cold_start_ucm_repeat_20261123'
    path=previous/arm[4:] if arm.startswith('old_') else ROOT/arm
    model,proc=policy(path)
    weight=sha(path/'adapter/adapter_model.safetensors')
    expected=init['adapter'] if arm=='R0' else read(path/'complete.json')['final']
    assert weight==expected
    reader=load_model(False)
    classify=scene_tool()
    index=read(NATIVE/'index.json')
    assert sha(NATIVE/'features.pt')==index['sha256']
    bank=torch.load(NATIVE/'features.pt',weights_only=True,map_location='cpu')
    features=bank['features'].double()
    even=np.array([int(hashlib.sha256(','.join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in bank['tokens'].tolist()])
    t=table()
    output=[]
    if arm=='R0':
        with (first_root/'results/round_alignment_candidates.csv').open(encoding='utf-8-sig') as fp:
            reference={(z['image'],int(z['slot'])):float(z['final_margin']) for z in csv.DictReader(fp)}
    for i,row in enumerate(read(SAMPLES)):
        im=rgb(t,row)
        torch.manual_seed(721000+i)
        single,traces=sample(model,proc,im,row['scene_query'])
        states,logs=finish_group(model,proc,im,row['scene_query'],single,traces,classify)
        mapping=index['maps'][row['image']]
        assert mapping['captions']==t['caption'][row['row']].as_py()
        keep=even.copy()
        keep[mapping['positive']]=False
        pix=reader.preprocess(im)[None].cuda()
        for slot,(h,z) in enumerate(zip(states,logs,strict=True)):
            v=reader.img_embeds(pix,agent_hidden=h[None].cuda())['image_feats'].cpu().double()
            sims=(v@features.T)[0]
            margin=float(sims[mapping['first']]-sims[keep].max())
            if arm=='R0':
                assert abs(margin-reference[row['image'],slot])<1e-6
            output.append(dict(arm=arm,image=row['image'],pair=row['pair'],slot=slot,margin=margin,
                tools=z['tool_calls'],first_tokens=' '.join(map(str,traces[slot][0])),final_sha=z['final_sha']))
    out=ROOT/'panel'
    out.mkdir(exist_ok=True)
    f=out/(arm+'.csv')
    csvwrite(f,output)
    dump(out/(arm+'.json'),dict(weight=weight,sha256=sha(f),n=len(output),seed=721000))
    print(arm,len(output),flush=True)
if __name__=='__main__':
    main(sys.argv[1])

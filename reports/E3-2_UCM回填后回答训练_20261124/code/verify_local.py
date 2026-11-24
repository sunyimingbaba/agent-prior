"""Verify the archived checkpoints and each validation row; replay recall with the independent earlier implementation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

def digest(p):
    with open(p,'rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('archive',type=Path)
    parser.add_argument('--repeat',action='store_true')
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[3]
    folder=Path(__file__).resolve().parents[1]
    if args.repeat:
        folder=folder/'repeat'
    sys.path.insert(0,str(project/'reports/E3_UCM候选奖励与跨图复验_20261123'))
    from gallery_replay import per_source, feature_scores
    inputs=project/'reports/E3-2_UCM双视图方位奖励_20261120/inputs'
    gallery=torch.load(inputs/'retrieval_text.pt',weights_only=True,map_location='cpu')
    first=json.loads((folder/'records/initial.json').read_bytes())
    states=json.loads((folder/'results/states.json').read_bytes())
    output={}
    refs=args.archive/'R0/adapter/adapter_model.safetensors'
    for arm in ('R0','R1','R2','first_R2','old_R2'):
        path=args.archive/arm/'adapter/adapter_model.safetensors'
        weight=digest(path)
        expected=first['adapter'] if arm=='R0' else json.loads((folder/'records'/arm/'complete.json').read_bytes())['final']
        assert weight==expected
        with safe_open(path,framework='pt') as f, safe_open(refs,framework='pt') as init:
            keys=sorted(f.keys())
            assert keys==sorted(init.keys())
            nonzero_b=sum(bool(torch.any(f.get_tensor(k)!=0)) for k in keys if '.lora_B.' in k)
            changed=sum(not torch.equal(f.get_tensor(k),init.get_tensor(k)) for k in keys)
        assert arm=='R0' and nonzero_b==changed==0 or arm!='R0' and nonzero_b>0 and changed>0
        proof=json.loads((folder/'records'/arm/'validation.json').read_bytes())
        features=[]
        for i,h in enumerate(proof['rows']):
            row=args.archive/arm/'eval'/f'{i:03d}.pt'
            assert digest(row)==h
            z=torch.load(row,weights_only=True,map_location='cpu')
            assert z['row']==i and z['adapter']==weight
            features.append(z['feature'])
        assert len(features)==210
        scores=feature_scores(torch.stack(features).double().numpy(),gallery['native'].double().numpy())[:,gallery['caption_order']]
        vals,den=per_source(scores,np.ones(1050,dtype=bool),np.repeat(np.arange(210),5))
        rec=vals.sum(0)/den.sum(0)*100
        raw=states[arm]['retrieval_percent']
        expected_rec=[raw[k] for k in ('I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10')]
        err=float(max(abs(rec-expected_rec)))
        assert err<1e-8
        output[arm]=dict(rows=len(features),recall_difference_pp=err,adapter=weight,nonzero_B=nonzero_b,changed_tensors=changed)
    (folder/'results/archive_verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')

if __name__=='__main__':
    torch.set_num_threads(4)
    main()
